"""

C&W attack on Malimg dataset

"""
import os
from timeit import default_timer
import numpy as np
import matplotlib
matplotlib.use('Agg')           # noqa: E402
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import keras
import tensorflow as tf
import argparse

# from attacks import cw

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="Dataset Directiory path")
args = vars(ap.parse_args())
DATASET_PATH = args["dataset"]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


img_size = 32
img_chan = 1
n_classes = 25
batch_size = 32

def cw(model, x, y=None, eps=1.0, ord_=2, T=2,
       optimizer=tf.train.AdamOptimizer(learning_rate=0.1), alpha=0.9,
       min_prob=0, clip=(0.0, 1.0)):
    """CarliniWagner (CW) attack.

    Only CW-L2 and CW-Linf are implemented since I do not see the point of
    embedding CW-L2 in CW-L1.  See https://arxiv.org/abs/1608.04644 for
    details.

    The idea of CW attack is to minimize a loss that comprises two parts: a)
    the p-norm distance between the original image and the adversarial image,
    and b) a term that encourages the incorrect classification of the
    adversarial images.

    Please note that CW is a optimization process, so it is tricky.  There are
    lots of hyper-parameters to tune in order to get the best result.  The
    binary search process for the best eps values is omitted here.  You could
    do grid search to find the best parameter configuration, if you like.  I
    demonstrate binary search for the best result in an example code.

    :param model: The model wrapper.
    :param x: The input clean sample, usually a placeholder.  NOTE that the
              shape of x MUST be static, i.e., fixed when constructing the
              graph.  This is because there are some variables that depends
              upon this shape.
    :param y: The target label.  Set to be the least-likely label when None.
    :param eps: The scaling factor for the second penalty term.
    :param ord_: The p-norm, 2 or inf.  Actually I only test whether it is 2
        or not 2.
    :param T: The temperature for sigmoid function.  In the original paper,
              the author used (tanh(x)+1)/2 = sigmoid(2x), i.e., t=2.  During
              our experiment, we found that this parameter also affects the
              quality of generated adversarial samples.
    :param optimizer: The optimizer used to minimize the CW loss.  Default to
        be tf.AdamOptimizer with learning rate 0.1. Note the learning rate is
        much larger than normal learning rate.
    :param alpha: Used only in CW-L0.  The decreasing factor for the upper
        bound of noise.
    :param min_prob: The minimum confidence of adversarial examples.
        Generally larger min_prob wil lresult in more noise.
    :param clip: A tuple (clip_min, clip_max), which denotes the range of
        values in x.

    :return: A tuple (train_op, xadv, noise).  Run train_op for some epochs to
             generate the adversarial image, then run xadv to get the final
             adversarial image.  Noise is in the sigmoid-space instead of the
             input space.  It is returned because we need to clear noise
             before each batched attacks.
    """
    xshape = x.get_shape().as_list()
    noise = tf.get_variable('noise', xshape, tf.float32,
                            initializer=tf.initializers.zeros)

    # scale input to (0, 1)
    x_scaled = (x - clip[0]) / (clip[1] - clip[0])

    # change to sigmoid-space, clip to avoid overflow.
    z = tf.clip_by_value(x_scaled, 1e-8, 1-1e-8)
    xinv = tf.log(z / (1 - z)) / T

    # add noise in sigmoid-space and map back to input domain
    xadv = tf.sigmoid(T * (xinv + noise))
    xadv = xadv * (clip[1] - clip[0]) + clip[0]

    ybar, logits = model(xadv, logits=True)
    ydim = ybar.get_shape().as_list()[1]

    if y is not None:
        y = tf.cond(tf.equal(tf.rank(y), 0),
                    lambda: tf.fill([xshape[0]], y),
                    lambda: tf.identity(y))
    else:
        # we set target to the least-likely label
        y = tf.argmin(ybar, axis=1, output_type=tf.int32)

    mask = tf.one_hot(y, ydim, on_value=0.0, off_value=float('inf'))
    yt = tf.reduce_max(logits - mask, axis=1)
    yo = tf.reduce_max(logits, axis=1)

    # encourage to classify to a wrong category
    loss0 = tf.nn.relu(yo - yt + min_prob)

    axis = list(range(1, len(xshape)))
    ord_ = float(ord_)

    # make sure the adversarial images are visually close
    if 2 == ord_:
        # CW-L2 Original paper uses the reduce_sum version.  These two
        # implementation does not differ much.

        # loss1 = tf.reduce_sum(tf.square(xadv-x), axis=axis)
        loss1 = tf.reduce_mean(tf.square(xadv-x))
    else:
        # CW-Linf
        tau0 = tf.fill([xshape[0]] + [1]*len(axis), clip[1])
        tau = tf.get_variable('cw8-noise-upperbound', dtype=tf.float32,
                              initializer=tau0, trainable=False)
        diff = xadv - x - tau

        # if all values are smaller than the upper bound value tau, we reduce
        # this value via tau*0.9 to make sure L-inf does not get stuck.
        tau = alpha * tf.to_float(tf.reduce_all(diff < 0, axis=axis))
        loss1 = tf.nn.relu(tf.reduce_sum(diff, axis=axis))

    loss = eps*loss0 + loss1
    train_op = optimizer.minimize(loss, var_list=[noise])

    # We may need to update tau after each iteration.  Refer to the CW-Linf
    # section in the original paper.
    if 2 != ord_:
        train_op = tf.group(train_op, tau)

    return train_op, xadv, noise

class Timer(object):
    def __init__(self, msg='Starting.....', timer=default_timer, factor=1,
                 fmt="------- elapsed {:.4f}s --------"):
        self.timer = timer
        self.factor = factor
        self.fmt = fmt
        self.end = None
        self.msg = msg

    def __call__(self):
        """
        Return the current time
        """
        return self.timer()

    def __enter__(self):
        """
        Set the start time
        """
        print(self.msg)
        self.start = self()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Set the end time
        """
        self.end = self()
        print(str(self))

    def __repr__(self):
        return self.fmt.format(self.elapsed)

    @property
    def elapsed(self):
        if self.end is None:
            # if elapsed is called in the context manager scope
            return (self() - self.start) * self.factor
        else:
            # if elapsed is called out of the context manager scope
            return (self.end - self.start) * self.factor


print('\nLoading Malimg')

def load_data():
    dataset = np.load(DATASET_PATH)
    features = dataset['arr'][:, 0]
    features = np.array([feature for feature in features])
    features = np.reshape(features, (features.shape[0], features.shape[1] * features.shape[2]))

    # if standardize:
    #     features = StandardScaler().fit_transform(features)

    labels = dataset['arr'][:, 1]
    labels = np.array([label for label in labels])

    x_test = []
    y_test = []

    for i in range(len(features)):
        x_test.append(np.reshape(features[i], [32, 32, 1]))
        y_test.append(labels[i])

    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    xTrain1 = x_test[0:1500]
    xTrain2 = x_test[1591:4091]
    xTrain3 = x_test[4540:5040]
    xTrain4 = x_test[5340:5453]
    xTrain5 = x_test[5553:5703]
    xTrain6 = x_test[5737:5797]
    xTrain7 = x_test[5860:5930]
    xTrain8 = x_test[6006:6106]
    xTrain9 = x_test[6206:6506]
    xTrain10 = x_test[6637:6707]
    xTrain11 = x_test[6769:6837]
    xTrain12 = x_test[6897:7197]
    xTrain13 = x_test[7305:7605]
    xTrain14 = x_test[7686:7836]
    xTrain15 = x_test[7884:7954]
    xTrain16 = x_test[8020:8120]
    xTrain17 = x_test[8179:8244]
    xTrain18 = x_test[8304:8354]
    xTrain19 = x_test[8401:8501]
    xTrain20 = x_test[8578:8668]
    xTrain21 = x_test[8740:8810]
    xTrain22 = x_test[8882:8942]
    xTrain23 = x_test[8998:9078]
    xTrain24 = x_test[9104:9204]
    xTrain25 = x_test[9262:9312]

    x_train = list()
    x_train = np.concatenate((xTrain1, xTrain2, xTrain3,xTrain4,xTrain5,
                         xTrain6,xTrain7,xTrain8,xTrain9,xTrain10,
                         xTrain11,xTrain12,xTrain13,xTrain14,xTrain15,
                         xTrain16,xTrain17,xTrain18,xTrain19,xTrain20,
                         xTrain21,xTrain22,xTrain23,xTrain24,xTrain25), axis=0)
    print(len(x_train))

    yTrain1 = y_test[0:1500]
    yTrain2 = y_test[1591:4091]
    yTrain3 = y_test[4540:5040]
    yTrain4 = y_test[5340:5453]
    yTrain5 = y_test[5553:5703]
    yTrain6 = y_test[5737:5797]
    yTrain7 = y_test[5860:5930]
    yTrain8 = y_test[6006:6106]
    yTrain9 = y_test[6206:6506]
    yTrain10 = y_test[6637:6707]
    yTrain11 = y_test[6769:6837]
    yTrain12 = y_test[6897:7197]
    yTrain13 = y_test[7305:7605]
    yTrain14 = y_test[7686:7836]
    yTrain15 = y_test[7884:7954]
    yTrain16 = y_test[8020:8120]
    yTrain17 = y_test[8179:8244]
    yTrain18 = y_test[8304:8354]
    yTrain19 = y_test[8401:8501]
    yTrain20 = y_test[8578:8668]
    yTrain21 = y_test[8740:8810]
    yTrain22 = y_test[8882:8942]
    yTrain23 = y_test[8998:9078]
    yTrain24 = y_test[9104:9204]
    yTrain25 = y_test[9262:9312]

    y_train = np.concatenate((yTrain1,yTrain2,yTrain3,yTrain4,yTrain5,
                         yTrain6,yTrain7,yTrain8,yTrain9,yTrain10,
                         yTrain11,yTrain12,yTrain13,yTrain14,yTrain15,
                         yTrain16,yTrain17,yTrain18,yTrain19,yTrain20,
                         yTrain21,yTrain22,yTrain23,yTrain24,yTrain25),axis = 0)

    xTest1 = x_test[1500:1591]
    xTest2 = x_test[4091:4540]
    xTest3 = x_test[5040:5340]
    xTest4 = x_test[5453:5553]
    xTest5 = x_test[5703:5737]
    xTest6 = x_test[5797:5860]
    xTest7 = x_test[5930:6006]
    xTest8 = x_test[6106:6206]
    xTest9 = x_test[6506:6637]
    xTest10 = x_test[6707:6769]
    xTest11 = x_test[6837:6897]
    xTest12 = x_test[7197:7305]
    xTest13 = x_test[7605:7686]
    xTest14 = x_test[7836:7884]
    xTest15 = x_test[7954:8020]
    xTest16 = x_test[8120:8179]
    xTest17 = x_test[8244:8304]
    xTest18 = x_test[8354:8401]
    xTest19 = x_test[8501:8578]
    xTest20 = x_test[8668:8740]
    xTest21 = x_test[8810:8882]
    xTest22 = x_test[8942:8998]
    xTest23 = x_test[9078:9104]
    xTest24 = x_test[9204:9262]
    xTest25 = x_test[9312:9342]

    x_test = np.concatenate((xTest1, xTest2, xTest3, xTest4, xTest5,
                        xTest6, xTest7, xTest8, xTest9, xTest10,
                        xTest11, xTest12, xTest13, xTest14, xTest15,
                        xTest16, xTest17, xTest18, xTest19, xTest20,
                        xTest21, xTest22, xTest23, xTest24, xTest25), axis=0)

    yTest1 = y_test[1500:1591]
    yTest2 = y_test[4091:4540]
    yTest3 = y_test[5040:5340]
    yTest4 = y_test[5453:5553]
    yTest5 = y_test[5703:5737]
    yTest6 = y_test[5797:5860]
    yTest7 = y_test[5930:6006]
    yTest8 = y_test[6106:6206]
    yTest9 = y_test[6506:6637]
    yTest10 = y_test[6707:6769]
    yTest11 = y_test[6837:6897]
    yTest12 = y_test[7197:7305]
    yTest13 = y_test[7605:7686]
    yTest14 = y_test[7836:7884]
    yTest15 = y_test[7954:8020]
    yTest16 = y_test[8120:8179]
    yTest17 = y_test[8244:8304]
    yTest18 = y_test[8354:8401]
    yTest19 = y_test[8501:8578]
    yTest20 = y_test[8668:8740]
    yTest21 = y_test[8810:8882]
    yTest22 = y_test[8942:8998]
    yTest23 = y_test[9078:9104]
    yTest24 = y_test[9204:9262]
    yTest25 = y_test[9312:9342]

    y_test = np.concatenate((yTest1, yTest2, yTest3, yTest4,yTest5,
                        yTest6, yTest7, yTest8, yTest9, yTest10,
                        yTest11, yTest12, yTest13, yTest14, yTest15,
                        yTest16, yTest17, yTest18, yTest19, yTest20,
                        yTest21, yTest22, yTest23, yTest24, yTest25), axis=0)

    return x_train, y_train, x_test, y_test

X_train, y_train,X_test, y_test = load_data()
X_train = np.reshape(X_train, [-1, img_size, img_size, img_chan])
X_train = X_train.astype(np.float32) / 255
X_test = np.reshape(X_test, [-1, img_size, img_size, img_chan])
X_test = X_test.astype(np.float32) / 255

to_categorical = keras.utils.to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('\nSpliting data')

ind = np.random.permutation(X_train.shape[0])
X_train, y_train = X_train[ind], y_train[ind]

VALIDATION_SPLIT = 0.1
n = int(X_train.shape[0] * (1-VALIDATION_SPLIT))
X_valid = X_train[n:]
X_train = X_train[:n]
y_valid = y_train[n:]
y_train = y_train[:n]

print('\nConstruction graph')


def model(x, logits=False, training=False):
    with tf.variable_scope('conv0'):
        z = tf.layers.conv2d(x, filters=32, kernel_size=[5, 5],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=1)

    with tf.variable_scope('conv1'):
        z = tf.layers.conv2d(z, filters=64, kernel_size=[5,5],
                             padding='same', activation=tf.nn.relu)
        z = tf.layers.max_pooling2d(z, pool_size=[2, 2], strides=1)

    with tf.variable_scope('flatten'):
        shape = z.get_shape().as_list()
        z = tf.reshape(z, [-1, np.prod(shape[1:])])

    with tf.variable_scope('mlp'):
        z = tf.layers.dense(z, units=1024, activation=tf.nn.relu)
        z = tf.layers.dropout(z, rate=0.5, training=training)

    logits_ = tf.layers.dense(z, units=25, name='logits')
    y = tf.nn.softmax(logits_, name='ybar')

    if logits:
        return y, logits_
    return y


class Dummy:
    pass


env = Dummy()


with tf.variable_scope('model', reuse = tf.AUTO_REUSE):
    env.x = tf.placeholder(tf.float32, (None, img_size, img_size, img_chan),
                           name='x')
    env.y = tf.placeholder(tf.float32, (None, n_classes), name='y')
    env.training = tf.placeholder_with_default(False, (), name='mode')

    env.ybar, logits = model(env.x, logits=True, training=env.training)

    with tf.variable_scope('acc'):
        count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
        env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

    with tf.variable_scope('loss'):
        xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                       logits=logits)
        env.loss = tf.reduce_mean(xent, name='loss')

    with tf.variable_scope('train_op'):
        optimizer = tf.train.AdamOptimizer()
        vs = tf.global_variables()
        env.train_op = optimizer.minimize(env.loss, var_list=vs)

    env.saver = tf.train.Saver()

    # Note here that the shape has to be fixed during the graph construction
    # since the internal variable depends upon the shape.
    env.x_fixed = tf.placeholder(
        tf.float32, (batch_size, img_size, img_size, img_chan),
        name='x_fixed')
    env.adv_eps = tf.placeholder(tf.float32, (), name='adv_eps')
    env.adv_y = tf.placeholder(tf.int32, (), name='adv_y')

    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    env.adv_train_op, env.xadv, env.noise = cw(model, env.x_fixed,
                                               y=env.adv_y, eps=env.adv_eps,
                                               optimizer=optimizer)

print('\nInitializing graph')

env.sess = tf.InteractiveSession()
env.sess.run(tf.global_variables_initializer())
env.sess.run(tf.local_variables_initializer())


def evaluate(env, X_data, y_data, batch_size=128):
    """
    Evaluate TF model by running env.loss and env.acc.
    """
    print('\nEvaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = env.sess.run(
            [env.loss, env.acc],
            feed_dict={env.x: X_data[start:end],
                       env.y: y_data[start:end]})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
    return loss, acc


def train(env, X_data, y_data, X_valid=None, y_valid=None, epochs=1,
          load=False, shuffle=True, batch_size=128, name='model'):
    """
    Train a TF model by running env.train_op.
    """
    if load:
        if not hasattr(env, 'saver'):
            return print('\nError: cannot find saver op')
        print('\nLoading saved model')
        return env.saver.restore(env.sess, 'model/{}'.format(name))

    print('\nTrain model')
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    for epoch in range(epochs):
        print('\nEpoch {0}/{1}'.format(epoch + 1, epochs))

        if shuffle:
            print('\nShuffling data')
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X_data = X_data[ind]
            y_data = y_data[ind]

        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            env.sess.run(env.train_op, feed_dict={env.x: X_data[start:end],
                                                  env.y: y_data[start:end],
                                                  env.training: True})
        if X_valid is not None:
            evaluate(env, X_valid, y_valid)

    if hasattr(env, 'saver'):
        print('\n Saving model')
        os.makedirs('model', exist_ok=True)
        env.saver.save(env.sess, 'model/{}'.format(name))


def predict(env, X_data, batch_size=128):
    """
    Do inference by running env.ybar.
    """
    print('\nPredicting')
    n_classes = env.ybar.get_shape().as_list()[1]

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        y_batch = env.sess.run(env.ybar, feed_dict={env.x: X_data[start:end]})
        yval[start:end] = y_batch
    print()
    return yval


def make_cw(env, X_data, epochs=1, eps=0.1, batch_size=batch_size):
    """
    Generate adversarial via CW optimization.
    """
    print('\nMaking adversarials via CW')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample + batch_size - 1) / batch_size)
    X_adv = np.empty_like(X_data)

    for batch in range(n_batch):
        with Timer('Batch {0}/{1}   '.format(batch + 1, n_batch)):
            end = min(n_sample, (batch+1) * batch_size)
            start = end - batch_size
            feed_dict = {
                env.x_fixed: X_data[start:end],
                env.adv_eps: eps,
                env.adv_y: np.random.choice(n_classes)}

            # reset the noise before every iteration
            env.sess.run(env.noise.initializer)
            for epoch in range(epochs):
                env.sess.run(env.adv_train_op, feed_dict=feed_dict)

            xadv = env.sess.run(env.xadv, feed_dict=feed_dict)
            X_adv[start:end] = xadv

    return X_adv


print('\nTraining')

train(env, X_train, y_train, X_valid, y_valid, load=False, epochs=5,
      name='mnist')

print('\nEvaluating on clean data')

evaluate(env, X_test, y_test)

print('\nGenerating adversarial data')

# It takes a while to run through the full dataset, thus, we demo the result
# through a smaller dataset.  We could actually find the best parameter
# configuration on a smaller dataset, and then apply to the full dataset.
n_sample = 128
ind = np.random.choice(X_test.shape[0], size=n_sample, replace=False)
X_test = X_test[ind]
y_test = y_test[ind]

X_adv = make_cw(env, X_test, eps=0.002, epochs=100)

print('\nEvaluating on adversarial data')

evaluate(env, X_adv, y_test)

print('\nRandomly sample adversarial data from each category')

y1 = predict(env, X_test)
y2 = predict(env, X_adv)

z0 = np.argmax(y_test, axis=1)
z1 = np.argmax(y1, axis=1)
z2 = np.argmax(y2, axis=1)

ind = np.logical_and(z0 == z1, z1 != z2)
# print('success: ', np.sum(ind))

ind = z0 == z1

X_test = X_test[ind]
X_adv = X_adv[ind]
z1 = z1[ind]
z2 = z2[ind]
y2 = y2[ind]

ind, = np.where(z1 != z2)
cur = np.random.choice(ind, size=n_classes)
X_org = np.squeeze(X_test[cur])
X_tmp = np.squeeze(X_adv[cur])
y_tmp = y2[cur]

fig = plt.figure(figsize=(n_classes+0.2, 3.2))
gs = gridspec.GridSpec(3, n_classes+1, width_ratios=[1]*n_classes + [0.1],
                       wspace=0.01, hspace=0.01)

label = np.argmax(y_tmp, axis=1)
proba = np.max(y_tmp, axis=1)

for i in range(n_classes):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(X_org[i], cmap='gray', interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[1, i])
    img = ax.imshow(X_tmp[i]-X_org[i], cmap='RdBu_r', vmin=-1,
                    vmax=1, interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[2, i])
    ax.imshow(X_tmp[i], cmap='gray', interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlabel('{0} ({1:.2f})'.format(label[i], proba[i]), fontsize=12)

ax = fig.add_subplot(gs[1, n_classes])
dummy = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=-1,
                                                                vmax=1))
dummy.set_array([])
fig.colorbar(mappable=dummy, cax=ax, ticks=[-1, 0, 1], ticklocation='right')

print('\nSaving figure')

gs.tight_layout(fig)
os.makedirs('img', exist_ok=True)
plt.savefig('img/adverserial_cw.png')
