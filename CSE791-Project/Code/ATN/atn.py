"""
ATN attack(Dataset - malimg)
"""

import tensorflow as tf
import numpy as np
import ATN.atn_model as atn
# from skimage import img_as_ubyte
# from keras.datasets import cifar10
import os
# from tensorflow.examples.tutorials.mnist import input_data
# import matplotlib.pyplot as plt
# import cv2
import argparse

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_boolean('train', False, 'Train and save the ATN model.')

# Placeholder nodes.

# images_holder = tf.placeholder(tf.float32, [None, 784])
# label_holder = tf.placeholder(tf.float32, [None, 10])
images_holder = tf.placeholder(tf.float32, [None, 32, 32, 1], name='X')
label_holder = tf.placeholder(tf.int64, [None], name='Y')
p_keep_holder = tf.placeholder(tf.float32)
rerank_holder = tf.placeholder(tf.float32, [None, 25])


# mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)


def main(arvg=None):
    """
        """
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="Dataset Directiory path")
    ap.add_argument("-train","--train", required=False,
                    help="for training")

    args = vars(ap.parse_args())
    DATASET_PATH = args["dataset"]
    xTn,yTn,xTt,yTt = load_data(DATASET_PATH)

    FLAGS.train = args["train"]
    if FLAGS.train:
        train(xTn, yTn, xTt, yTt)
    else:
        test(xTt, yTt)


def load_data(DATASET_PATH):
    print('loading data')
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

    x_train = tf.concat([xTrain1,xTrain2,xTrain3,xTrain4,xTrain5,
                         xTrain6,xTrain7,xTrain8,xTrain9,xTrain10,
                         xTrain11,xTrain12,xTrain13,xTrain14,xTrain15,
                         xTrain16,xTrain17,xTrain18,xTrain19,xTrain20,
                         xTrain21,xTrain22,xTrain23,xTrain24,xTrain25],0)

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

    y_train = tf.concat([yTrain1,yTrain2,yTrain3,yTrain4,yTrain5,
                         yTrain6,yTrain7,yTrain8,yTrain9,yTrain10,
                         yTrain11,yTrain12,yTrain13,yTrain14,yTrain15,
                         yTrain16,yTrain17,yTrain18,yTrain19,yTrain20,
                         yTrain21,yTrain22,yTrain23,yTrain24,yTrain25],0)

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
    x_test = tf.concat([xTest1, xTest2, xTest3, xTest4, xTest5,
                        xTest6, xTest7, xTest8, xTest9, xTest10,
                        xTest11, xTest12, xTest13, xTest14, xTest15,
                        xTest16, xTest17, xTest18, xTest19, xTest20,
                        xTest21, xTest22, xTest23, xTest24, xTest25], 0)

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

    y_test = tf.concat([yTest1, yTest2, yTest3, yTest4,yTest5,
                        yTest6, yTest7, yTest8, yTest9, yTest10,
                        yTest11, yTest12, yTest13, yTest14, yTest15,
                        yTest16, yTest17, yTest18, yTest19, yTest20,
                        yTest21, yTest22, yTest23, yTest24, yTest25], 0)

    print('shape of xtrain :',np.shape(x_train))
    print('shape of ytrain :', np.shape(y_train))
    print('shape of xtest :',np.shape(x_test))
    print('shape of ytest :', np.shape(y_test))
    return x_train, y_train, x_test, y_test

def test(xTt, yTt):
    """
        """
    print('testing ')
    label_holder2 = tf.one_hot(label_holder, 25)
    model = atn.ATN(images_holder, label_holder2, p_keep_holder, rerank_holder)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        model.load(sess, './ATN/Models/AE_for_ATN')
        yTt_onehot = tf.one_hot(yTt,25)
        adv_images = sess.run(
                              model.prediction,
                              feed_dict={images_holder: xTt.eval()}
                              )

        print('shape of xTest :',np.shape(xTt),' ; shape of yTest :',np.shape(yTt))
        print('shape of adv_images :',np.shape(adv_images))

        adv_imgs = adv_images.reshape(adv_images.shape[0], 32, 32,1)
        adv_imgs1 = adv_images.reshape(adv_images.shape[0], 32, 32)
        print('after reshaping , shape of adv_imgs :', np.shape(adv_imgs))
        # _, cnn_acc = sess.run([model._target.optimization, model._target.accuracy], feed_dict={images_holder: xTt.eval(),
        #                                                                                        label_holder2: yTt_onehot.eval(),
        #                                                                                        p_keep_holder: 1.0})
        # print('cnn optimization : ', _)
        # print('cnn accuracy : ', cnn_acc)



        print('Original accuracy: {0:0.5f}'.format(sess.run(model._target.accuracy,
                                                            feed_dict={images_holder: xTt.eval(),label_holder2: yTt_onehot.eval(),p_keep_holder: 1.0})))
        print('Attacked accuracy: {0:0.5f}'.format(sess.run(model._target.accuracy,
                                                                feed_dict={images_holder: adv_imgs,label_holder2: yTt_onehot.eval(),p_keep_holder: 1.0})))



def train(xTrain, yTrain, xTest, yTest):
    print('training ')
    attack_target = 8
    alpha = 1.5
    training_epochs = 3
    batch_size = 100
    total_batch = 160
    label_holder1 = tf.one_hot(label_holder, 25)
    model = atn.ATN(images_holder, label_holder1, p_keep_holder, rerank_holder)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model._target.load(sess, './ATN/Models/AE_for_ATN/BasicCNN')
        for epoch in range(training_epochs):
            for in1 in range(total_batch):
                # np.random.shuffle(s)
                # xTr = xTrain[s]
                # yTr = yTrain[s]
                xTrain = tf.random_shuffle(xTrain, seed=8)
                yTrain = tf.random_shuffle(yTrain, seed=8)
                batch_xs = xTrain[:20]
                batch_ys = yTrain[:20]
                batch1_ys = tf.one_hot(batch_ys,25)
                # loss, _, train_acc = sess.run([model._target.loss, model._target.optimization, model._target.accuracy],
                #                       feed_dict={images_holder: batch_xs.eval(),label_holder1: batch1_ys.eval(),p_keep_holder: 1.0})
                # test_loss, _1, test_acc = sess.run([model._target.loss,model._target.optimization, model._target.accuracy],
                #                       feed_dict={images_holder: xTest.eval(),
                #                                  label_holder1: tf.one_hot(yTest,25).eval(),
                #                                  p_keep_holder: 1.0})
                #
                # print('iterration : ',i,'epoch : ', epoch, '; cnn accuracy : ', train_acc, ' ; cnn loss :', loss, ' ; test accuracy : ',
                #       test_acc, ' ; test loss : ', test_loss)

                r_res = sess.run(model._target.prediction,feed_dict={images_holder: batch_xs.eval(),p_keep_holder: 1.0})
                print(' r_res = sess.run(model._target.prediction,feed_dict={images_holder: batch_xs.eval(),p_keep_holder: 1.0})')
                r_res[:, attack_target] = np.max(r_res, axis=1) * alpha
                print('r_res[:, attack_target] = np.max(r_res, axis=1) * alpha executed')
                norm_div = np.linalg.norm(r_res, axis=1)
                print('norm_div = np.linalg.norm(r_res, axis=1) & len(r_res) = ', len(r_res))
                for i in range(len(r_res)):
                    r_res[i] /= norm_div[i]
                    print('i : ',i)

                    _, loss = sess.run(model.optimization, feed_dict={images_holder: batch_xs.eval(),p_keep_holder: 1.0,rerank_holder: r_res})
                print('loss calculated : ',loss,'for iterration : ',in1 , ' ; for eopch : ', epoch)

        print("Optimization Finished!")

        model.save(sess, './ATN/Models/AE_for_ATN')
    print("Trained params have been saved to './ATN/Models/AE_for_ATN'")



if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.app.run()



