import tensorflow as tf
from keras.datasets import cifar10
import numpy as np
import os
import matplotlib as mtlb
import cv2
import sys

def train():
    # Import data (Samples, 28, 28, 1)
    learning_rate= 0.0000003
    (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
    xTrain = np.reshape(xTrain, (xTrain.shape[0], -1))
    xTest = np.reshape(xTest, (xTest.shape[0], -1))
    yTrain = np.squeeze(yTrain)
    yTest = np.squeeze(yTest)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 32*32*3],name='x')
    y_ = tf.placeholder(tf.int64, [None],name='y_')

    # Variables
    W1 = tf.Variable(tf.zeros([3072, 10]),name="W1")
    b1 = tf.Variable(tf.zeros([10]),name="b1")

    y = tf.matmul(x, W1)
    y= tf.add(y,b1,name='y')
    # Define loss and optimizer
    cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(y_, 10), logits=y))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    for i in range(6000):
        s = np.arange(xTrain.shape[0])
        np.random.shuffle(s)
        xTr = xTrain[s]
        yTr = yTrain[s]
        batch_xs = xTr[:1024]
        batch_ys = yTr[:1024]
        loss, _ ,trainacc= sess.run([cross_entropy, train_step,accuracy], feed_dict={x: batch_xs, y_: batch_ys})
        TestLoss,testacc=sess.run([cross_entropy,accuracy],feed_dict={x: xTest,y_: yTest})
        if(i%100==0):

            print('Iteration {:5d}: trainLoss {:g}  trainAccuracy {:5f} testLoss {:g} testAccuracy {:5f} '.format(i, loss,trainacc*100,TestLoss,testacc*100))


    saver = tf.train.Saver()
    save_path = saver.save(sess, "model/model.ckpt")
    print("Testing accuracy :", sess.run(accuracy, feed_dict={x: xTest, y_: yTest})*100)
    print("**********MODEL SAVED*********************************")






def test(image):
    classesName = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
    xTrain = np.reshape(xTrain, (xTrain.shape[0], -1))
    xTest = np.reshape(xTest, (xTest.shape[0], -1))
    yTrain = np.squeeze(yTrain)
    yTest = np.squeeze(yTest)
    sess = tf.InteractiveSession()

    img = cv2.imread(image, 1)
    res = cv2.resize(img, (32, 32))
    res = np.array(res).reshape(1, 32, 32, 3)
    res = np.reshape(res, (res.shape[0], -1))

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./model/model.ckpt.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./model/'))

        graph = tf.get_default_graph()
        y_ = graph.get_tensor_by_name('y_:0')
        x = graph.get_tensor_by_name('x:0')
        y = graph.get_tensor_by_name('y:0')
        s1 = np.arange(res.shape[0])
        np.random.shuffle(s1)
        xTe = res[s1]
        yTe = yTest[s1]
        #print(sess.run([],feed_dict={x: res, y_: yTe}))
        prediction = tf.argmax(y, 1)
        answer = prediction.eval(session=sess,feed_dict={x: res})
        print(classesName[answer[0]])




if len(sys.argv) == 2:
    if sys.argv[1] == "train":
        train()

if len(sys.argv) == 3:
    if sys.argv[1] == "test":
        image = sys.argv[2]
        test(image)

