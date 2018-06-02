import tensorflow as tf
from keras.datasets import cifar10
import numpy as np
import os
import cv2
import sys
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def train():

    learning_rate= 4e-3
    (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
    xTrain = xTrain[:49000,:].astype(np.float)
    xTest = xTest.astype(np.float)
    yTrain = np.squeeze(yTrain)
    yTest = np.squeeze(yTest)

    #normalization of the data i.e zero centric data
    meanData=np.mean(xTrain,axis=0)
    xTrain -= meanData
    xTest -= meanData

    # Create the model and running on a gpu

    x = tf.placeholder(tf.float32, [None, 32,32,3],name='input_x')
    y_ = tf.placeholder(tf.int64, [None],name='input_y')

    # weights and biases
    #layer 1
    W1 = tf.get_variable("conv_layer1_W", shape=[5, 5, 3, 32])  # shape= filter size + number of filters
    b1 = tf.get_variable("conv_layer1_b", shape=[32])  # bias for the filters
    layer1=tf.nn.conv2d(x,W1,strides=[1,1,1,1],padding="VALID")
    layer1+=tf.nn.bias_add(layer1,b1,name='conv_layer1_out')
    relu1=tf.nn.relu(layer1) #given stride =1
    maxpool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID') #pooling layer with stride 2

    #layer 2
    W2 = tf.get_variable("conv_layer2_W", shape=[3,3,32,64])
    b2 = tf.get_variable("conv_layer2_b", shape=[64])
    layer2 = tf.nn.conv2d(maxpool1,W2,strides=[1,1,1,1],padding="VALID")
    layer2+=tf.nn.bias_add(layer2,b2,name='conv_layer2_out')
    relu2 = tf.nn.relu(layer2)
    maxpool2=tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")

    #classification
    W4 = tf.get_variable("W4", shape=[2304, 10])
    b4 = tf.get_variable("b4", shape=[10])
    hflat1=tf.reshape(maxpool2,[-1,2304])
    y = tf.matmul(hflat1, W4)
    y=tf.add(y,b4,name='y')


    # Define loss and optimizer
    cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(y_, 10), logits=y),name='output_y')
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    for i in range(2000):
        s = np.arange(xTrain.shape[0])
        np.random.shuffle(s)
        xTr = xTrain[s]
        yTr = yTrain[s]
        batch_xs = xTr[:1024]
        batch_ys = yTr[:1024]
        loss, _ ,trainacc= sess.run([cross_entropy, train_step,accuracy], feed_dict={x: batch_xs, y_: batch_ys})
        TestLoss,testacc=sess.run([cross_entropy,accuracy],feed_dict={x: xTest,y_: yTest})
        if(i%10==0):

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
    sess = tf.Session()
    saver = tf.train.import_meta_graph('./model/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./model/'))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('input_x:0')
    y = graph.get_tensor_by_name('output_y:0')
    conv_layer = graph.get_tensor_by_name('conv_layer1_out:0')
    prediction = tf.argmax(y, 1)
    answer = sess.run(prediction, feed_dict={x: res})
    print(classesName[answer[0]])

    classification = sess.run(conv_layer, feed_dict={x: res})
    plotNNFilter(classification)

def plotNNFilter(units):
    filters=units.shape[3]
    plt.figure(1,figsize=(28,28))
    n_columns=6
    n_rows=math.ceil(filters/n_columns)+1
    for i in range(filters):
        plt.subplot(n_rows,n_columns,i+1)
        plt.imshow(units[0,:,:,i])
    plt.show()
    plt.savefig('CONV_rslt.png')


if(sys.argv[1]=='train'or'test'):
    if len(sys.argv)==2:
         if sys.argv[1]=="train":
             train()


    if len(sys.argv)==3:
        if(sys.argv[1]=='test'):
            image=sys.argv[2]
            test(image)

else:
    raise Exception("The specified method"+ sys.argv[1]+ "has not been implemented ")




