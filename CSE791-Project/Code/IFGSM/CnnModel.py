import tensorflow as tf
import matplotlib
import numpy as np
from numpy import newaxis
import sys
import os
import math
import matplotlib as mp
import matplotlib.pyplot as plt
import argparse

def load_data():
    print("dataset loading - ", DATASET_PATH)
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


def train_model_on_malImg():
    learning_rate = 5e-4
    epochs = 500
    batch_size = 256

    # Loading cifar10 dataset.
    xTrain, yTrain, xTest, yTest = load_data()

    yTrain = np.squeeze(yTrain)
    yTest = np.squeeze(yTest)

    # Create model
    x = tf.placeholder(tf.float32, [None, 32, 32, 1], 'x')
    y = tf.placeholder(tf.int64, [None], 'expected_output')

    conv_layer1 = tf.layers.conv2d(inputs=x, filters=32, padding='same', kernel_size=5, strides=1, activation=tf.nn.relu)
    max_pooling_layer1 = tf.layers.max_pooling2d(inputs=conv_layer1, pool_size=[2, 2], strides=2)

    conv_layer2 = tf.layers.conv2d(inputs=max_pooling_layer1, filters=64, padding='same', kernel_size=5, strides=1, activation=tf.nn.relu)
    max_pooling_layer2 = tf.layers.max_pooling2d(inputs=conv_layer2, pool_size=[2, 2], strides=2)

    conv_layer3 = tf.layers.conv2d(inputs=max_pooling_layer2, filters=64, padding='same', kernel_size=5, strides=1, activation=tf.nn.relu)
    max_pooling_layer3 = tf.layers.max_pooling2d(inputs=conv_layer3, pool_size=[2, 2], strides=2)

    reshape_vector = tf.reshape(max_pooling_layer3, [-1, 1024])
    fully_connected_layer1 = tf.layers.dense(inputs=reshape_vector, units=1024, activation=tf.nn.relu)
    y_out = tf.layers.dense(inputs=fully_connected_layer1, units=25, name='y_out')

    total_loss = tf.losses.hinge_loss(tf.one_hot(y, 25), logits=y_out)
    mean_loss = tf.reduce_mean(total_loss, name="mean_loss")

    adam_optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = adam_optimizer.minimize(mean_loss)

    # Define correct Prediction and accuracy
    correct_prediction = tf.equal(tf.argmax(y_out, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        print("Loop\t\tTrain Loss\t\tTrain Acc %\t\tTest loss\t\tTest Acc %")
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            # Shuffle input like last time.
            s = np.arange(xTrain.shape[0])
            np.random.shuffle(s)
            xTr = xTrain[s]
            yTr = yTrain[s]
            batch_xs = xTr[:batch_size]
            batch_ys = yTr[:batch_size]
            train_loss, train_acc, _ = sess.run([mean_loss, accuracy, train_step], feed_dict={x: batch_xs, y: batch_ys})
            test_loss, test_acc = sess.run([mean_loss, accuracy], feed_dict={x: xTest, y: yTest})
            print('{0}\t\t{1:0.6f}\t\t{2:0.6f}\t\t{3:0.6f}\t\t{4:0.6f}'.format(
                int(epoch), train_loss, train_acc * 100, test_loss, test_acc * 100))
            if epoch % 100 == 0:
                save_path = saver.save(sess, './IFGSM/model/model.ckpt')

        # save session
        save_path = saver.save(sess, './IFGSM/model/model.ckpt')
        print("Model saved in file: ", save_path)
    sess.close()


# Algorithm from paper Harnessing adversarial examples
def iterative_fast_gradient_sign_method(img_data, actual_label, max_iterations, epsilon=0.25, visualize=False):

    # Format input data.
    input_data = img_data[np.newaxis, :]

    # Create input session and load the model.
    sess = tf.Session()
    saver = tf.train.import_meta_graph('./IFGSM/model/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./IFGSM/model/'))
    graph = tf.get_default_graph()

    # Load the tensor to supply the input image.
    input_image = graph.get_tensor_by_name('x:0')  # Input tensor for model
    y_out = graph.get_tensor_by_name('y_out/BiasAdd:0')  # output of convolution layer

    # Generate perturbation.
    perturbation = (epsilon * tf.sign(tf.gradients(y_out, input_image)))
    # Add it to original image.
    adversarial_img = input_image + perturbation
    # Clip the adversarial image values and keep them between 0 and 255.
    clipped_img = tf.clip_by_value(adversarial_img, 0, 255)
    # Cast values to integers.
    output_image_tensor = tf.cast(clipped_img, tf.int64)

    # use a placeholder to feed the input data.
    input_img_placeholder = input_data

    # iteratively add perturbation to the image for max iterations
    for i in range(0, max_iterations):
        input_img_placeholder = np.reshape(sess.run([output_image_tensor], feed_dict={input_image: input_img_placeholder}),
                                           [1, 32, 32, 1])

    # Visualize the image
    if visualize:
        pair_visual(np.reshape(img_data, [32, 32]), np.reshape(input_img_placeholder[0], [32, 32]))
        input()

    # return the perturbed image.
    return np.reshape(input_img_placeholder, [32, 32, 1])


def craft_adversarial_examples(x_test, y_test, num, max_iterations):
    print('Crafting adversarial examples using: I-FGSM.')
    print('Generating:, ', num, ' adversarial examples using I-FGSM. Num iterations: ', max_iterations)
    import random
    adversarial_examples, ground_truth = list(), list()
    for i in range(0, num):
        index = random.randint(0, len(x_test) - 1)
        adv_image = iterative_fast_gradient_sign_method(x_test[index], y_test[index], max_iterations)
        adversarial_examples.append(adv_image)
        ground_truth.append(yTest[index])
        if len(adversarial_examples) % 10 == 0:
            print(len(adversarial_examples))

    # attack the model
    sess = tf.Session()
    saver = tf.train.import_meta_graph('./IFGSM/model/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./IFGSM/model/'))
    graph = tf.get_default_graph()

    input_image = graph.get_tensor_by_name('x:0')  # Input tensor for model
    y_out = graph.get_tensor_by_name('y_out/BiasAdd:0')  # output of convolution layer
    y_exp = graph.get_tensor_by_name('expected_output:0')
    correct_prediction = tf.equal(tf.argmax(y_out, 1), y_exp)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    acc = sess.run([accuracy], feed_dict={input_image: adversarial_examples, y_exp: ground_truth})
    print('Accuracy with new adversarial set:', acc[0] * 100)
    input()


#
def pair_visual(original, adversarial, figure=None):
    """
    This function displays two images: the original and the adversarial sample
    :param original: the original input
    :param adversarial: the input after perterbations have been applied
    :param figure: if we've already displayed images, use the same plot
    :return: the matplot figure to reuse for future samples
    """
    import matplotlib.pyplot as plt

    # Ensure our inputs are of proper shape
    assert(len(original.shape) == 2 or len(original.shape) == 3)

    # To avoid creating figures per input sample, reuse the sample plot
    if figure is None:
        plt.ion()
        figure = plt.figure()
        figure.canvas.set_window_title('Pair Visualization')

    # Add the images to the plot
    perterbations = adversarial - original
    for index, image in enumerate((original, perterbations, adversarial)):
        figure.add_subplot(1, 3, index + 1)
        plt.axis('off')
        # If the image is 2D, then we have 1 color channel
        if len(image.shape) == 2:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)

        # Give the plot some time to update
        plt.pause(0.01)

    # Draw the plot and return
    plt.show()
    return figure



ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="Dataset Directiory path")
ap.add_argument("-train","--train", required=False,
                help="for training")
ap.add_argument("-test","--test", required=False,
                help="for testing")

args = vars(ap.parse_args())
DATASET_PATH = args["dataset"]

if args["train"] == "true":
    train_model_on_malImg()
elif args["test"] == "true":
    xTrain, yTrain, xTest, yTest = load_data()
    craft_adversarial_examples(xTest, yTest, 10, 3)

input()
