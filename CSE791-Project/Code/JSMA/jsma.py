"""
JSMA attack based on MALIMG Dataset
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags
import logging
import argparse

from cleverhans.attacks import SaliencyMapMethod
from cleverhans.utils import other_classes, set_log_level
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, model_argmax
from cleverhans.utils_keras import KerasModelWrapper, cnn_model
from JSMA.jsma_model import make_basic_cnn

FLAGS = flags.FLAGS
# DATASET_PATH = "../malimg.npz"

def mnist_tutorial_jsma(train_start=0, train_end=7016, test_start=0,
                        test_end=2323, viz_enabled=True, nb_epochs=6,
                        batch_size=128, nb_classes=25, source_samples=10,
                        learning_rate=0.001):
    """
    MNIST tutorial for the Jacobian-based saliency map approach (JSMA)
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param viz_enabled: (boolean) activate plots of adversarial examples
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param nb_classes: number of output classes
    :param source_samples: number of test inputs to attack
    :param learning_rate: learning rate for training
    :return: an AccuracyReport object
    """
    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # MNIST-specific dimensions
    img_rows = 32
    img_cols = 32
    channels = 1

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    print("Created TensorFlow session.")

    set_log_level(logging.DEBUG)

    # Get MNIST test data
    # X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
    #                                               train_end=train_end,
    #                                               test_start=test_start,
    #                                               test_end=test_end)

    X_train, Y_train, X_test, Y_test = load_malimg()

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))
    y = tf.placeholder(tf.float32, shape=(None, 25))

    # Define TF model graph
    model = make_basic_cnn()
    preds = model(x)
    print("Defined TensorFlow model graph.")

    ###########################################################################
    # Training the model using TensorFlow
    ###########################################################################

    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    sess.run(tf.global_variables_initializer())
    rng = np.random.RandomState([2017, 8, 30])
    model_train(sess, x, y, preds, X_train, Y_train, args=train_params,
                rng=rng)

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params)
    assert X_test.shape[0] == test_end - test_start, X_test.shape
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
    report.clean_train_clean_eval = accuracy

    ###########################################################################
    # Craft adversarial examples using the Jacobian-based saliency map approach
    ###########################################################################
    print('Crafting ' + str(source_samples) + ' * ' + str(nb_classes-1) +
          ' adversarial examples')

    # Keep track of success (adversarial example classified in target)
    results = np.zeros((nb_classes, source_samples), dtype='i')

    # Rate of perturbed features for each test set example and target class
    perturbations = np.zeros((nb_classes, source_samples), dtype='f')

    # Initialize our array for grid visualization
    grid_shape = (nb_classes, nb_classes, img_rows, img_cols, channels)
    grid_viz_data = np.zeros(grid_shape, dtype='f')

    # Instantiate a SaliencyMapMethod attack object
    jsma = SaliencyMapMethod(model, back='tf', sess=sess)
    jsma_params = {'theta': 0.1, 'gamma': 1.,
                   'clip_min': 0., 'clip_max': 1.,
                   'y_target': None}

    figure = None
    # Loop over the samples we want to perturb into adversarial examples
    for sample_ind in xrange(0, source_samples):
        print('--------------------------------------')
        print('Attacking input %i/%i' % (sample_ind + 1, source_samples))
        sample = X_test[sample_ind:(sample_ind+1)]

        # We want to find an adversarial example for each possible target class
        # (i.e. all classes that differ from the label given in the dataset)
        current_class = int(np.argmax(Y_test[sample_ind]))
        target_classes = other_classes(nb_classes, current_class)

        # For the grid visualization, keep original images along the diagonal
        grid_viz_data[current_class, current_class, :, :, :] = np.reshape(
            sample, (img_rows, img_cols, channels))

        # Loop over all target classes
        for target in target_classes:
            print('Generating adv. example for target class %i' % target)

            # This call runs the Jacobian-based saliency map approach
            one_hot_target = np.zeros((1, nb_classes), dtype=np.float32)
            one_hot_target[0, target] = 1
            jsma_params['y_target'] = one_hot_target
            adv_x = jsma.generate_np(sample, **jsma_params)

            # Check if success was achieved
            res = int(model_argmax(sess, x, preds, adv_x) == target)
            # Computer number of modified features
            adv_x_reshape = adv_x.reshape(-1)
            test_in_reshape = X_test[sample_ind].reshape(-1)
            nb_changed = np.where(adv_x_reshape != test_in_reshape)[0].shape[0]
            percent_perturb = float(nb_changed) / adv_x.reshape(-1).shape[0]

            # Display the original and adversarial images side-by-side
            if viz_enabled:
                figure = pair_visual(
                    np.reshape(sample, (img_rows, img_cols)),
                    np.reshape(adv_x, (img_rows, img_cols)), figure , target)


            # Add our adversarial example to our grid data
            grid_viz_data[target, current_class, :, :, :] = np.reshape(
                adv_x, (img_rows, img_cols, channels))

            # Update the arrays for later analysis
            results[target, sample_ind] = res
            perturbations[target, sample_ind] = percent_perturb

    print('--------------------------------------')

    # Compute the number of adversarial examples that were successfully found
    nb_targets_tried = ((nb_classes - 1) * source_samples)
    succ_rate = float(np.sum(results)) / nb_targets_tried
    print('Avg. rate of successful adv. examples {0:.4f}'.format(succ_rate))
    report.clean_train_adv_eval = 1. - succ_rate

    # Compute the average distortion introduced by the algorithm
    percent_perturbed = np.mean(perturbations)
    print('Avg. rate of perturbed features {0:.4f}'.format(percent_perturbed))

    # Compute the average distortion introduced for successful samples only
    percent_perturb_succ = np.mean(perturbations * (results == 1))
    print('Avg. rate of perturbed features for successful '
          'adversarial examples {0:.4f}'.format(percent_perturb_succ))

    # Close TF session
    sess.close()

    # Finally, block & display a grid of all the adversarial examples
    # if viz_enabled:
    #     import matplotlib.pyplot as plt
    #     plt.close(figure)
    #     _ = grid_visual(grid_viz_data)

    return report

def one_hot(labels, n_class = 6):
	""" One-hot encoding """
	expansion = np.eye(n_class)
	y = expansion[:, labels-1].T
	assert y.shape[1] == n_class, "Wrong number of labels!"

	return y

def load_malimg():
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

    x_train = list()
    x_train = np.concatenate((xTrain1, xTrain2, xTrain3, xTrain4, xTrain5,
                              xTrain6, xTrain7, xTrain8, xTrain9, xTrain10,
                              xTrain11, xTrain12, xTrain13, xTrain14, xTrain15,
                              xTrain16, xTrain17, xTrain18, xTrain19, xTrain20,
                              xTrain21, xTrain22, xTrain23, xTrain24, xTrain25), axis=0)
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

    y_train = np.concatenate((yTrain1, yTrain2, yTrain3, yTrain4, yTrain5,
                              yTrain6, yTrain7, yTrain8, yTrain9, yTrain10,
                              yTrain11, yTrain12, yTrain13, yTrain14, yTrain15,
                              yTrain16, yTrain17, yTrain18, yTrain19, yTrain20,
                              yTrain21, yTrain22, yTrain23, yTrain24, yTrain25), axis=0)

    y_train = one_hot(y_train,25)

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

    y_test = np.concatenate((yTest1, yTest2, yTest3, yTest4, yTest5,
                             yTest6, yTest7, yTest8, yTest9, yTest10,
                             yTest11, yTest12, yTest13, yTest14, yTest15,
                             yTest16, yTest17, yTest18, yTest19, yTest20,
                             yTest21, yTest22, yTest23, yTest24, yTest25), axis=0)

    y_test = one_hot(y_test,25)

    print('shape of xtrain :',np.shape(x_train))
    print('shape of ytrain :', np.shape(y_train))
    print('shape of xtest :',np.shape(x_test))
    print('shape of ytest :', np.shape(y_test))
    return x_train, y_train, x_test, y_test

def main(argv=None):
    mnist_tutorial_jsma(viz_enabled=FLAGS.viz_enabled,
                        nb_epochs=FLAGS.nb_epochs,
                        batch_size=FLAGS.batch_size,
                        nb_classes=FLAGS.nb_classes,
                        source_samples=FLAGS.source_samples,
                        learning_rate=FLAGS.learning_rate)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="Dataset Directiory path")
    args = vars(ap.parse_args())
    DATASET_PATH = args["dataset"]
    flags.DEFINE_boolean('viz_enabled', True, 'Visualize adversarial ex.')
    flags.DEFINE_integer('nb_epochs', 20, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_integer('nb_classes', 25, 'Number of output classes')
    flags.DEFINE_integer('source_samples', 2, 'Nb of test inputs to attack')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')

    tf.app.run()
