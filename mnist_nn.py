import argparse
import logging
import numpy as np
import pandas as pd
import os

# Define paths for datasets copied locally
DATASET_DIR = os.path.join(os.getcwd(), "dataset")
TRAIN_CSV = os.path.join(DATASET_DIR, "mnist_train.csv")
TEST_CSV = os.path.join(DATASET_DIR, "mnist_test.csv")

# Define activation/loss functions and their derivatives
sigmoid = lambda x: 1. / (1 + np.exp(-x))
grad_sigmoid = lambda x: x * (1 - x)
tanh = lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
grad_tanh = lambda x: 1 - x ** 2
softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
crossent = lambda x, y: -(1./x.shape[0]) * (np.sum(np.multiply(x, np.log(y))))

# Gradient descent update function Wnew = Wold - learning_rate * gradient(Wold)
grad_update = lambda w, grad_w, lr: w - (lr * grad_w)


class NeruralNetwork(object):
    """ 
    Neural Network Implmentation 
    """
    def __init__(self, input_nodes, hidden_neurons, output_nodes):
        self.activation_funcs = ["sigmoid", "tanh"]
        
        # Xavier initialization of weights
        # weights and bias between input nodes and hidden layer
        self.w1 = np.random.randn(input_nodes, hidden_neurons) / np.sqrt(input_nodes)
        self.b1 = np.zeros((1, hidden_neurons)) / np.sqrt(input_nodes)

        # weight and bias between hidden layer and output nodes
        self.w2 = np.random.randn(hidden_neurons, output_nodes) / np.sqrt(hidden_neurons)
        self.b2 = np.zeros((1, output_nodes)) / np.sqrt(hidden_neurons)


    def feedforward(self, X, w1, b1, w2, b2, activation_fn):
        """
        Feed forward pass logic
        """
        if activation_fn not in self.activation_funcs:
            raise AssertionError("Activation function {} is not defined".format(
                activation_fn
            ))

        # Update rule non-linearity(w.X + b)
        h1 = globals()[activation_fn](np.dot(X, w1) + b1)
        op = globals()["softmax"](np.dot(h1, w2) + b2)
        return h1, op

    def backpropogate(self, X, h1, w2, ypredicted, yactual, gradfn, w1=None):
        """
        Back propogation logic
        """
        # calculate difference between actual and predicted value
        delta_ouput = (ypredicted - yactual) / yactual.shape[0]

        # h^T . L
        grad_w2 = np.dot(h1.T, delta_ouput)
        grad_b2 = np.sum(delta_ouput, axis=0, keepdims=True)

        # w2^T . L
        grad_h1 = np.dot(delta_ouput, w2.T)
        delta_h1 = grad_h1 * globals()[gradfn](h1)

        # X^T . delta(h1)
        grad_w1 = np.dot(X.T, delta_h1)
        grad_b1 = np.sum(delta_h1, axis=0, keepdims=True)

        return grad_w1, grad_b1, grad_w2, grad_b2

    def update_weights(self, w1, gw1, b1, gb1, w2, gw2, b2, gb2, lr):
        """
        Update weights using Wnew = Wold - learning_rate * gradient(Wold)
        """
        w1new = globals()["grad_update"](w1, gw1, lr)
        b1new = globals()["grad_update"](b1, gb1, lr)
        w2new = globals()["grad_update"](w2, gw2, lr)
        b2new = globals()["grad_update"](b2, gb2, lr)
        return w1new, b1new, w2new, b2new


    def train(self, X, Y, iters, learning_rate, activation_fn, print_freq=1000):
        """
        Training logic
        """
        logging.info("***** Training Phase *****")
        
        w1, b1, w2, b2 = self.w1, self.b1, self.w2, self.b2
        
        for iter in range(1, iters+1):            
            # feed forward pass
            h1, op = self.feedforward(X, w1, b1, w2, b2, activation_fn)
            
            # backpropogation
            grad_w1, grad_b1, grad_w2, grad_b2 = self.backpropogate(
                X, h1, w2, op, Y, "grad_{}".format(activation_fn)
            )

            # update weights
            w1, b1, w2, b2 = self.update_weights(
                w1, grad_w1, b1, grad_b1, w2, grad_w2, b2, grad_b2, learning_rate
            )

            # feed forward pass with updated weight
            h1, op = self.feedforward(X, w1, b1, w2, b2, activation_fn)

            # total loss using crossentropy loss function
            loss = globals()["crossent"](Y, op)

            # accuracy calculation i.e. Correct Prediction / Total Prediction
            digits_actual = np.argmax(Y, axis=1)
            digits_predicted = np.argmax(op, axis=1)
            accuracy = (np.count_nonzero(digits_actual == digits_predicted) / digits_actual.shape[0]) * 100

            logging.debug("Iteraion: {} --> Loss: {}, Accuracy: {}".format(
                iter, loss, accuracy))
            if iter % print_freq == 0:
                logging.info("Iteration: {} --> Loss: {}, Accuracy: {}".format(
                    iter, loss, accuracy
                ))

        logging.info("Training Finished!")
        logging.info("-" * 80)
        return w1, b1, w2, b2

    def test(self, X, Y, w1, b1, w2, b2, activation_fn):
        """
        Testing logic
        """
        logging.info("***** Testing Phase *****")
        _, op = self.feedforward(X, w1, b1, w2, b2, activation_fn)
        digits_actual = np.argmax(Y, axis=1)
        digits_predicted = np.argmax(op, axis=1)
        accuracy = (np.count_nonzero(digits_actual == digits_predicted) / digits_actual.shape[0]) * 100
        # logging.info(op)
        # logging.info("Digit Actual: {}, Digit Predicted: {}".format(digits_actual, digits_predicted))
        logging.info("Accuracy: {}".format(accuracy))
        logging.info("Testing Finished!")
        logging.info("-" * 80)


def load_dataset(train_csv, test_csv):
    """
    Prepare dataset for training
    """
    train_data = np.loadtxt(train_csv, delimiter=",")
    test_data = np.loadtxt(test_csv, delimiter=",")

    # split dataframes into images and labels
    # 1st column is digit and rest 784 column represets 28x28 image
    train_imgs = np.asfarray(train_data[:, 1:])
    train_labels = np.asfarray(train_data[:, :1])
    test_imgs = np.asfarray(test_data[:, 1:])
    test_labels = np.asfarray(test_data[:, :1])
    
    return train_imgs, train_labels, test_imgs, test_labels


def train_val_split(X, Y, split_ratio):
    """
    Splits training data in to training and validation data
    """
    len_x = X.shape[0]
    shuffled_idx = np.random.permutation(len_x)

    xtrain = X[shuffled_idx[:int(split_ratio * len_x)], :]
    ytrain = Y[shuffled_idx[:int(split_ratio * len_x)], :]

    xval = X[shuffled_idx[int(split_ratio * len_x):], :]
    yval = Y[shuffled_idx[int(split_ratio * len_x):], :]

    return xtrain, ytrain, xval, yval

def verification():
    """
    Verification of Implementation on XOR data
    """
    xtrain = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]]).astype(np.float32)
    ytrain = np.asarray([0, 1, 1, 0]).astype(np.int32)
    ytrain = np.eye(2)[ytrain]
    
    xval = np.asarray([[1, 0], [0, 1], [1, 1], [0, 0]]).astype(np.float32)
    yval = np.asarray([1, 1, 0, 0]).astype(np.int32)
    yval = np.eye(2)[yval]

    nn_obj = NeruralNetwork(2, 10, 2)
    w1, b1, w2, b2 = nn_obj.train(xtrain, ytrain,
        args.iters, args.learning_rate, args.activation_fn, print_freq=5000)
    nn_obj.test(xval, yval, w1, b1, w2, b2, args.activation_fn)


def train_full_and_validate():
    """
    Training on full set of 60K images
    and 
    Validation on test images
    """
    train_imgs, train_labels, test_imgs, test_labels = load_dataset(
        args.train_csv, args.test_csv
    )

    # One-hot encoding of labels
    train_labels = np.array(pd.get_dummies(train_labels.squeeze()))
    test_labels = np.array(pd.get_dummies(test_labels.squeeze()))

    nn_obj = NeruralNetwork(784, args.hidden_neurons, 10)
    w1, b1, w2, b2 = nn_obj.train(train_imgs, train_labels,
        args.iters, args.learning_rate, args.activation_fn, print_freq=25)
    nn_obj.test(test_imgs, test_labels, w1, b1, w2, b2, args.activation_fn)
    

def main():
    train_imgs, train_labels, test_imgs, test_labels = load_dataset(
        args.train_csv, args.test_csv
    )
    
    xtrain, ytrain, xval, yval = train_val_split(
        train_imgs, train_labels, args.split_ratio
    )
    
    # One-hot encoding of labels
    ytrain = np.array(pd.get_dummies(ytrain.squeeze()))
    yval = np.array(pd.get_dummies(yval.squeeze()))
    
    nn_obj = NeruralNetwork(784, args.hidden_neurons, 10)
    w1, b1, w2, b2 = nn_obj.train(xtrain, ytrain,
        args.iters, args.learning_rate, args.activation_fn, print_freq=25)
    nn_obj.test(xval, yval, w1, b1, w2, b2, args.activation_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default=TRAIN_CSV,
                        help="csv file path of training data")
    parser.add_argument("--test_csv", type=str, default=TEST_CSV,
                        help="csv file path of test data")
    parser.add_argument("--split_ratio", type=float, default=0.75,
                        help="Ratio at which dataset need to be split between"
                             " training vs validation data")
    parser.add_argument("--activation_fn", type=str, default="sigmoid",
                        help="activation to be used for hidden layer")
    parser.add_argument("--iters", type=int, default=500,
                        help="number of iterations to train for")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="learning rate for training purpose")
    parser.add_argument("--hidden_neurons", type=int, default=128,
                        help="Number of hidden neurons to use")
    parser.add_argument("--verify_xor", action="store_true",
                        help="Provide this flag to verify implementation with XOR dataset")
    parser.add_argument("--train_full", action="store_true",
                        help="Provide this flag when need to train on full dataset")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.verify_xor:
        verification()
    elif args.train_full:
        train_full_and_validate()
    else:
        main()
