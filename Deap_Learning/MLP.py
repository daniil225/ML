import sys

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np



class NeuralNetMLP(object):
    """ Neural net forward propagation / classifer base on multilayer perceptron
    Parametrs
    ---------
    n_hidden : int (default: 30)
        Amount hidden elements
    l2 : float (default: 0.0)
        value lambda for requralization L2
        Reguralization is apsent if l2=0.0 (accepted by default)
    epochs: int (default: 100)
        Number of passes on dataset
    eta: float(default: 0.001)
        Speed of stadiing
    shuffle: bool(default:True)
        if True then trainig data is shuffled
        every epoch to prevent loops
    minibatch_size: int(default:1)
        Number trainig data on minibatch
    seed: int (default: None)
        Random starting value for initialization of weights and shuffled

    Attributes
    ----------
    eval_ : dict
        Dictionari in which cost indicators are collected at training and correctness
        at testing for every epoch duaring the training
    """
    def __init__(self, n_hidden=30, l2=0.0, epochs=100, eta=0.001, shuffle=True, minibatch_size=1, seed=None):
        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size

    def _onehot(self, y, n_classes):
        """ encodes labels in onehot encodeed representation

        :param
        ------
        y: Array, form = [n_examples]
            target values
        :return
        -------
         onehot: Array, form = (n_examples, n_labels)
        """
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.
        return onehot.T

    def _sigmoid(self,z):
        """ Calculate logistical (sigmoid) function """
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def _forward(self, X):
        """ Calculate step forward propagation """
        # step 1: general enter hidden layer
        # scalar multiplication [n_examples, n_features]
        #                    and[n_features, n_hidden]
        # -> [n_examples, n_hidden]
        z_h = np.dot(X, self.w_h) + self.b_h

        # step 2: activation hidden layer
        a_h = self._sigmoid(z_h)

        # step 3: general enter output layer
        # scalar mul [n_examples, n_hidden]
        #          and [n_hidden, n_classlabels]
        # -> [n_examples, n_classlabels]
        z_out = np.dot(a_h, self.w_out) + self.b_out

        # step 4: activtion output layer
        a_out = self._sigmoid(z_out)
        return z_h, a_h, z_out, a_out

    def _compute_cost(self, y_enc, output):
        """ Calculate cost function

        :param y_enc: array, form = (n_examples, n_labels)
            Target of classes in onehot code
        :param output: array, form = [n_examples, n_output_units]
            Activation output layer (forward prop)
        :return:
        cost : float
            Reguralization cost
        """
        L2_term = (self.l2 *
                   (np.sum(self.w_h ** 2.) +
                    np.sum(self.w_out ** 2.)))
        term1 = -y_enc * (np.log(output))
        term2 = (1.-y_enc) * (np.log(1. - output))
        cost = np.sum(term1 - term2) + L2_term
        return cost

    def predict(self, X):
        """ predict target of classes
         :param
         X: array, form = [n_examples, n_features]
            Input layer with initial signs
        :return
        y_pred: array, form= [n_examples]
            Predictable target of classes
         """
        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred

    def fit(self, X_train, y_train, X_valid, y_valid):
        """ figure out weights from data
         :param
         X_train : array, form = [n_examples, n_features]
            Input layer with initial signs
        y_train: array, form = [n_examples]
            target labels of classes
        X_valid: array, shape = [n_examples, n_features]
            specimen signs for testing duaring the training
        y_valid: array, shape = [n_examples]
            Labels signs for testing duaring the trainig
        :return
        self
         """
        n_output = np.unique(y_train).shape[0] # number class of labels
        n_features = X_train.shape[1]

        ####################
        #initialize weights#
        ####################

        # Weights for input layer -> hiden layer
        self.b_h = np.zeros(self.n_hidden)
        self.w_h = self.random.normal(loc=0.0, scale=0.1, size=(n_features, self.n_hidden))

        # Weights for hidden layer -> output layer
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1, size=(self.n_hidden, n_output))

        epoch_strlen = len(str(self.epochs)) # for format
        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': []}

        y_train_enc = self._onehot(y_train, n_output)

        # iteration on trainig epochs
        for i in range(self.epochs):
            # iteration on minibatch
            indices = np.arange(X_train.shape[0])

            if self.shuffle:
                self.random.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - self.minibatch_size + 1, self.minibatch_size):
                batch_idx = indices[start_idx: start_idx + self.minibatch_size]

                #forward popagation
                z_h, a_h, z_out, a_out = self._forward(X_train[batch_idx])

                ##################
                #Back propagation#
                ##################
                # [n_examples, n_classlabel]
                delta_out = a_out - y_train_enc[batch_idx]
                #[n_examples, n_hidden]
                sigmoid_derivative_h = a_h * (1. - a_h)
                # scaler mul [n_examples, n_classlabels]
                # and [n_classlabels, n_hidden]
                # -> [n_examples, n_hidden]
                delta_h = (np.dot(delta_out, self.w_out.T) * sigmoid_derivative_h)

                # scaller mul [n_features, n_examples]
                # and [n_features, n_hidden]
                # -> [n_feature, n_hidden]
                grad_w_h = np.dot(X_train[batch_idx].T, delta_h)
                grad_b_h = np.sum(delta_h, axis=0)

                # scaller mul [n_hidden, n_examples]
                # and [n_examples, n_classlabels]
                # -> [n_hidden, n_classlabel]
                grad_w_out = np.dot(a_h.T, delta_out)
                grad_b_out = np.sum(delta_out, axis=0)

                #Reguralization and updating weights
                delta_w_h = (grad_w_h + self.l2 * self.w_h)
                delta_b_h = grad_b_h # bais nonreg
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h
                delta_w_out = (grad_w_out + self.l2 * self.w_out)
                delta_b_out = grad_b_out
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out

            ############
            # Estimate #
            ############
            z_h, a_h, z_out, a_out = self._forward(X_train)
            cost = self._compute_cost(y_enc=y_train_enc, output=a_out)
            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)
            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) / X_train.shape[0])
            valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float) / X_valid.shape[0])
            sys.stderr.write('\r%0*d/%d | Издержки: %.2f | Правильность при обучении/ при проверке: %.2f%% / %.2f%%' %(epoch_strlen, i+1, self.epochs, cost, train_acc*100, valid_acc*100))
            sys.stderr.flush()
            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)
        return self





nn = NeuralNetMLP(n_hidden=100, l2=0.01, epochs=200, eta=0.0005, minibatch_size=100, shuffle=True, seed=1)

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
y = y.astype(int)
X = ((X / 255.) - .5) * 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=123, stratify=y)

nn.fit(X_train=X_train[:55000],
       y_train=y_train[:55000],
       X_valid=X_train[55000:],
       y_valid=y_train[55000:])


import matplotlib.pyplot as plt
plt.plot(range(nn.epochs), nn.eval_['cost'])
plt.ylabel('Издержки')
plt.xlabel('Эпохи')
plt.show()