import numpy as np
from sklearn import svm
import itertools


class MulticlassSVM:

    def __init__(self, mode):
        if mode != 'ovr' and mode != 'ovo' and mode != 'crammer-singer':
            raise ValueError('mode must be ovr or ovo or crammer-singer')
        self.mode = mode
        self.labels = None
        self.binary_svm = None
        self.W = None

    def fit(self, X, y):
        if self.mode == 'ovr':
            self.fit_ovr(X, y)
        elif self.mode == 'ovo':
            self.fit_ovo(X, y)
        elif self.mode == 'crammer-singer':
            self.fit_cs(X, y)

    def fit_ovr(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovr_student(X, y)

    def fit_ovo(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovo_student(X, y)

    def fit_cs(self, X, y):
        self.labels = np.unique(y)
        X_intercept = np.hstack([X, np.ones((len(X), 1))])

        N, d = X_intercept.shape
        K = len(self.labels)

        W = np.zeros((K, d))

        n_iter = 1500
        learning_rate = 1e-8
        for i in range(n_iter):
            W -= learning_rate * self.grad_student(W, X_intercept, y)

        self.W = W

    def predict(self, X):
        if self.mode == 'ovr':
            return self.predict_ovr(X)
        elif self.mode == 'ovo':
            return self.predict_ovo(X)
        else:
            return self.predict_cs(X)

    def predict_ovr(self, X):
        scores = self.scores_ovr_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_ovo(self, X):
        scores = self.scores_ovo_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_cs(self, X):
        X_intercept = np.hstack([X, np.ones((len(X), 1))])
        return self.labels[np.argmax(self.W.dot(X_intercept.T), axis=0)]

    def bsvm_ovr_student(self, X, y):
        """
        Train OVR binary classfiers.

        Args:
            X: training features.
            y: training labels.

        Returns:
            binary_svm: a dictionary with labels as keys,
                        and binary SVM models as values.
        """
        binary_svm = {}
        for i in self.labels:
            y_i = (y == i).astype(int)
            # using random_state=12345 for reproductivity
            b_svm = svm.LinearSVC(random_state=12345)
            b_svm.fit(X, y_i)
            binary_svm[i] = b_svm
        return binary_svm

    def bsvm_ovo_student(self, X, y):
        """
        Train OVO binary classfiers.

        Arguments:
            X: training features.
            y: training labels.

        Returns:
            binary_svm: a dictionary with label pairs as keys,
                        and binary SVM models as values.
        """
        binary_svm = {}

        # split the training samples for every label
        entry_dict = {}
        for i in self.labels:
            ind = np.where(y == i)[0]
            entry_dict[i] = (X[ind], y[ind])

        for i, j in itertools.combinations(self.labels, 2):
            # eliminates non-i and non-j samples
            X_new = np.concatenate([entry_dict[i][0], entry_dict[j][0]])
            # note y_new is using the original labels
            y_new = np.concatenate([entry_dict[i][1], entry_dict[j][1]])
            # using random_state=12345 for reproductivity
            b_svm = svm.LinearSVC(random_state=12345)
            b_svm.fit(X_new, y_new)
            binary_svm[(i, j)] = b_svm

        return binary_svm

    def scores_ovr_student(self, X):
        """
        Compute class scores for OVR.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        """
        scores = np.array([]).reshape((X.shape[0], 0))
        for i in self.binary_svm:
            b_svm = self.binary_svm[i]
            conf_score = b_svm.decision_function(X)
            scores = np.hstack([scores, conf_score[:, None]])
        return scores

    def scores_ovo_student(self, X):
        """
        Compute class scores for OVO.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        """
        scores = np.zeros((X.shape[0], self.labels.shape[0]))
        for i, j in self.binary_svm:
            b_svm = self.binary_svm[(i, j)]
            # note y_pred is using the original labels
            y_pred = b_svm.predict(X)
            i_ind = np.where(y_pred == i)[0]
            j_ind = np.where(y_pred == j)[0]
            scores[i_ind, i] += 1
            scores[j_ind, j] += 1
        return scores

    def loss_student(self, W, X, y, C=1.0):
        """
        Compute loss function given W, X, y.

        For exact definitions, please check the MP document.

        Arguments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The value of loss function given W, X and y.
        """
        if self.labels is None:
            self.labels = np.unique(y)

        # loss of regularization term
        l2_loss = 0.5 * np.sum(W**2)

        # gradient of the other term
        # get the matrix of term 1 - delta(j, y_i) + w_j^T * x_i
        loss_aug_inf = 1 - (self.labels[:, None] == y[None, :]) + np.matmul(W, np.transpose(X))  # (K, N)
        # sum over N of max value in loss_aug_inf
        loss_aug_inf_max_sum = np.sum(np.max(loss_aug_inf, axis=0))
        # sum over N of w_{y_i}^T * x_i
        wx_sum = np.sum(W[y] * X)
        multiclass_loss = C * (loss_aug_inf_max_sum - wx_sum)

        total_loss = l2_loss + multiclass_loss
        return total_loss

    def grad_student(self, W, X, y, C=1.0):
        """
        Compute gradient function w.r.t. W given W, X, y.

        For exact definitions, please check the MP document.

        Arguments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The gradient of loss function w.r.t. W,
            in a numpy array of shape (K, d).
        """
        if self.labels is None:
            self.labels = np.unique(y)

        # gradient of regularization term
        l2_grad = W

        # gradient of the other term
        # get the matrix of term 1 - delta(j, y_i) + w_j^T * x_i
        loss_aug_inf = 1 - (self.labels[:, None] == y[None, :]) + np.matmul(W, np.transpose(X))  # (K, N)
        # get the j_max that maximizes the above matrix for every sample
        j_max = np.argmax(loss_aug_inf, axis=0)  # (N,)
        # gradient of sum(...) is:   x_i, if k == j_max_i and k != y_i  (pos_case)
        #                           -x_i, if k != j_max_i and k == y_i  (neg_case)
        #                              0, otherwise
        pos_case = np.logical_and((self.labels[:, None] == j_max[None, :]), (self.labels[:, None] != y[None, :]))
        neg_case = np.logical_and((self.labels[:, None] != j_max[None, :]), (self.labels[:, None] == y[None, :]))
        multiclass_grad = C * np.matmul(pos_case.astype(int) - neg_case.astype(int) , X)

        total_grad = l2_grad + multiclass_grad
        return total_grad
