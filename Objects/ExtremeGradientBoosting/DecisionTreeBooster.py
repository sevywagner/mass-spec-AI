import numpy as np
from Helpers.metricsHelpers import binaryCrossentropy
from concurrent.futures import ThreadPoolExecutor

class DecisionTreeBooster:
    def __init__(self, X, gradients, hessians, learning_rate=.1, n_learners=200, subsample=.25, max_depth=5, min_samples=3, gamma=0.0, reg_lambda=1.0):
        self.learning_rate = learning_rate
        self.n_learners = n_learners
        self.subsample = subsample
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.best_gain = 0
        self.best_feature = None
        self.best_threshold = None
        self.value = -np.sum(gradients) / (np.sum(hessians) + reg_lambda)
        if self.max_depth > 0:
            self.fit(X, gradients, hessians)

    def fit(self, X, gradients, hessians):
        '''
            Function:
                Construct a decision tree booster

            Parameters:
                X (np.array(np.array(np.float32))): data samples
                gradients (np.array(np.float32)): first order derivative of the loss function
                hessians (np.array(np.float32)): second order derivative of the loss function
        '''
        if (self.max_depth == 0 or len(X) < self.min_samples):
            return self.value

        self.find_best_split(X, gradients, hessians)
        if self.is_leaf:
            return
    
        left_mask = X[:, self.best_feature] <= self.best_threshold
        right_mask = ~left_mask

        self.left = DecisionTreeBooster(X[left_mask], gradients[left_mask], hessians[left_mask], max_depth=self.max_depth - 1)
        self.right = DecisionTreeBooster(X[right_mask], gradients[right_mask], hessians[right_mask], max_depth=self.max_depth - 1)

    
    @property
    def is_leaf(self):
        return self.best_gain == 0
    
    def get_gain(self, gradients, hessians, grad_left, hess_left, grad_right, hess_right):
        '''
            Function:
                Calculate the gain of a split

            Parameters:
                gradients (np.float32): sum of the entire gradient array
                hessians (np.float32): sum of the entire hessian array
                grad_left (np.float32): sum of gradients on the left of the split
                hess_left (np.float32): sum of hessians on the left of the split
                grad_right (np.float32): sum of gradients on the right of the split
                hess_right (np.float32): sum of hessians on the right of the split

            Returns:
                gain (np.float32): calculated gain
        '''
        gain = .5 * (
            (grad_left ** 2 / (hess_left + self.reg_lambda)) + (grad_right ** 2 / (hess_right + self.reg_lambda)) - 
            (gradients ** 2 / (hessians + self.reg_lambda))
        ) - self.gamma / 2
        return gain

    def find_best_split(self, X, gradients, hessians):
        '''
            Function:
                Find the best feature and threshold to split at in the tree using greedy exact algorithm
            
            Parameters:
                X (np.array(np.array(np.float32))): data samples
                gradients (np.array(np.float32)): first order derivative of the loss function
                hessians (np.array(np.float32)): second order derivative of the loss function
        '''
        grad_sum, hess_sum = np.sum(gradients), np.sum(hessians)
        
        for i in range(X.shape[1]):
            grad_left, hess_left = 0., 0.

            x = X[:, i]
            mask = np.argsort(x)
            g_sort = gradients[mask]
            h_sort = hessians[mask]
            x_sort = x[mask]

            g_sort_cumsum = np.cumsum(g_sort)
            h_sort_cumsum = np.cumsum(h_sort)
            
            for j in range(len(g_sort) - 1):
                if (x_sort[j] == x_sort[j + 1]):
                    continue

                grad_left = g_sort_cumsum[j]
                hess_left = h_sort_cumsum[j]
                grad_right = grad_sum - grad_left
                hess_right = hess_sum - hess_left

                gain = self.get_gain(grad_sum, hess_sum, grad_left, hess_left, grad_right, hess_right)
                if gain > self.best_gain:
                    self.best_gain = gain
                    self.best_feature = i
                    self.best_threshold = (x_sort[j] + x_sort[j + 1]) / 2

    def predict_sample(self, sample):
        '''
            Function:
                Recursively predict a sample based on the best split feature and threshold
            
            Parameters:
                sample np.array(np.float32): 1 data sample
            
            Returns:
                prediction (np.float32): the predicted label
        '''
        if self.is_leaf:
            return self.value
        
        subtree = self.left if sample[self.best_feature] <= self.best_threshold else self.right
        return subtree.predict_sample(sample)

    def predict(self, X):
        '''
            Function:
                Predict an array of data samples
            
            Parameters:
                X (np.array(np.array(np.float32))): data samples

            Returns: 
                y_pred: np.array(np.float32)
        '''
        return np.array([self.predict_sample(sample) for sample in X], dtype=np.float32)