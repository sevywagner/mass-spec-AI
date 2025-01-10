import numpy as np
from Helpers.metricsHelpers import binaryCrossentropy

class DecisionTreeRegressor:
    def __init__(self, min_samples = 3, max_depth=5):
        self.min_samples = min_samples
        self.max_depth = max_depth
    
    def fit(self, X, y):
        '''
            Function:
                Construct decision tree with recursive algorithm
            
            Parameters:
                X (np.array(np.array(np.float64))): dependant variables
                y (np.array(np.int8): independant variable
        '''
        self.tree = self.constructTree(X, y, 0)

    def constructTree(self, X, y, depth):
        '''
            Function:
                Recursive algorithm that constructs desision tree

            Parameters:
                X (np.array(np.array(np.float64))): dependant variables
                y (np.array(np.int8): independant variable
                depth (int): depth of tree

            Returns:
                tree (dict): decision tree
                loss (float): binary crossentropy loss over training
        '''
        num_samples, num_features = X.shape

        if (depth > self.max_depth or num_samples < self.min_samples):
            return { 'value': np.mean(y) }

        min_loss, best_feature, best_threshold = float('inf'), None, None
        for i in range(num_features):
            thresholds = np.unique(X[:, i])
            for threshold in thresholds:
                left_indeces = X[:, i] < threshold
                right_indeces = X[:, i] >= threshold

                if len(y[left_indeces]) == 0 or len(y[right_indeces]) == 0:
                    continue

                loss_left = binaryCrossentropy(y[left_indeces], np.mean(y[left_indeces]))
                loss_right = binaryCrossentropy(y[right_indeces], np.mean(y[right_indeces]))
                weighted_loss = ((len(y[left_indeces]) * loss_left) + (len(y[right_indeces]) * loss_right)) / num_samples

                if min_loss > weighted_loss:
                    min_loss = weighted_loss
                    best_feature = i
                    best_threshold = threshold
            
        if (best_feature == None):
            return { 'value': np.mean(y) }

        left_indeces = X[:, best_feature] < best_threshold
        right_indeces = ~left_indeces

        left_subtree = self.constructTree(X[left_indeces], y[left_indeces], depth + 1)
        right_subtree = self.constructTree(X[right_indeces], y[right_indeces], depth + 1)

        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }

    def predict(self, X):
        '''
            Function:
                Runs recursive algorithm to make predictions

            Parameters:
                X (np.array(np.array(np.float64))): dependant variables

            Returns:
                preds (np.array(np.float64)): independant variable predictions
        '''
        return np.array([self.predict_sample(sample, self.tree) for sample in X], dtype=np.float64)

    def predict_sample(self, sample, tree):
        '''
            Function:
                Recursive algorithm to make predictions based on the tree

            Parameters:
                sample (np.array(np.float64)): sample from the X data
                tree (dict): subtree that current recursion iteration is working with

            Return:
                pred (np.float64): prediction
        '''
        if ('value' in tree):
            return tree['value']
        
        if (sample[tree['feature']] < tree['threshold']):
            return self.predict_sample(sample, tree['left'])
        else:
            return self.predict_sample(sample, tree['right'])