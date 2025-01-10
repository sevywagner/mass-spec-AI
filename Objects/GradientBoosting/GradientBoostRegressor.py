from Objects.GradientBoosting.DecisionTreeRegressor import DecisionTreeRegressor
from Helpers.metricsHelpers import getAccuracy, binaryCrossentropy
from Objects.Model import Model
from Helpers.metricsHelpers import sigmoid
from sklearn.base import BaseEstimator
import numpy as np

# Inherit from BaseEstimator for use of GridSearchCV during hyperparameter tuning
class GradientBoostRegressor(BaseEstimator, Model):
    def __init__(self, learning_rate=.1, base_learner=DecisionTreeRegressor, n_learners=200, subsample=.25, max_depth=5, min_samples=3):
        self.learning_rate = learning_rate
        self.base_learner = base_learner
        self.n_learners = n_learners
        self.subsample = subsample 
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.trees = []
    
    def calcResiduals(self, y, y_pred):
        '''
            Function:
                Calculate pseudo residuals based on the derivative of the binary cross entropy formula
            
            Parameters:
                y (np.array(int)): true y values
                y_pred (np.array(float)): predicted y values

            Returns:
                residuals (np.array(float)): pseudo residuals
        '''
        return y - sigmoid(y_pred)

    def fit(self, X, y, validation_set=None):
        '''
            Function:
                Train the decision trees using the provided data
            
            Parameters:
                X (np.array(np.array(float64))): X data
                y (np.array(int8)): y data
            
            Returns:
                preds (np.array(float64)): predictions on the training data
        '''
        fm = np.full(shape=y.shape, fill_value=0, dtype=np.float64)
        
        for i in range(self.n_learners):
            rands = np.random.randint(0, len(X), size=int(len(X) * .25))

            # y_subset = np.memmap('subset_y.dat', dtype=np.int8, mode='w+', shape=(len(rands),))
            # subset = np.memmap('subset_x.dat', dtype=np.float64, mode='w+', shape=(len(rands), len(X[0])))

            subset = np.array([X[i] for i in rands], dtype=np.float64)
            y_subset = np.array([y[i] for i in rands], dtype=np.int8)

            residual_learner = self.base_learner(max_depth=self.max_depth, min_samples=self.min_samples)
            residuals = self.calcResiduals(y_subset, fm[rands])
            residual_learner.fit(subset, residuals)
            self.trees.append(residual_learner)

            fm = fm + self.learning_rate * residual_learner.predict(X)

            self.loss = binaryCrossentropy(y, fm)
            if (i % 25 == 0):
                print(f"Learner {i} training loss: {self.loss}")
                if validation_set != None:
                    validation_preds = self.predict(validation_set[0], train=True)
                    print("Validation set accuracy: ", getAccuracy(validation_set[1], validation_preds))

            # y_subset.flush()
            # subset.flush()

        return fm, self.loss
    
    def predict(self, X, train=False):
        '''
            Function:
                Use gradient boosting to predict values

            Parameters:
                X (np.array(np.array(np.float64))): dependant variables
            
            Returns:
                preds (np.array(np.float64)): independant value predictions
        '''
        preds = np.array([])
        for tree in self.trees:
            delta = self.learning_rate * tree.predict(X)
            preds = delta if len(preds) == 0 else preds + delta

        preds = sigmoid(preds)
        if train:
            preds = (preds >= .5).astype(int)
        return preds