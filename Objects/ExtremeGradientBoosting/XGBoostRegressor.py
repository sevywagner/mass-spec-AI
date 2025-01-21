from sklearn.base import BaseEstimator
from Helpers.metricsHelpers import sigmoid, binaryCrossentropy
import numpy as np
from Objects.ExtremeGradientBoosting.DecisionTreeBooster import DecisionTreeBooster
from Objects.Model import Model
import math

class XGBoostRegressor(BaseEstimator, Model):
    def __init__(self, learning_rate=.01, n_learners=200, subsample=.25, max_depth=5, min_samples=2, gamma=0.0, reg_lambda=1.0):
        self.learning_rate = learning_rate
        self.n_learners = n_learners
        self.subsample = subsample
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.trees = []

    def get_gradients_hessians(self, y, y_pred):
        '''
            Function:
                Calculate gradients(first order derivative of the loss function)
                and hessians (second order derivative of the loss function)
            
            Parameters:
                y (np.array(np.int8)): true labels
                y_pred (np.array(np.float32)): predicted labels

            Returns:
                gradients, hessians (np.array(float32), np.array(float32)): the calculated gradients and hessians
        '''
        predictions = sigmoid(y_pred)
        gradients = predictions - y
        hessians = predictions * (1 - predictions)
        return gradients, hessians
    
    def fit(self, X, y, validation_set=None, tolerance=1e-4):
        '''
            Function:
                Train the XGBoost model using binary crossentropy loss function
            
            Parameters:
                X (np.array(np.array(np.float32))): data samples
                y (np.array(np.int8)): labels
        '''
        np.random.seed(42)
        fm = np.full(shape=len(y), fill_value=0.0)
        early_stop_ctr = 0
        loss = 0.

        if (validation_set):
            val_x, val_y = validation_set
            fm_val = np.full(shape=len(y), fill_value=0.0)

        for i in range(self.n_learners):
            mask = np.random.randint(0, len(fm), math.floor(self.subsample * len(fm)))
            gradients, hessians = self.get_gradients_hessians(y[mask], fm[mask])
            learner = DecisionTreeBooster(X[mask], gradients, hessians, 
                                          min_samples=self.min_samples, 
                                          learning_rate=self.learning_rate,
                                          gamma=self.gamma,
                                          reg_lambda=self.reg_lambda,
                                          subsample=self.subsample,
                                          max_depth=self.max_depth)
            
            fm += (self.learning_rate * learner.predict(X))
            self.trees.append(learner)
            
            if (validation_set):
                fm_val += (self.learning_rate * learner.predict(val_x))
                temp_loss = binaryCrossentropy(val_y, fm_val)
            else:
                temp_loss = binaryCrossentropy(y, fm)

            if (abs(loss - temp_loss) / max(abs(loss), 1e-7) < tolerance):
                early_stop_ctr += 1
            else:
                early_stop_ctr = 0

            if (early_stop_ctr == 3):
                print(f'Early stop invoked at ${i + 1} learners')
                break

            loss = temp_loss
            if ((i + 1) % 5 == 0):
                print(f'Learner #{i + 1} loss: {loss}')


    def predict(self, X):
        '''
            Function:
                Make predictions

            Parameters:
                X (np.array(np.array(np.float32))): data samples

            Returns:
                y_pred (np.array(np.float32)): predicted labels
        '''
        preds = np.array([])
        for tree in self.trees:
            delta = self.learning_rate * tree.predict(X)
            preds = delta if len(preds) == 0 else preds + delta

        return sigmoid(preds)