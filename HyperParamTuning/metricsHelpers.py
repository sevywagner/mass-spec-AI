import numpy as np
from chemHelpers import calculateMass

def getAccuracy(y, y_preds, rounded=True):
    if not rounded:
        y_preds = np.round(y_preds)
    
    mask = y == y_preds
    return len(y[mask]) / len(y)

def binaryCrossentropy(y, y_pred):
    '''
        Function:
            Binary Crossentropy loss function

        Parameters:
            y (np.array(int)): true y values
            y_pred (np.array(float)): predicted y values

        Returns:
            loss (float): loss
    '''
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def sigmoid(arr):
    '''
        Function:
            Turn an array into an array of probabilities between 0 and 1

        Parameters:
            arr (np.array(np.float64)): list of floats
        
        Return:
            probablities (np.array(np.float64)): list of probabilites
    '''
    return 1 / (1 + np.exp(-arr))

def calculatePPM(compound, mass):
    return float((mass - calculateMass(compound, encoded=True)) / mass) * 1000000