from Helpers.preprocessingHelpers import getData
from Helpers.metricsHelpers import getAccuracy, binaryCrossentropy
import numpy as np

x, y = getData('combFeatTrainCombos.txt')

def createGBClassifier(name, *params):
    from Objects.GradientBoosting.GradientBoostRegressor import GradientBoostRegressor
    '''
        Function:
            Create a gradient boosting classifier and save it to a file
        
        Parameters:
            name (str): name of the file to put the model in
    '''
    gb = GradientBoostRegressor(n_learners=1)
    fm, loss = gb.fit(X=x, y=y)
    gb.saveModel(f"./Models/GradientBoosting/{name}.pkl")
    # print("GB training loss: ", loss)

def createXGBClassifier(name):
    from Objects.ExtremeGradientBoosting.XGBoostRegressor import XGBoostRegressor
    '''
        Function:
            Create a gradient boosting classifier and save it to a file
        
        Parameters:
            name (str): name of the file to put the model in
    '''
    gb = XGBoostRegressor(n_learners=5, subsample=.3, learning_rate=.1)
    gb.fit(X=x, y=y)
    gb.saveModel(f"./Models/ExtremeGradientBoosting/{name}.pkl")
    # preds = gb.predict(x)
    # print('Val accuracy:', getAccuracy(y_test, preds, rounded=False))
    # print('Val loss:', binaryCrossentropy(y_test, preds))

def createTFModel(name):
    '''
    '''
    import tensorflow as tf
    from tensorflow.keras import layers

    inputs = layers.Input(shape=(5,))
    x = layers.Dense(16, activation="relu") (inputs)
    x = layers.Dense(32, activation="relu") (x)
    x = layers.Dense(16, activation="relu") (x)
    outputs = layers.Dense(1, activation="sigmoid") (x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['binary_accuracy'])
    
    model.fit(x, y, epochs=10)
    model.save(f'./Models/Regression/{name}.keras')

# createGBClassifier('combFeat')
createXGBClassifier('combFeat')
# createTFModel('newFeats')