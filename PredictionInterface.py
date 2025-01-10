from Objects.GradientBoosting.GradientBoostRegressor import GradientBoostRegressor
from Objects.ExtremeGradientBoosting.XGBoostRegressor import XGBoostRegressor
from Helpers.postprocessingHelpers import getMostLikelyCompounds, processMassValue
import os
import tensorflow as tf

def showAllModelsOfType(modelType):
    '''
        Function:
            Display all of the pretrained models of a certain type

        Parameters:
            modelType (int): an integer 1-3 the type of the model (1) GB (2) XGB (3) TF

        Returns:
            models (list(str)): filenames of the models
    '''
    models = []
    for i, name in enumerate(os.listdir(f'./Models/{folderNames[modelType - 1]}')):
        print(str(i + 1) + ': ', name.split('.')[0])
        models.append(name)

    return models

model = None
modelType = int(input('Select a model type:\n1. GB\n2. XGB\n3. TF\n'))
folderNames = ['GradientBoosting', 'ExtremeGradientBoosting', 'Regression']

models = showAllModelsOfType(modelType)
modelNum = int(input('Select a model: '))

path = f'./Models/{folderNames[modelType - 1]}/{models[modelNum - 1]}'
if modelType == 1:
    model = GradientBoostRegressor.loadModel(path)
elif modelType == 2:
    model = XGBoostRegressor.loadModel(path)
else:
    model = tf.keras.models.load_model(path)


choice = input('Choose input. (a) Mass Value (b) Graphs: ')

if (choice == 'a'):
    processMassValue(model, float(input("Enter a mass value")), 'comb')
else:
    getMostLikelyCompounds(model, 'peakData.txt', f'{folderNames[modelType - 1]}/{models[modelNum - 1].split('.')[0]}.txt')

# (NH4)C6H10O4+ 164.09
# (NH4)C9H14O2+ 172.132237