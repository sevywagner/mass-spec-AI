from Objects.ExtremeGradientBoosting.XGBoostRegressor import XGBoostRegressor
from Helpers.preprocessingHelpers import getData
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

x, y = getData('combFeatTestCombos.txt')

model = XGBoostRegressor.loadModel('./Models/ExtremeGradientBoosting/test.pkl')
# model = XGBoostRegressor.loadModel('./Models/ExtremeGradientBoosting/combFeat-default.pkl')
# model = XGBoostRegressor.loadModel('./Models/GradientBoosting/combFeat.pkl')
# model = XGBoostRegressor.loadModel('./Models/GradientBoosting/combFeat-default.pkl')

preds = np.round(model.predict(x))
cm = confusion_matrix(y, preds)
print('Accuracy: ', accuracy_score(y, preds))
print(cm)