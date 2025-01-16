# ----------------------
# Data preperation
# ----------------------

from preprocessingHelpers import getData
import numpy as np

x_train, y_train = getData('combFeatTrainCombos.txt')
x_test, y_test = getData('combFeatTestCombos.txt')


# ----------------------
# Tuning
# ----------------------

from sklearn.model_selection import GridSearchCV, KFold
from XGBoostRegressor import XGBoostRegressor

cross_val = KFold(n_splits=5, shuffle=True, random_state=42)
hyper_param_ranges = {
    'learning_rate': [.1, .01],
    'n_learners': [100, 200]
}

model = XGBoostRegressor()
tuned_params = GridSearchCV(model,
                            hyper_param_ranges,
                            scoring='f1',
                            cv=cross_val,
                            n_jobs=1)

tuned_params.fit(X=x_train, y=y_train)

print(tuned_params.best_params_)
print(tuned_params.best_score_)