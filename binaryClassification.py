# ----------------------
# Data preperation
# ----------------------

from Helpers.preprocessingHelpers import getData
import numpy as np

x_train, x_test, y_train, y_test = getData('newCombos.txt')

# ----------------------
# Baseline model
# ----------------------

import tensorflow as tf
from tensorflow.keras import layers

inputs = layers.Input(shape=(13,))
x = layers.Dense(13, activation="relu") (inputs)
x = layers.Dense(13, activation="relu") (x)
outputs = layers.Dense(1, activation="sigmoid") (x)

model_0 = tf.keras.Model(inputs, outputs)

model_0.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

print(model_0.summary())

history = model_0.fit(x_train,
                      y_train,
                      epochs=5)

tf_model_accuracy = model_0.evaluate(x_test, y_test)

# ----------------------
# Gradient Boosting
# ----------------------
from Objects.GradientBoosting.GradientBoostRegressor import GradientBoostRegressor
from Helpers.metricsHelpers import getAccuracy

gb = GradientBoostRegressor()
gb.fit(x_train, y_train)
preds = gb.predict(x_test)

gb_model_accuracy = getAccuracy(y_test, preds, rounded=False)

# ----------------------
# Gradient Boosting
# ----------------------
from Objects.ExtremeGradientBoosting.XGBoostRegressor import XGBoostRegressor
from Helpers.metricsHelpers import getAccuracy

xgb = XGBoostRegressor(n_learners=200)
xgb.fit(x_train, y_train)
preds = xgb.predict(x_test)

print(preds)

xgb_model_accuracy = getAccuracy(y_test, preds, rounded=False)

# ----------------------
# XGBoost
# ----------------------

import xgboost

xgb_model = xgboost.XGBClassifier()

xgb_model.fit(x_train, y_train)
a = xgb_model.predict(x_test)

xgboost_model_accuracy = getAccuracy(y_test, a)

# ----------------------
# Analysis
# ----------------------

print("TF Binary Classification: ", tf_model_accuracy[1])
print("Gradient Boosting model: ", gb_model_accuracy)
print("XGBoost From Scratch Boosting Model: ", xgb_model_accuracy)
print("XGBoost model: ", xgboost_model_accuracy)