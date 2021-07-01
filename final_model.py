import warnings
warnings.filterwarnings("ignore")
from xgboost import XGBClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
import pandas as pd
from google.colab import drive
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
import numpy as np

def load_test_data():
  #import test dataset
  path_test = "/content/drive/MyDrive/ml/test_data.csv"
  test_data = pd.read_csv(path_test, header=None)

  return test_data

def load_train_data():
  #import train dataset
  path_train = "/content/drive/MyDrive/ml/train_data.csv"
  train_data = pd.read_csv(path_train, header=None)

  return train_data

def load_labels():
  #import labels
  path_labels = "/content/drive/MyDrive/ml/train_labels.csv"
  labels = pd.read_csv(path_labels, header=None)

  return labels

# print on screen result of score, confusion matrix and classification report
def baseline(X_train, X_test, y_train, y_test):
  dummy_clf = DummyClassifier()
  dummy_clf.fit(X_train, y_train)
  score_dummy=(dummy_clf.score(X_train, y_train))  
  pred_dummy = dummy_clf.predict(X_test) 
  print(f'Baseline score: {score_dummy}')

  result_confusion_matrix = confusion_matrix(y_test, dummy_clf.predict(X_test))
  print(f'confusion_matrix:\n{result_confusion_matrix}')

  result_classification_report = classification_report(y_test,pred_dummy)
  print(f'classification_report:\n{result_classification_report}')

# train model and print on screen confusion matrix and classification report
def training_model(X_train, X_test, y_train, y_test):
  training_model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                colsample_bynode=1, colsample_bytree=1, gamma=0,
                learning_rate=0.1, max_delta_step=0, max_depth=3,
                min_child_weight=1, missing=None, n_estimators=50, n_jobs=1,
                nthread=None, objective='binary:logistic', random_state=0,
                reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                silent=None, subsample=1, verbosity=1)
  training_model.fit(X_train, y_train)
  # make predictions for test data
  training_y_pred = training_model.predict(X_test)

  result_confusion_matrix = confusion_matrix(y_test, training_y_pred)
  print(f'confusion_matrix:\n{result_confusion_matrix}')

  result_classification_report = classification_report(y_test,training_y_pred)
  print(f'classification_report:\n{result_classification_report}')

def predict_model(X_train, y_train, test_data):
  model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                colsample_bynode=1, colsample_bytree=1, gamma=0,
                learning_rate=0.1, max_delta_step=0, max_depth=3,
                min_child_weight=1, missing=None, n_estimators=50, n_jobs=1,
                nthread=None, objective='binary:logistic', random_state=0,
                reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                silent=None, subsample=1, verbosity=1)
  # make predictions for test data
  model.fit(X_train, y_train)
  y_pred = model.predict(test_data)

  return y_pred

# save dato to csv file
def save_data(data, file_name):
  # create dataframe of data
  df_data = pd.DataFrame(data)
  # saving the dataframe 
  df_data.to_csv(file_name)

def Main():

  drive.mount('/content/drive')
  
  test_data = load_test_data()

  train_data = load_train_data()

  labels = load_labels()

  #assign X, y to data
  X, y = train_data, labels

  #test train split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

  print("Baseline\n")
  baseline(X_train, X_test, y_train, y_test)

  print("Training model\n")
  training_model(X_train, X_test, y_train, y_test)

  y_pred = predict_model(X_train, y_train, test_data)

  save_data(y_pred, 'y_pred.csv')

Main()