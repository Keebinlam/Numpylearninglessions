import opendatasets as od #to download datasets from online sources like google and kaggle
import os
import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import joblib
import jovian

dataset_url = 'https://www.kaggle.com/jsphyg/weather-dataset-rattle-package'
od.download(dataset_url)
data_dir = './weather-dataset-rattle-package'
os.listdir(data_dir)
train_csv = data_dir + '/weatherAUS.csv'
raw_df = pd.read_csv(train_csv)
#when thinking about problem for categorizing, If the problem is looking to assign an input to a discret category.
#If it needs to be categorized, this would fall in the classification question. (will it rain, or not rain) While with linear, its more like
#here is a patient with x,y,z, chararectoristics, what should we charge them for the bill
#Problems where a continuous numeric value must be predicted for each input are known as regression problems.
#when you think continuous value, think regression, when you think classification, think dicreet classes
#logicstic regression is commonly used in solving binary classification problems
#we take linear combination (or weighted sum of features)
# we apply the sigmoid function to the result to get a number between 0 and 1
# the number represent the probability of the input being 'Yes'
#instead of RMSE, we use the cross entrophy loss function to evaluate the results
#regression and classification are a part of supervised machine learning because they use labled data