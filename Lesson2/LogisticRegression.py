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
raw_df.info() #to check if there is any missing data
#for 'rain today, and rain tomorrow, there is a lot of missing data. So we want to only use records that has rain today and tomorrow for our mode
raw_df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True) #were dropping data where rain today and raintomorrow are empty
raw_df.info() #here the total entries has dropped but rain today and rain tomorrow are equal
#lets take a look at some data
px.histogram(raw_df, x='Location', color='RainToday')
px.scatter(raw_df.sample(2000), 
           title='Min Temp. vs Max Temp.',
           x='MinTemp', 
           y='MaxTemp', 
           color='RainToday')
px.bar(raw_df.sample(2000), x='Location', y='Rainfall')
px.histogram(raw_df, x='Temp3pm', color='RainTomorrow')
px.histogram(raw_df, x='Temp9am', color='RainTomorrow')
px.histogram(raw_df, x='RainToday', color='RainTomorrow') #this chart tells us for example, 92k instances that when it does not rain today, it also do not rain tomorrow
#also an equal split where yes it rained today, but will or will not rain tomorrow. 