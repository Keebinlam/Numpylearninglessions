# to download datasets from online sources like google and kaggle
import opendatasets as od
import os
import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
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
# when thinking about problem for categorizing, If the problem is looking to assign an input to a discret category.
# If it needs to be categorized, this would fall in the classification question. (will it rain, or not rain) While with linear, its more like
# here is a patient with x,y,z, chararectoristics, what should we charge them for the bill
# Problems where a continuous numeric value must be predicted for each input are known as regression problems.
# when you think continuous value, think regression, when you think classification, think dicreet classes
# logicstic regression is commonly used in solving binary classification problems
# we take linear combination (or weighted sum of features)
# we apply the sigmoid function to the result to get a number between 0 and 1
# the number represent the probability of the input being 'Yes'
# instead of RMSE, we use the cross entrophy loss function to evaluate the results
# regression and classification are a part of supervised machine learning because they use labled data
raw_df.info()  # to check if there is any missing data
# for 'rain today, and rain tomorrow, there is a lot of missing data. So we want to only use records that has rain today and tomorrow for our mode
# were dropping data where rain today and raintomorrow are empty
raw_df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)
raw_df.info()  # here the total entries has dropped but rain today and rain tomorrow are equal
# lets take a look at some data
px.histogram(raw_df, x='Location', color='RainToday')
px.scatter(raw_df.sample(2000),
           title='Min Temp. vs Max Temp.',
           x='MinTemp',
           y='MaxTemp',
           color='RainToday')
px.bar(raw_df.sample(2000), x='Location', y='Rainfall')
px.histogram(raw_df, x='Temp3pm', color='RainTomorrow')
px.histogram(raw_df, x='Temp9am', color='RainTomorrow')
# this chart tells us for example, 92k instances that when it does not rain today, it also do not rain tomorrow
px.histogram(raw_df, x='RainToday', color='RainTomorrow')
# also an equal split where yes it rained today, but will or will not rain tomorrow.
use_sample = False  # when working with a large dataset, its good idea to work with sample
# set up these three lines, so that way if you want to use the full data set, change use_sample to = False, you can switch the 'use sample' to True to  use the sample size
sample_size = .1
if use_sample:
    # this is taking a 1400 size size to help run the program faster
    raw_df = raw_df.sample(frac=sample_size).copy()
    # learning best practices, 1) split data into a parts.
# first is the training set, used to train the model
# second the validation set, uses to evaluate the model, and help tune the parameters
# third the test set,  used to compare different models. Do not use test set in training or validation
# so in typical fashion, you will use 60 percent of the data to do training and 20 percent for validation. But 20 percent will let left out to test by running different models and approachs

# scikit learn has a build in data splitter for training and test sets
# random state will pick from a random generator from 0-42 to pick from.
# (test data frame)20 percent is of the data is put into 'test_df" the rest will be in train_val_df
train_val_df, test_df = train_test_split(
    raw_df, test_size=0.2, random_state=42)
# (training data) taking the valaues that is left after taking out from 'test_df)
train_df, val_df = train_test_split(
    train_val_df, test_size=0.25, random_state=42)
# here is the breakdown of the data sizes
print('train_df.shape :', train_df.shape)
print('val_df.shape :', val_df.shape)
print('test_df.shape :', test_df.shape)
# when working with dates, there is some considerations
# its sometimes better to separate the training, validation, test set with time.
# so that the model is trained on data from the bast and evalyared on data from the future
# its good practice to take the data from the last date or jst before that as the validiation set
year = pd.to_datetime(raw_df.Date).dt.year

train_df = raw_df[year < 2015]  # training data from 2008 to 2014
val_df = raw_df[year == 2015]  # validation data on data in 2015
test_df = raw_df[year > 2015]  # testing data on data from 2016 and newer

# Were going to ignore the previous data splitting from scikit learn, and split the data by date
print('train_df.shape :', train_df.shape)
print('val_df.shape :', val_df.shape)
print('test_df.shape :', test_df.shape)
train_df
val_df
test_df
# next step in machine learning is identifying inputs and targets
# not all data is important so its good to remove data from the model
# here we are making inputs columns, ignoreing the date, and ignore 'rain tomorrow'
inputs_cols = list(train_df.columns)[1:-1]
target_col = 'RainTomorrow'
print(inputs_cols)
print(target_col)
# now we need to apply the input columns to each of the data set
# we also have a copy so we do not effect the original data
# lets start with training data
train_inputs = train_df[inputs_cols].copy()
train_targets = train_df[target_col].copy()
# apply inoputs and targets for validation data
val_inputs = val_df[inputs_cols].copy()
val_targets = val_df[target_col].copy()
# apply inoputs and targets for test data
test_inputs = test_df[inputs_cols].copy()
test_targets = test_df[target_col].copy()
