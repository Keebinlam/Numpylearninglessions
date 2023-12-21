# to download datasets from online sources like google and kaggle
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
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
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
import jovian

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

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

# now lets adjust our dataframe to make sure if there is any category data like (direction: north, west) or (is smoke: yes, no) to numrecical data (north = 1, smoking equal = 0)
# lets first seperate numrical values and catergorical values within the train inputs, because we know the training target df is just categorical column
numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()
# lets take a look at the new columns
train_inputs[numeric_cols].describe()
train_inputs[categorical_cols].nunique()
# now we can start processing the data, lets work out filling missing data
# replacing missing value with the average value in the colum using the imputation
imputer = SimpleImputer(strategy='mean')  # from scikitlearn

# lets check for missing data in our big data frame
# isna shows if a value is missing. the .sum is showing adding all the instances of n/a
raw_df[numeric_cols].isna().sum()

# we can do it again on the training input, but you can do it on all the sets
train_inputs[numeric_cols].isna().sum()
# SimpleImputer is going to look through all the columns in the data frame, its going to find the mean for each column.
imputer.fit(raw_df[numeric_cols])
# now we can see the calculation from the imputer
# these are the mean for each column listed out
list(imputer.statistics_)
# now we need to fill this information in to the training, validation and test set
# for this we will use 'transform'
train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])
# we can check if the values na has been filled with the mean
train_inputs[numeric_cols].isna().sum()
# now lets scale our data to a smaller range (0-1) scale to zero
# the reason why we need to do this is because data with a large range will tend to domintate the loss
scaler = MinMaxScaler()
# now we are fitting the scaling module to our data
scaler.fit(raw_df[numeric_cols])
# it is now scaled lets do it for our training, validiation, and test data
# from out data that has been seperated to numerical value and data that has been filled with the mean where value is na, we can tranform the whole data with the scaler
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])
# here is the new data after scaling it
train_inputs[numeric_cols]
# we did a good job preparing the numerical data, lets work on categorical data
# Lets encode catergorical data
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
# we are using the one hot encoder from scikit learn, and telling it to ignore values that are blank
encoder.fit(raw_df[categorical_cols])
# using the encoder on the raw dataframe for categorical data
encoder.categories_
# the encoder has created a list of each categories for each of the catergorical columns in the data set
# we can generate columns names for each individual catergory using  'get_feature_names_out '
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
print(encoded_cols)
# now lets transform the inputs with the encoded catergorical columns
train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])
pd.set_option('display.max_columns', None)
# this is to uncap the limit of columns you can see. You can also download the df to a csv and check the columns
train_inputs

# finally after all that, we are ready to train the logistic regression model
model = LogisticRegression(solver='liblinear')

model.fit(train_inputs[numeric_cols + encoded_cols], train_targets)
print(numeric_cols + encoded_cols)
# scikit learn can have targets as categorical data, but inputs must be numerical
print(model.coef_.tolist())
print(model.intercept_)
# you can interpret the data at this point based on the weights. Like (maxtemp = -2.879 coef) which means the higher the temp, the less likely it will rain
# or (windguspeed: 6.7644) showing that the more windy it is, the more likely it will rain tomorrow.p
dasd = pd.DataFrame({
    'feature': (numeric_cols + encoded_cols),
    'weight': model.coef_.tolist()[0]
})
# making prediction and evaluating the model
# we created these variables since were using [numeric_cols + encoded_cols] columns a lot, this will help organize when we use each of the models for predictions
x_train = train_inputs[numeric_cols + encoded_cols]
x_val = val_inputs[numeric_cols + encoded_cols]
x_test = test_inputs[numeric_cols + encoded_cols]
# here is the predict function for the train data
train_pred = model.predict(x_train)
# this what how model predicts the value to be (Based off training data)
train_pred,
# this is what the actual results from that data (Based off training data)
train_targets
# now we need to check for accuracy, by comparing the results of the pred and targer data
# for this we can use 'accuracy_score' by scikit learn (plug in target first, then prediction)
accuracy_score(train_targets, train_pred)
# it has a 85 percent accuracy!
# but for logistic regression, we can get the probablity, which are yes and no's
train_probs = model.predict_proba(x_train)
train_probs  # its the probablity of it being a no or a yes
# we can visualize the the accurate by using the confusion matrix to breakdown correct and incorrect inouts aginast the actual and predicted
confusion_matrix(train_targets, train_pred, normalize='true')

# TN, FP
# FN, TP

# depending on the problem, its better to optimize for FN or FP, than total accuracy of the model
# the breakdown, 94 percent of the time, when is it not going to rain, the modeal guess it right, and only 5 percent of the time does it get it wrong
# however, in 47 percent of the time when it is raining, the model get it wrong and said its not raining. So not good


def predict_and_plot(inputs, targets, name=''):
    preds = model.predict(inputs)

    accuracy = accuracy_score(targets, preds)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

    cf = confusion_matrix(targets, preds, normalize='true')
    plt.figure()
    sns.heatmap(cf, annot=True)
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('{} Confusion Matrix'.format(name))

    return preds


train_preds = predict_and_plot(x_train, train_targets, 'Training')

val_pred = model.predict(x_val)
val_pred
accuracy_score(val_targets, val_pred)
test_pred = model.predict(x_test)
test_pred
accuracy_score(test_targets, test_pred)
# what we know now is that since the validation data, and the test data are so close to the accuracy of the training data, we say the training is 84 percent accurate
# now you want to test the accurate against other dumb models. Like have a model with just 'no' will it be better when we just say no for rain everyday?
# or try against a human, have a meteorlist go through the data, and for 100 days, guess if it will rain tomorrow. if they score less than the model, we can saw that the AI we built is better than a human
# now lets make a new input prediction
new_input = {'Date': '2021-06-19',
             'Location': 'Katherine',
             'MinTemp': 23.2,
             'MaxTemp': 33.2,
             'Rainfall': 10.2,
             'Evaporation': 4.2,
             'Sunshine': np.nan,
             'WindGustDir': 'NNW',
             'WindGustSpeed': 52.0,
             'WindDir9am': 'NW',
             'WindDir3pm': 'NNE',
             'WindSpeed9am': 13.0,
             'WindSpeed3pm': 20.0,
             'Humidity9am': 89.0,
             'Humidity3pm': 58.0,
             'Pressure9am': 1004.8,
             'Pressure3pm': 1001.5,
             'Cloud9am': 8.0,
             'Cloud3pm': 5.0,
             'Temp9am': 25.7,
             'Temp3pm': 33.0,
             'RainToday': 'Yes'}
# now we have to apply the pre-processing steps for our data, to the new inputs
# 1) imputations of missing values using 'imputer' imputer = SimpleImputer(strategy = 'mean')
# 2) Scale the data scaler = MinMaxScaler()
# 3) encode the categories using the encoder encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
# lets first make the new data into a dataframe
new_input_df = pd.DataFrame([new_input])
# now lets plug in out 3 steps
new_input_df[numeric_cols] = imputer.transform(new_input_df[numeric_cols])
new_input_df[numeric_cols] = scaler.transform(new_input_df[numeric_cols])
new_input_df[encoded_cols] = encoder.transform(new_input_df[categorical_cols])
X_new_input = new_input_df[numeric_cols + encoded_cols]
X_new_input
prediction = model.predict(X_new_input)[0]
prediction
prob = model.predict_proba(X_new_input)[0]
# here is the function to help with new inputs


# a function called predict input, that will give his a yes or now, and the chances of that yes or no
def predict_input(single_input):
    # when we get an input, turn it into a dataframe called 'input_df' so now all inputs will be called this when pushed through
    input_df = pd.DataFrame([single_input])
    # from the DF, fill in all none values with the mean for that column. The mean was calculated when we created the imputer.fit(raw_df[numeric_cols]) which is taking each column, from our original dataset, and getting te mean for all of them. Using transform, we will back into the DF
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    # after all the empty fields are in, we need to scale the data to sum of 0, the scaler is taking all numerical columns from our original dataset, and uses the min-max scaling (min-max normalization) method on all the values in the data to 0 or 1
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    # now to handle the categorical data, at this more input_df has data filled, and has been normalized to 0 to 1, We created an encorder, that takes all the unique object values and creates a new column for each. It's vectorized, meaning it will be given a 1 based of the orginal data, and 0 for the columns that it is not.
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    # we put together the new columns, and data pre processing into one nice varible
    X_input = input_df[numeric_cols + encoded_cols]
    # we run the model prediction by fitting the into scikit learn
    pred = model.predict(X_input)[0]
    # do the same to get a probablilty
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    return pred, prob  # display the new input


# inside of 'model' contains this "train_inputs[numeric_cols + encoded_cols], train_targets)"
# which is our complete ML model using the training data that was given to us to use to make this model.
new_input2 = {'Date': '2021-06-19',
              'Location': 'Launceston',
              'MinTemp': 23.2,
              'MaxTemp': 33.2,
              'Rainfall': 10.2,
              'Evaporation': 4.2,
              'Sunshine': np.nan,
              'WindGustDir': 'NNW',
              'WindGustSpeed': 52.0,
              'WindDir9am': 'NW',
              'WindDir3pm': 'NNE',
              'WindSpeed9am': 13.0,
              'WindSpeed3pm': 20.0,
              'Humidity9am': 89.0,
              'Humidity3pm': 58.0,
              'Pressure9am': 1004.8,
              'Pressure3pm': 1001.5,
              'Cloud9am': 8.0,
              'Cloud3pm': 5.0,
              'Temp9am': 25.7,
              'Temp3pm': 33.0,
              'RainToday': 'Yes'}
predict_input(new_input2)
input_cols = list(train_df.columns)[1:-1]
target_col
# you assume a certain relationship between the inputs, and the output. This is called the model. The model has inherent parameter called weights which are intially randomized.
# then you train the model which means you make prediction and compare them with actual data. This is where you need labeled data
# then from the comparision, you compute the loss, and you reduce the loss using an optimation technique
# then you can use is after logiciall evaulting it
# Saving the model, lets first creat a dictionary containing all the required objects
# this is so we can send the model to a developer so that they can make it into whatever it needs to be
# They do not need to run the model, but will use the coeffiecnt, the intercept, sigmoid to work as a standalone application
# Giving them that, is almost like giving them a dense representation of all the intelligence that is present in the model

aussie_rain = {
    'model': model,
    'imputer': imputer,
    'scaler': scaler,
    'encoder': encoder,
    'input_cols': input_cols,
    'target_col': target_col,
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'encoded_cols': encoded_cols
}
# we are taking all the backend data and functions hidden in these keys,and transfering them
# the model intellignce is dirived from the weights

___________________________

# Download the dataset
# This is using the open dataset module. To download dataset from google and kaggle
od.download('https://www.kaggle.com/jsphyg/weather-dataset-rattle-package')
# after downloadeding the data, we use panda to read the file and put it into a varible
raw_df = pd.read_csv('weather-dataset-rattle-package/weatherAUS.csv')
# Since we are determining the the target as 'rain tomorrow' we drop this column so the ML model is not trained on it
raw_df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)

# Create training, validation and test sets
# We start to segment our data, break them apart by year to prepare the training, test and validation data set
year = pd.to_datetime(raw_df.Date).dt.year
train_df, val_df, test_df = raw_df[year <
                                   # train data is 2015 and news, valudation is just 2015, and our test data is anything older than 2015
                                   2015], raw_df[year == 2015], raw_df[year > 2015]

# Create inputs and targets
# now that we data has been split, we can focus on just the training data and apply it to the rest later. Here we are setting up the columns that the models will be trained on. We exclude the date, and the rain tomorrow from our input
input_cols = list(train_df.columns)[1:-1]
# how target column is the column we want out ML to predict, so we signify it by putting it in a varible we can call
target_col = 'RainTomorrow'
train_inputs, train_targets = train_df[input_cols].copy(
    # here taking the assigned columns varibles, and applying them to our training data. Letting out system know, 'training input = from training data, apply just the input columns that i listed', we do the same for target column in the training data
), train_df[target_col].copy()
# we apply the column list for input and targets to our validation dataset
val_inputs, val_targets = val_df[input_cols].copy(), val_df[target_col].copy()
test_inputs, test_targets = test_df[input_cols].copy(  # and lastly the test set
), test_df[target_col].copy()

# Identify numeric and categorical columns
numeric_cols = train_inputs.select_dtypes(  # now within out inputs, we have a mix of numbers and strings. Well need to sepearate those for the mode to work.
    # we first say; from the training input data set, we want only the numbers in this variables
    include=np.number).columns.tolist()[:-1]
# from the training data set for inputs, grab all the catergories
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()

# Impute missing numerical values
# we are now taking the original data source, and getting the mean for each column.
imputer = SimpleImputer(strategy='mean').fit(raw_df[numeric_cols])
# inside the training data for inputs, for valye inside the number columns, if anythings is missing, please fill that data with the mean
train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
# we do the same for the validation and test set
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

# Scale numeric features
# now we scale the data to range 0 and 1 to nornalize our data
scaler = MinMaxScaler().fit(raw_df[numeric_cols])
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

# One-hot encode categorical features
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore').fit(
    # for this part, we are making an a new column for every unique value from the object column. We then vectorize it, indicating 1 as itself, and a 0 for where it is not.
    raw_df[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
# we have to apply this tot he training set within the catergory columns varibles
train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
# we have to apply this tot he training set within the validation columns varibles
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
# we have to apply this tot he training set within the test columns varibles
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

_____________

# Select the columns to be used for training/prediction
# here we finally filled in all the data that are missing in the number columns, and scaled them to 0-1, and combine them with the new category columns that has been vectorizes
X_train = train_inputs[numeric_cols + encoded_cols]
# since all the columns and preprocessing steps has been done, we can apply these steps to our vallidation data to get a full validation input
X_val = val_inputs[numeric_cols + encoded_cols]
# and now we have a full test input with all the pre processing
X_test = test_inputs[numeric_cols + encoded_cols]

# Create and train the model
# we call in a varible that houses our model. The model will take our inputs and targets for each dataset. Run the math behind it, and optimize it automatcally. It will then hold that functionalyty in the 'model' subvalue
model = LogisticRegression(solver='liblinear')
# we input our preprocessed inputs from our training data set, and the target set that we plucked for the begining since it does not need to be preprocessed. into the model.
model.fit(X_train, train_targets)

# Generate predictions and probabilities
# The model is now done, and houses stuff like the coeffiecnt(weight), the intercept (bias) for each column in our training data. But we create a new varible that hold the prediction function for the model.
train_preds = model.predict(X_train)
# this line of code is using the model to make predictions on the X_train dataset, and then storing those predictions in the train_preds variable.
train_probs = model.predict_proba(X_train)
# From the data and the model it holds. it is making a prediction be on out original training data, and predicting if it will rain or not. We can compare the actual data we have, to what it guessed per row and see how often it is right and wrong
accuracy_score(train_targets, train_preds)

# Helper function to predict, compute accuracy & plot confustion matrix


def predict_and_plot(inputs, targets, name=''):
    preds = model.predict(inputs)
    accuracy = accuracy_score(targets, preds)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    cf = confusion_matrix(targets, preds, normalize='true')
    plt.figure()
    sns.heatmap(cf, annot=True)
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('{} Confusion Matrix'.format(name))
    return preds

# Here is a function that handled everything, taking all the steps we made and putting it into a function that we can apply on new data


def predict_input(single_input):
    # when we get an input, turn it into a dataframe called 'input_df' so now all inputs will be called this when pushed through
    input_df = pd.DataFrame([single_input])
    # from the DF, fill in all none values with the mean for that column. The mean was calculated when we created the imputer.fit(raw_df[numeric_cols]) which is taking each column, from our original dataset, and getting te mean for all of them. Using transform, we will back into the DF
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    # after all the empty fields are in, we need to scale the data to sum of 0, the scaler is taking all numerical columns from our original dataset, and uses the min-max scaling (min-max normalization) method on all the values in the data to 0 or 1
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    # now to handle the categorical data, at this more input_df has data filled, and has been normalized to 0 to 1, We created an encorder, that takes all the unique object values and creates a new column for each. It's vectorized, meaning it will be given a 1 based of the orginal data, and 0 for the columns that it is not.
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    # we put together the new columns, and data pre processing into one nice varible
    X_input = input_df[numeric_cols + encoded_cols]
    # we run the model prediction by fitting the into scikit learn
    pred = model.predict(X_input)[0]
    # do the same to get a probablilty
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    return pred, prob  # display the new input


# Evaluate on validation and test set
val_preds = predict_and_plot(X_val, val_targets, 'Validation')
test_preds = predict_and_plot(X_test, test_targets, 'Test')

# lastly, we would run the through all of it again for the validation and test set to see how close we are and to check if our model is incorrect.
