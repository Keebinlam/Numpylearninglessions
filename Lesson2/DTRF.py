# all the imports set up
import jovian
import pandas as pd
import numpy as np
import opendatasets as od
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

# decision tree and random forest powerful and will be the most ML models I will be using
jovian.commit(filename='DTRF.ipynb')
od.download('https://www.kaggle.com/jsphyg/weather-dataset-rattle-package')
os.listdir('weather-dataset-rattle-package')
raw_df = pd.read_csv('weather-dataset-rattle-package/weatherAUS.csv')

# We are dropping the rows of data in the target column where the value is null
raw_df.dropna(subset=['RainTomorrow'], inplace=True)
# converting the year to an interger so we can segment the data to training, val and test
year = pd.to_datetime(raw_df.Date).dt.year
# separting the data
train_df = raw_df[year > 2015]
val_df = raw_df[year == 2015]
test_df = raw_df[year < 2015]

# creating the input and target columns for the dataset
input_cols = list(train_df.columns)[1:-1]
target_col = 'RainTomorrow'

# now we apply the column split for the training data set
train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()

val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()

test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()

# from the input columns, we need to now break those into numeric and categorical columns
# since all the columns are the same for each dataset, it doesnt matter where it is being referenced from, so lets use train_inputs for all the datasets
numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()

imputer = SimpleImputer(strategy='mean').fit(raw_df[numeric_cols])
scaler = MinMaxScaler().fit(raw_df[numeric_cols])
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore').fit(
    raw_df[categorical_cols])

# double checking the new list of num columns #floats64, int 64
print(numeric_cols)
# double checking the new list of cat columns #objects, string
print(categorical_cols)

# now we want to impute missing numeric values. lets check what is missing first
train_inputs[numeric_cols].isna().sum()

# here we are writing over our orginal train input with numeric columns, to train input numeric columns with the imputer values for missing values
train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])

val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

# now lets take a look at the missing values
train_inputs[numeric_cols].isna().sum()

# much like the imputer, we do the same thing with scaler. Make sure to call the fit
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])

val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

# Now we need to process the categorical columns, can first we need to fill in missing data for the cat cols
# here we create the encoded list first
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

# then we add the new list to a new varible for catergories
train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])

val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

X_train = train_inputs[numeric_cols + encoded_cols]

X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]
# decision tree in general is a parlance represents a hieractical series of binary decisions
