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

# decision tree in general is a parlance represents a hieractical series of binary decisions
# it works then same, except we let the computer configure the optimal structure and hierarcgy instead of manual criteria
# root nodes are columns, leaf nodes are the last columns in the decision tree, root nodes leading to another node is a branch
# The tree is just the pattern from datas algorythm
# root node are determined by the gini index or information gain.
# know gini index and impurity and entropy with the information gain
# this the decision tree model we will be using, the 42 is crazy, its just a common nominclcher to use 42 because of hitch hikers guide to the galasy, we are able to replicate the 42 'seed'that way
model = DecisionTreeClassifier(random_state=42)
pd.value_counts(train_targets)
# the optimal decision tree has been created fronm the training data
model.fit(X_train, train_targets)
# it knows yes or no value because we fit the target in the model.
# we are fitting the model to give us a predictions
train_pred = model.predict(X_train)
pd.value_counts(train_pred)
# we are comparing the actual target values, with the values from the models predictions
accuracy_score(train_pred, train_targets)
# what we are seeing is the model is almost 100 percent accurate when predicting if it will rain or not with our training model
# To see how confident our mode is with deciding each input
train_prob = model.predict_proba(X_train)
train_prob
# Since its picking from 0-1, 1 being super confident yes and 0 being absolute no, our model is saying it is super sure one value or another
# we should probably check the data from our validation data to make sure 100 percent is correct
model.score(X_val, val_targets)
val_targets.value_counts()/len(val_targets)
# when we hear regularzation, it means we are trying to correct overfitting
# we can check the parameters. First is the criteria, with either gini or entory, which measures the loss
# max_depth, to reduce the total depth in the tree
# default criterion is gini
# and max_leaf_node
model1 = RandomForestClassifier()
model1.fit(X_train, train_targets)
train_pred1 = model1.predict(X_train)
# unlike the decision tree, it is scoring higher with the validation data.
model1.score(X_val, val_targets)
# lets visualize the tree
plt.figure(figsize=(80, 20))
plot_tree(model, feature_names=X_train.columns, max_depth=2, filled=True)
# here is to find out the importance of each feature
X_train.columns
model.feature_importances_
# lets make it into a dataframe so we can see it better
importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_}).sort_values(
    'importance', ascending=False)
importance_df.head(10)
plt.title('Feature Importance')
sns.barplot(data=importance_df.head(10), x='importance', y='feature')
