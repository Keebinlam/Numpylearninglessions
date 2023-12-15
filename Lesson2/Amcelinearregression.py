# because we are taking information from a url, using urlretrieval will allow us to pull a url
# pandas to read the csv
# numpy to calucuate data
#because we are taking information from a url, using urlretrieval will allow us to pull a url 
#pandas to read the csv
#numpy to calucuate data 
from urllib.request import urlretrieve
import pandas as pd
import numpy as np
#to save data
import jovian
#to visualize the data
import plotly.express as px #used to interactive visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
#to use the linear regression/sgdr model from scikitlearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn import preprocessing #for vecoring columns "one hot"
from sklearn.preprocessing import StandardScaler #making all the columns center to zero
#this will make sure the notebook is not a pop up, and that charts will be an output. That way the information doesnt go away


# start with getting the data from the url
medicaldataurl = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'

# we'll want to extract the data from the url, and make it into a CSV so pandas can read it
urlretrieve(medicaldataurl, 'acmemedical.csv')

# now that we have the amcemedical data saved as a csv, well need to turn it into a df using pandas
acemedicaldf = pd.read_csv('acmemedical.csv')

# to display the acme dataframe
print(acemedicaldf)

# we can check other things about our data, to make sure its nice and clean. Infor is to show the types of data, and know if there is any missing data
# decribe gives us aggregate insights about the data
acemedicaldf.info()
acemedicaldf.describe()

# The following settings will improve the default style and font sizes for our charts.
# only applys to matplotly and seaborn and not plotly
# Sets the background style of the plots to a dark theme with grid lines using Seaborn, a visualization library.
sns.set_style('darkgrid')
# Adjusts the default font size for text in Matplotlib plots to 14 points.
matplotlib.rcParams['font.size'] = 14
# Configures the default size of Matplotlib figures to 10 inches wide and 6 inches tall.
matplotlib.rcParams['figure.figsize'] = (10, 6)
# Sets the background color of the figure to transparent (denoted by '#00000000') in Matplotlib plots.
matplotlib.rcParams['figure.facecolor'] = '#00000000'

# we have everything set up, its time to break up some of the columns to get an idea of how the data is being presented
# the first one is age
acemedicaldf.age.describe()

# now we can visualize the age data
# we create a varible that stores the data we pull from, the style, adding title and number of bins
figure_age = px.histogram(
    acemedicaldf, x='age', marginal='box', nbins=47, title='Distribution of age',)
figure_age.update_layout(bargap=0.1)  # to change the gap between the bars
figure_age.show()  # to show the new graph

# I set a new varible for the bmi chart to make it easy to refer
figure_bmi = px.histogram(acemedicaldf, x='bmi', marginal='box', color_discrete_sequence=[
                          # we did the same this as in age, except changing the color of the bars red
                          'red'], title='Distribution of bmi')
figure_bmi.update_layout(bargap=0.1)
figure_bmi.show()

# whats cool about this part, is now we are assignined a color grey to smokers. and green as none smoker. So this chart will show charge ditrubituin, as well as showing if they smoke or now.
figure_charges = px.histogram(acemedicaldf, x='charges', marginal='box', color='smoker',
                              color_discrete_sequence=['green', 'grey'], title='Distribution of charges')
figure_charges.update_layout(bargap=0.1)
figure_charges.show()
# in this case, we see that none smoker are charged less, while smokers often gets chargered more

# get the total number of smoker to none smoke
# from datafrom, on the 'smoker column', counter the values
acemedicaldf.smoker.value_counts()

px.histogram(acemedicaldf, x='smoker', color='sex', title='Smoker')

# this one is for fun, but in this chart we get to see the number of time a gender is charges for a certain range.
px.histogram(acemedicaldf, x='charges', color='sex',
             color_discrete_sequence=['pink', 'green'])

figure_region = figure_bmi = px.histogram(acemedicaldf, x='region', marginal='box', color='sex', color_discrete_sequence=[
                                          'red', 'blue'], title='Distribution of region and gender')
figure_region.update_layout(bargap=0.1)
figure_region.show()
# this chart shows how many of each gender by region

fig_age_charge = px.scatter(acemedicaldf, x='age', y='charges')
fig_age_charge.show()

# now lets check on charges against bmi
fig_bmi_charges = px.scatter(acemedicaldf, x='bmi', y='charges',
                             color='smoker', hover_data=['sex'], title='bmi vs charge')
# from what it looks like, the bmi is not a major factor of price increase, but here again, it shows smoking is a major indicator of cost
fig_bmi_charges.update_traces(marker_size=5)
# we do noticed a tick up at 30, if you are smoking
fig_bmi_charges.show()

# lets try some violin graphs. It has this unique attribue to visualize mass of similar counts for a variable
px.violin(acemedicaldf, x='region', y='charges')
# from this graph, it shows that there are a lot of people paying between 5k to 11k for all region

sns.barplot(acemedicaldf, x='children', y='charges')

# okay now we are going into correlation coeffiencnt, meaning the strength and direction of a relationship between 2 varible. The closer to 1, the stronger, the closer to 0, the weaker or no corrleation
# keep in mind it does not work well outside of linear relationships, outliers effect it, and does not imply causeation
# now we can compute correlation using pandas
acemedicaldf.charges.corr(acemedicaldf.age)
# here we are computing the correlation coeffeient of charges from the dataframe, to age, from the same data frame

# we can check the corr for bmi as well
acemedicaldf.charges.corr(acemedicaldf.bmi)
# value returned is the correclation coaeffient

# lets check the corrlation between age of a person and if the number of children
acemedicaldf.age.corr(acemedicaldf.children)

# now I want to check the smoker and the charges correlation, but smoker is in a "yes or no' format, we can change that
smoker_values = {'no': 0, 'yes': 1}
smoker_numeric = acemedicaldf.smoker.map(smoker_values)
acemedicaldf.charges.corr(smoker_numeric)

# this part was tricky because it didnt follow the same as the guide
# but to prepare the .corr to run against all column, we first have to state to only gather columns with data type = number
# from acemedicaldf, only select columns with numbers
numeric_df = acemedicaldf.select_dtypes(include=['number'])
# from all number columns is selected, THEN do the correlation coeffiecent against all number columns
correlation_matrix = numeric_df.corr()
print(correlation_matrix)

# this is a heatmap from seasborn displaying the corrleation coeffiencent of all numberical columns in the acemedicaldf data frame
sns.heatmap(correlation_matrix, cmap='Reds', annot=True)
plt.title('Correlation Matrix')

# do not need to use a Linear regression Single Feature, or Loss/Cost Function, just use scikit learn
# in geometri, y = mx +b. (m being the slope, and b being the interceptor: or the point that the slope starts at Y =0.
# but in ML, m = w which means weigh, and b is bias. This is just a linear equation
# We need a way to measure numerically how well the line fits the points.
# Once the "measure of fit" has been computed, we need a way to modify w and b to improve the the fit.
# If we can solve the above problems, it should be possible for a computer to determine w and b for the best fit line, starting from a random guess.

non_smoker_df = acemedicaldf[acemedicaldf.smoker == 'no']


def test(w, x, b):
    return w * x + b


x = non_smoker_df.age
w = 100
b = 200

estimated_charges = test(w, x, b)

plt.plot(x, estimated_charges, 'r-o')
plt.xlabel('Age')
plt.ylabel('Estimated Charges')

target = non_smoker_df.charges

plt.plot(x, estimated_charges, 'r', alpha=0.9)
plt.scatter(x, target, s=8, alpha=0.8)
plt.xlabel('Age')
plt.ylabel('Charges')
plt.legend(['Estimate', 'Actual'])

x = non_smoker_df.age
target = non_smoker_df.charges
estimated_charges = test(w, x, b)

# to summarize what just happen, I created a graph that plots the acme data, as well as a linear function (not actual linear line from correlation coeffiecnt
# then created a function showing that takes in x and b paremeters, into a linear equation. The purpose of this is to plug in different numbers, and see how the line moves
# the goal is to get the link to match as close to the general sloped line of the acme data
# now we need to know how to measure how accurate our line is. This is important in ML because we can measure how close or not close we are

# root mean square error


def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(targets - predictions)))


predicted = test(x, w, b)

rmse(target, predicted)


def try_parameters(w, b):
    ages = non_smoker_df.age
    target = non_smoker_df.charges
    predictions = test(x, w, b)

    plt.plot(x, predictions, 'r', alpha=0.9)
    plt.scatter(x, target, s=8, alpha=0.8)
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.legend(['Prediction', 'Actual'])

    loss = rmse(target, predictions)
    print("RMSE Loss: ", loss)

    try_parameters(220, 50)

    # in practice, I do not need to calculate a line, and measure the amount of residual of that line compared to the line of best.
# the linear equeation function is used to plot a line, but using Ordinary Least Squares and Stochastic gradient descent will help adjust the slope and intercept
# rmse shows how far or close we are to the line of best fit
# the Ordinary Least Squares and Stochastic gradient descent are used to reduce the value of RMSE, indicating the accuracy of the line plots

    from sklearn.linear_model import SGDRegressor


# Filter the data for smokers
smoker_df = acemedical_df[acemedical_df['smoker'] == 'yes']

# Define the inputs (age) and targets (charges)
inputs = smoker_df[['age']].values
targets = smoker_df['charges'].values

# Create and train the model
sgdr = SGDRegressor()
sgdr.fit(inputs, targets)

# Make predictions
predictions = sgdr.predict(inputs)

# Plotting the data and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(smoker_df['age'], smoker_df['charges'],
            color='blue', alpha=0.5, label='Actual charges')
plt.plot(smoker_df['age'], predictions,
         color='red', label='SGD Regression Line')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.title('SGD Regression for Charges vs. Age of Smokers')
plt.legend()
plt.show()


#create input and Target (to fit in parameters)
#create and train a model
#generate preditctions
#compute loss to evaluate the model
#By automating these functions, scikit-learn's LinearRegression allows users to focus on interpreting results and applying them to their specific problems, without getting bogged down in the complexities of the underlying mathematical processes.
#from reading the sci kit learn notes, essentially the library used in scikit learn. The LinearRegression model automatically calculates the coefficients (parameters) for the linear equation that best fits the given data
 #It also computes the intercept (the value of the target when all predictors are zero), which is a crucial part of the linear equation.

inputs = non_smoker_df[['age']] #here we are making the conversion to two columns, to make the data fit correctly. column 2 is just counting
targets = non_smoker_df.charges #the model needs input and target. x and y, while weight and bias will be calculated by the scikitlearn library
lrm = LinearRegression() #inside the LinearRegression(), we need (input, and target)

lrm.fit(inputs, target) #we are putting in the age, and charges to the model
#this also codes the loss 
#this line of code does everything for you

lrm.fit(inputs, target) #we are putting in the age, and charges to the model
#this also codes the loss 
#this line of code does everything for you

b = lrm.intercept_ #this is the bias (y=mx=b)

predictions = lrm.predict(inputs) #we do this to overwrite the first prediction variable 

rmse(targets, predictions)

try_parameters(lrm.coef_, lrm.intercept_) #putting in the model into a plot 

#lets try the SGDRegressor class
sgr = SGDRegressor()
sgr.fit(inputs, targets)

predictions = sgr.predict(inputs)

sgr.coef_

sgr.intercept_

try_parameters(sgr.coef_, sgr.intercept_)

#now lets try for smokers
smoker_df = acemedicaldf[acemedicaldf.smoker == 'yes']
targets2 = smoker_df.charges
inputs2 = smoker_df[['age']]
lrm2 = LinearRegression()

lrm2.fit(inputs2, targets2)


predictions2 = lrm2.predict(inputs2)

lrm2.coef_

lrm2.intercept_

targets2 = smoker_df.charges

def try_parameters(w, b): #i just realize that this whole function is to plot out data, and also plot the model, and also spit back out the residual
    ages = smoker_df.age
    target = smoker_df.charges

    plt.plot(smoker_df.age, redictions2, 'r', alpha=0.9);
    plt.scatter(smoker_df.age, targets2, s=8,alpha=0.8);
    plt.xlabel('Age');
    plt.ylabel('Charges')
    plt.legend(['Prediction', 'Actual']);

try_parameters(lrm2.coef_, lrm2.intercept_)

 plt.scatter(smoker_df.age, smoker_df.charge)

#now lets try for smokers using the sgr
smoker_df = acemedicaldf[acemedicaldf.smoker == 'yes']
targets3 = smoker_df.charges
inputs3 = smoker_df[['age']]
x2 = smoker_df.age
sgdr = SGDRegressor()

sgdr.fit(inputs3, targets3)

sgdr.coef_

sgdr.intercept_

predictions3 = sgdr.predict(inputs3)

def try_parameters3(w, b): #i just realize that this whole function is to plot out data, and also plot the model, and also spit back out the residual
    ages = smoker_df.age
    target = smoker_df.charges
    prediction3 = test(x2, w, b)
    plt.plot(smoker_df.age, predictions3, 'r', alpha=0.9);
    plt.scatter(smoker_df.age, targets3, s=8, alpha=0.8);
    plt.xlabel('Age');
    plt.ylabel('Charges')
    plt.legend(['Prediction', 'Actual']);

try_parameters3(sgdr.coef_, sgdr.intercept_)

#machine learning is simply computing the best paremeters to model a relationship between some feature and 

#1) model, #cost function, #optimizer


#using linear regression with multiple features, in this try we will BMI into linear regression model. 
inputs, targets = non_smoker_df[['age', 'bmi']], non_smoker_df['charges'] 
#all we did here was just added 'bmi' in the input variable, as a weight in a linear relationship, everything else stays the same



model = LinearRegression().fit(inputs, targets) #you can immdieatly add the .fit to the model

predictions = model.predict(inputs)
                        

                        loss = rmse(targets, predictions) #we can print the loss right away too
print('Loss:', loss) #we can see bmi does not change the cost very much

model.coef_, model.intercept_ #the weight and bias didnt change much from just age

inputs, targets = acemedicaldf[['age', 'bmi', 'children']], acemedicaldf['charges'] #this is a model from the data with both smoker and nonesmoker



    models2 = LinearRegression().fit(inputs, targets)

predictions2 = models2.predict(inputs)

loss = rmse(targets, predictions2) 
print('Loss:', loss)

#in some cases, we want to catorgize data, because with linear regression, we needed to use numbers. but with category we have none number valuyes
#like with smoker/none smoker
# 1) convert into binary (1,0) for 2 cats
# 2) 'one hot', having multple columns, and if that data row has a 1 in one of the selected column than it is that value
# 3) if you need to preseve the order of multple cats, (1,2,3,4) can still show in a linear relationship

#here is the binary example with smoker and none smoker
sns.barplot(data=acemedicaldf, x= 'smoker', y='charges')

#making smoker data into binary
smoker_codes = {'no': 0, 'yes': 1}
acemedicaldf['smoker_code'] = acemedicaldf.smoker.map(smoker_codes)

acemedicaldf #what happen here is we added a new column to the data, and to fill that column, we say, if the value in smoker is yes or no, fill the smoker code column with 0 or 1 based on the smoker colum

acemedicaldf.charges.corr((acemedicaldf.smoker_code)) #if we look at the graph on top, we can see that there is a clear showing that smokers inluences the charges
#this line of code confirms that smokers are closely correlated to charges

inputs, targets = acemedicaldf[['age', 'bmi', 'children', 'smoker_code']], acemedicaldf['charges']

model = LinearRegression().fit(inputs, targets)

predictions1 = model.predict(inputs)

loss = rmse(targets, predictions1) 
print('Loss:', loss)
#this data here shows thats smokers as a category is a significant influence on charges

#now lets try with sex
sex_code = {'male': 0, 'female': 1}
acemedicaldf['sex_code'] = acemedicaldf.sex.map(sex_code)
acemedicaldf

inputs, targets = acemedicaldf[['age', 'bmi', 'children','smoker_code', 'sex_code']], acemedicaldf['charges']

model11 = LinearRegression().fit(inputs, targets)

prediction11 = model11.predict(inputs)
loss = rmse(targets, prediction11) 
print('Loss:', loss)

#one hot encoding
enc = preprocessing.OneHotEncoder() #this new 'onehotencoder' takes the unique values from a column in the data
enc.fit(acemedicaldf[['region']]) #here we are telling it to take the values from region
enc.categories_ #here is the catgories after it has been processed
one_hot = enc.transform(acemedicaldf[['region']]).toarray() #here are are vectorizing the column, and give its relationship a '1' and zeros for the rest amoung the new columns
one_hot
acemedicaldf[['northeast', 'northwest', 'southeast', 'southwest']] = one_hot

acemedicaldf #now we can see the new columns. Where a record was previously just ' northeast' is now northeast = 1,
# 'northwest', 'southeast', 'southwest' all = 0
#lets run this through the linear model again but with the new data from regions
inputsq, targetsq = acemedicaldf[['age', 'bmi', 'children','smoker_code', 'sex_code', 'northeast', 'northwest', 'southeast', 'southwest']], acemedicaldf['charges']
modelq = LinearRegression().fit(inputsq, targetsq)
predictionsq = modelq.predict(inputsq)
loss = rmse(targetsq, predictionsq)
print('Loss:', loss)
modelq.coef_ #Remeber that using the coeffienct is the w or weight the equation in the linear regression (slope) and by using this .ceof, if we have multiple
#features in the inputs (columns) we can actually get the weights for each. And in this case, we see 23848, which is the largest value. meaning the column "smoker" influences the charge the more
modelq.intercept_ 
#so since the ranges in the columns are so varies, like in charges going up super high, while region is just 0 or 1, to get a better reading. we should standardize them
numeric_cols = ['age', 'bmi', 'children'] #we dont need to standardize smoker, or region because they are already zero to 1
scaler = StandardScaler() #this computes the means and variance of the columns in the data frame
scaler.fit(acemedicaldf[numeric_cols])
scaler.mean_, scaler.var_
scaled_inputs = scaler.transform(acemedicaldf[numeric_cols]) #now we are scaling the columns
scaled_inputs
cat_cols = ['smoker_code', 'sex_code', 'northeast', 'northwest', 'southeast', 'southwest']
categorical_data = acemedicaldf[cat_cols].values
inputs = np.concatenate((scaled_inputs, categorical_data), axis=1)
targets = acemedicaldf.charges
modelqq = LinearRegression().fit(inputs, targets)
predictions = modelqq.predict(inputs)
loss = rmse(targets, predictions)
print('Loss:', loss)
modelqq.coef_
#a thing to note, if we have new data coming in, we have to remember to scale the data 
# Explore the data and find correlations between inputs and targets
# Pick the right model, loss functions and optimizer for the problem at hand, Scikitlearn handles all of this
# Scale numeric variables and one-hot encode categorical data. Make sure to scale the data to have a mean of 0 and a standard deviation of 1
# Set aside a test set (using a fraction of the training set)
# Train the model
# Make predictions on the test set and compute the loss