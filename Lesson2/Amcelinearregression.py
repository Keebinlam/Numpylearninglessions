# because we are taking information from a url, using urlretrieval will allow us to pull a url
# pandas to read the csv
# numpy to calucuate data
from urllib.request import urlretrieve
import pandas as pd
import numpy as np
# to save data
import jovian
# to visualize the data
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


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

#do not need to use a Linear regression Single Feature, or Loss/Cost Function, just use scikit learn
#in geometri, y = mx +b. (m being the slope, and b being the interceptor: or the point that the slope starts at Y =0.
#but in ML, m = w which means weigh, and b is bias. This is just a linear equation
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

plt.plot(x, estimated_charges, 'r-o');
plt.xlabel('Age');
plt.ylabel('Estimated Charges');

target = non_smoker_df.charges

plt.plot(x, estimated_charges, 'r', alpha=0.9);
plt.scatter(x, target, s=8,alpha=0.8);
plt.xlabel('Age');
plt.ylabel('Charges')
plt.legend(['Estimate', 'Actual']);

x = non_smoker_df.age
target = non_smoker_df.charges
estimated_charges = test(w, x, b)

#to summarize what just happen, I created a graph that plots the acme data, as well as a linear function (not actual linear line from correlation coeffiecnt
#then created a function showing that takes in x and b paremeters, into a linear equation. The purpose of this is to plug in different numbers, and see how the line moves
#the goal is to get the link to match as close to the general sloped line of the acme data
#now we need to know how to measure how accurate our line is. This is important in ML because we can measure how close or not close we are 

#root mean square error 
def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(targets - predictions)))

predicted = test(x, w, b)

rmse(target, predicted)

def try_parameters(w, b):
    ages = non_smoker_df.age
    target = non_smoker_df.charges
    predictions = test(x, w, b)
    
    plt.plot(x, predictions, 'r', alpha=0.9);
    plt.scatter(x, target, s=8,alpha=0.8);
    plt.xlabel('Age');
    plt.ylabel('Charges')
    plt.legend(['Prediction', 'Actual']);
    
    loss = rmse(target, predictions)
    print("RMSE Loss: ", loss)

    try_parameters(220, 50)

    #in practice, I do not need to calculate a line, and measure the amount of residual of that line compared to the line of best. 
#the linear equeation function is used to plot a line, but using Ordinary Least Squares and Stochastic gradient descent will help adjust the slope and intercept
#rmse shows how far or close we are to the line of best fit
#the Ordinary Least Squares and Stochastic gradient descent are used to reduce the value of RMSE, indicating the accuracy of the line plots