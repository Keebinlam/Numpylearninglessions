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
