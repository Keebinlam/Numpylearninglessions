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
