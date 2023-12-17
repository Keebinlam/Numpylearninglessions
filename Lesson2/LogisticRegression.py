import opendatasets as od
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
%matplotlib inline 