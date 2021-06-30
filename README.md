# Notes
We recommend creating a folder entitled 'Time Series' and saving all the python file in it. This will allow you including files paths that will point to modules we created and that you need to import

# Dataset
We analyzed the problem of day-ahead electricity spot price forecasting in Norway. 

Our dataset includes the consecutive recordings of 2600 days from January 2nd, 2013 and February 14th, 2020 (source NORDPOOL https://www.nordpoolgroup.com/). After the preprocessing phase, the dataset was divided into a training set (the first 1600 days), validation set (the next 400 days), and test set (the last 600 days).

We used seven input variables selected (explanatory variables):

1. Electricity consumption prognosis, MWh
2. Electricity production prognosis, MWh
3. Wind prognosis, MWh
4. One quarter forward contract price, EUR
5. One year forward contract price, EUR
6. Brent oil price prognosis, EUR
7. Coal price, EUR

# Missing Values
In this study, we deployed five different techniques to deal with missing values and compared their results before finally selecting the most optimal one for our dataset:

1. Ignore: we simply ignore the missing values by deleting them.
2. Mean: the missing values are replaced with the mean of the weekly electricity prices. 
3. Cubic spline interpolation: this mathematical technique constructs new data points within the boundaries of a set of other known points. 
4. Nearest value: the missing values are replaced with those of the nearest day. 
5. K-nearest neighbor (KNN): in this study, we set k as 7 and replaced the missing values with the average of their seven nearest measured data points.  

KNN performs best on our dataset. Therefore, we used it to fill the values of our missing data points in the final experimental design. 

# Correlation Coefficient
To solidify our choice of input variables for the dataset, the correlation between its different features (target and explanatory) was explored by computing the Pearson, Spearman, and Kendall correlation matrices.

# Models Building
To ensure that our results were robust and not specific to any one neural network architecture, we developed five models to test our loss function: FFNN, convolutional neural network (CNN), recursive neural network (RNN), long-short term memory neural network (LSTM), and gated recurrent unit (GRU) neural network.
Our design follows the principles of simplicity, where we keep each model at 1 hidden layer with 64 corresponding neurons. As the activation function, we primarily used ReLU. We also used the RMSprop as the optimization algorithm for the models’ stochastic gradient descent.
