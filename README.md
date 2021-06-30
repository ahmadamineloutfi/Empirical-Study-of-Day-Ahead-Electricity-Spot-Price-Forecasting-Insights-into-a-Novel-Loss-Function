# Notes
We recommend creating a folder entitled 'Time Series' and saving all the python file in it. This will allow you including files paths that will point to modules we created and that you need to import

# Dataset
We analyzed the problem of day-ahead electricity spot price forecasting in Norway. 

Our dataset includes the consecutive recordings of 2600 days from January 2nd, 2013 and February 14th, 2020 (source NORDPOOL https://www.nordpoolgroup.com/). After the preprocessing phase, the dataset was divided into a training set (the first 1600 days), validation set (the next 400 days), and test set (the last 600 days).

We used seven input variables selected (explanatory variables):
- Electricity consumption prognosis, MWh
- Electricity production prognosis, MWh
- Wind prognosis, MWh
- One quarter forward contract price, EUR
- One year forward contract price, EUR
- Brent oil price prognosis, EUR
- Coal price, EUR

The main dataset set used is entitled: 
- ' #E_data.xlsx'. 

In addition you will need another dataset that will be used solely to reproduce the plots of the loss functions (MAE,MSE, Theil UI, Theil UII, Theil UII square) where the true target value is 100, and the predicted values range between -10,000 to 10,000. It is entitled:
- 'plot of the loss functions.xlsx'

Graphs:
To reproduce the graphs 1, 2, 3, 4, 5 and 6 as well as table 2, you  need the following python file:
1. 'plots for the loss functions.py'

- Reproducing graphs 7, 8 and 9, is part of the correlation coefficients python file (below).
- Repoducing table 3, is part of the missing value main python file (below).
- Reproducing tables 4, 5 and 6 as well as graphs 10, 11 and 12, is part of the model building main python file (below).

# Missing Values
In this study, we deployed five different techniques to deal with missing values and compared their results before finally selecting the most optimal one for our dataset:

- Ignore: we simply ignore the missing values by deleting them.
- Mean: the missing values are replaced with the mean of the weekly electricity prices. 
- Cubic spline interpolation: this mathematical technique constructs new data points within the boundaries of a set of other known points. 
- Nearest value: the missing values are replaced with those of the nearest day. 
- K-nearest neighbor (KNN): in this study, we set k as 7 and replaced the missing values with the average of their seven nearest measured data points.  

KNN performs best on our dataset. Therefore, we used it to fill the values of our missing data points in the final experimental design. 

To reproduce the results you need the following python files:
1. 'main_missing_values.py': to import the main file.
2. 'ignore.py': to import 'Ignore' technique module.
3. 'mean.py': to import 'Mean' technique module.
4. 'Cubic_spline.py': to import 'cubic spiline' technique module.
5. 'nearest_value.py': to import 'nearest value' technique module.
6. 'KNN.PY': to import 'KNN' technique module.
7. 'model.py': to import the model (FFNN) module.
8. 'loss.py': to import the loss function (MSE) module.
9. 'metric.py': to import the evaluation metric (MSE) module.

# Correlation Coefficient
To solidify our choice of input variables for the dataset, the correlation between its different features (target and explanatory) was explored by computing the Pearson, Spearman, and Kendall correlation matrices.

To reproduce the results you need the following python files:
1. 'main_correlation_coefficients.py': to import the main file.
2. 'KNN.PY': to import 'KNN' technique module.

# Models Building
To ensure that our results were robust and not specific to any one neural network architecture, we developed five models to test our loss function: FFNN, convolutional neural network (CNN), recursive neural network (RNN), long-short term memory neural network (LSTM), and gated recurrent unit (GRU) neural network.
Our design follows the principles of simplicity, where we keep each model at 1 hidden layer with 64 corresponding neurons. As the activation function, we primarily used ReLU. We also used the RMSprop as the optimization algorithm for the models’ stochastic gradient descent.

To reproduce the results you need the following python files:
1. 'main csv.py': to import the main file.
2. 'KNN.PY': to import 'KNN' technique module.
3. 'loss.py': to import the loss functions (MAE, MSE, Theill UII, Theill UII square) module.
4. 'metric.py': to import the evaluation metrics (MAE & MSE) module.
5. 'model.py': to import the models (FFNN,CNN,LSTM,RNN,GRU) module.
