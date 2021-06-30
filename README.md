# Notes
We recommend creating a folder entitled 'Time Series' and saving all the python file in it. This will allow you including files paths that will point to modules we created and that you need to import

# Dataset


# Missing Values
In this study, we deployed five different techniques to deal with missing values and compared their results before finally selecting the most optimal one for our dataset:

1. Ignore: we simply ignore the missing values by deleting them.
2. Mean: the missing values are replaced with the mean of the weekly electricity prices. 
3. Cubic spline interpolation: this mathematical technique constructs new data points within the boundaries of a set of other known points. 
4. Nearest value: the missing values are replaced with those of the nearest day. 
5. K-nearest neighbor (KNN): in this study, we set k as 7 and replaced the missing values with the average of their seven nearest measured data points.  

KNN performs best on our dataset. Therefore, we used it to fill the values of our missing data points in the final experimental design. 

# Correlation Coefficient

# Models Building


# Graphs and Tables
