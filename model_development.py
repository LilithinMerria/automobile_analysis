import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

lm = LinearRegression()

path = "D:\DataAnalytics\IBM\clean_auto.csv"
auto = pd.read_csv(path)
print(auto.head())

# Predit the price using highway-mpg
X = auto[["highway-mpg"]]
Y = auto["price"]

# Fit X and Y
lm.fit(X, Y)

# Print the price prediction, intercept and coefficient 
Yhat = lm.predict(X)
print("The Yhat for highway is = ", Yhat[0:5])
print("The intercept a is = ", lm.intercept_)
print("The coefficient b is = ", lm.coef_)

# Predict the price using engine-size
X = auto[["engine-size"]]
Y = auto["price"]
lm.fit(X, Y)
Yhat = lm.predict(X)
print("The Yhat for engine-size is = ", Yhat[0:5])
print("The intercept of engine-size is = ", lm.intercept_)
print("the slope of engine-size is = ", lm.coef_)

### Multiple Linear Regression
# Using Horsepower, Curb-weight, Engine-size and Highway-mpg 
# since they're good predictors based on our exploratory_analysis

Z = auto[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]
lm.fit(Z, auto["price"])
Yhat = lm.predict(Z)
print("The Yhat for this multiple LR is = ", Yhat[0:5])
print("The intercept of this multiple LR is = ", lm.intercept_)
print("The slope of this multiple LR is = ", lm.coef_)

### Prediction using visualization
# Plot highway-mpg and price
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=auto)
plt.ylim(0,)
plt.show()

# Plot residual highway-mpg and price
plt.figure(figsize=(width, height))
sns.residplot(x="highway-mpg", y="price", data=auto)
plt.show()

## Multiple Linear Regression

plt.figure(figsize=(width, height))
ax1 = sns.distplot(auto["price"], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values", ax=ax1)

plt.title("Actual vs Fitted Values for Price")
plt.xlabel("Price (in dollars)")
plt.ylabel("Cars' proportions")

plt.show()

### Polynomial Regression
# On highway-mpg based on the residual result
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()

# Getting the variables
x = auto["highway-mpg"]
y = auto["price"]

# Using 3rd polynomial
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print("The third polynomial is : ", p)

# Plotting the function
PlotPolly(p, x, y, "highway-mpg")
np.polyfit(x, y, 3)

# 11th order polynomial of highway-mpg
f1 = np.polyfit(x, y, 11)
p1 = np.poly1d(f1)
PlotPolly(p1, x, y, "highway-mpg")
np.polyfit(x, y, 11)

### Polynomial transform
pr = PolynomialFeatures(degree=2)
Z_pr = pr.fit_transform(Z)
print(Z.shape)
print(Z_pr.shape)

### Pipeline
# Create pipeline using a list of tuples including name of the model
# and its constructor
Input=[("scale",StandardScaler()), ("polynomial", PolynomialFeatures(include_bias=False)), ("model", LinearRegression())]

# Input the list as argument to the pipeline constructor
pipe = Pipeline(Input)
pipe.fit(Z, y)

# Normalize, perform transform and predict 
ypipe = pipe.predict(Z)
print(ypipe[0:4])

# In-sample evaluation
# Simple Linear Regression for highway-mpg
lm.fit(X,Y)
# find R^2
print("The R^2 is = ", lm.score(X,Y))

# Predict the output Yhat using X
Yhat =lm.predict(X)
print("The output of the first four predicted value is: ", Yhat[0:4])

# Compare the Actual results with the Predicted results
mse = mean_squared_error(auto["price"], Yhat)
print("The mean square error of the price and predicted value is : ", mse)

## Multiple Linear Regression
lm.fit(Z, auto["price"])
print("The R-square is = ", lm.score(Z, auto["price"]))

# Predict
Y_predict_multifit = lm.predict(Z)
print("The mean square error of price and predicted value using multifit is: ", 
    mean_squared_error(auto["price"], Y_predict_multifit))

# Polynomial fit
r_squared = r2_score(y, p(x))
print("The R_squared value is : ", r_squared)
print("The mse is : ", mean_squared_error(auto["price"], p(x)))

### Prediction and Decision making 
new_input = np.arange(1, 100, 1).reshape(-1, 1)
lm.fit(X, Y)
yhat = lm.predict(new_input)
print("The prediction output value is: ", yhat[0:5])

# Plot the data
plt.plot(new_input, yhat)
plt.show()

# Comparing these three models, we conclude that the MLR model is the best model to be able to predict price from our dataset.










