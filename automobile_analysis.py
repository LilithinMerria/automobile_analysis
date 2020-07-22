import numpy as np
import pandas as pd 
import matplotlib as plt
from matplotlib import pyplot 

auto_path = "D:\DataAnalytics\IBM\imports-85.data"
auto = pd.read_csv(auto_path, header=None)
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
auto.columns = headers
print(auto.dtypes)
print(auto.describe(include="all"))
print(auto.info)

# Drop missing value(s) on the price column and replace "?" by "nan"
auto.dropna(subset=["price"], axis=0)
auto.replace("?", np.nan, inplace=True)
print(auto.head(7))


### Data Cleaning
# Check the missing values
missing_data = auto.isnull()
print(missing_data)

# Count missing value(s) in each column
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")

# Take the mean of "normalized-losses", "stroke", "bore", "horsepower" and "peak-rpm"
avg_norm = auto["normalized-losses"].astype("float").mean(axis=0)
avg_stroke = auto["stroke"].astype("float").mean(axis=0)
avg_bore = auto["bore"].astype("float").mean(axis=0)
avg_peak = auto["peak-rpm"].astype("float").mean(axis=0)
avg_hpower = auto["horsepower"].astype("float").mean(axis=0)

# Replace "normalized-losses", "stroke", "bore", "horsepower" and "peak-rpm"'s missing value(s) by their mean
auto["normalized-losses"].replace(np.nan, avg_norm, inplace=True)
auto["stroke"].replace(np.nan, avg_stroke, inplace=True)
auto["bore"].replace(np.nan, avg_bore, inplace=True)
auto["peak-rpm"].replace(np.nan, avg_peak, inplace=True)
auto["horsepower"].replace(np.nan, avg_hpower, inplace=True)

print(auto["num-of-doors"].value_counts().idxmax())

#Replace the missing value(s) of "num-of-doors" with "four"
auto["num-of-doors"].replace(np.nan, "four", inplace=True)

# Drop nan from price then reset index
auto.dropna(subset=["price"], axis=0, inplace=True)
auto.reset_index(drop=True, inplace=True)
print(auto.head())

# Convert data types to correct format
auto[["stroke", "bore", "peak-rpm"]] = auto[["stroke", "bore", "peak-rpm"]].astype("float")
auto[["normalized-losses", "price"]] = auto[["normalized-losses", "price"]].astype("int64")
#print(auto.dtypes)

# Data normalization
auto["length"] = auto["length"] / auto["length"].max()
auto["width"] = auto["width"] / auto["width"].max()
auto["height"] = auto["height"] / auto["height"].max()
#print(auto[["length", "width", "height"]].head())
auto["city-L/100km"] = 233/auto["city-mpg"]
auto["highway-L/100km"] = 233/auto["highway-mpg"]

# Binning of Horsepower with low, medium and high price
auto["horsepower"] = auto["horsepower"].astype(int, copy=True)

# Plot the histogram of horsepower
plt.pyplot.hist(auto["horsepower"])

# setting x/y axis and title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
plt.pyplot.show()

# Creating bins using linspace
bins = np.linspace(min(auto["horsepower"]), max(auto["horsepower"]), 4)
group_names = ["low", "Medium", "High"]
auto["horsepower-binned"] = pd.cut(auto["horsepower"], bins, labels=group_names, include_lowest=True)
#print(auto[["horsepower", "horsepower-binned"]].head(20))
#print(auto["horsepower-binned"].value_counts())

# Plot distribution of each bin
a = (0, 1, 2)

plt.pyplot.hist(auto["horsepower"], bins=3)

plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
plt.pyplot.show()

# Dumming variable
dummy_var1 = pd.get_dummies(auto["fuel-type"])
dummy_var2 = pd.get_dummies(auto["aspiration"])
dummy_var1.rename(columns={"fuel-type-diesel": "gas", "fuel-type-diesel": "diesel"}, inplace=True)
dummy_var2.rename(columns={"aspiration-std": "std", "aspiration-turbo": "turbo"}, inplace=True)
#print(dummy_var1, dummy_var2)

# Merge auto and dummies
auto = pd.concat([auto, dummy_var1, dummy_var2], axis=1)
auto.drop(auto[["fuel-type", "aspiration"]], axis=1, inplace=True)
print(auto.head())

# Save dataset
auto.to_csv("D:\DataAnalytics\IBM\clean_auto.csv")


