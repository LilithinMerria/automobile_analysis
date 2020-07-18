import pandas as pd 

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

# Drop missing value(s) on the price column
auto.dropna(subset=["price"], axis=0)
print(auto.head(7))
