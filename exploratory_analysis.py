import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns  
from scipy import stats

# Finding the main characteristics which have the most impact on the car price".
# Import the clean_auto.csv file
path = "D:\DataAnalytics\IBM\clean_auto.csv"
auto1 = pd.read_csv(path)
print(auto1.head())
print(auto1.dtypes)

# Calculate and print the correlation of float64 and int64 types 
#print(auto1.corr())

# Finding the correlation between bore, stroke, compression-ratio and horsepower
print(auto1[["bore", "stroke", "compression-ratio", "horsepower"]].corr())

# Linear relationship between engine size and price using regplot
print(auto1[["engine-size", "price"]].corr())
sns.regplot(x = "engine-size", y = "price", data=auto1)
plt.ylim(0,)
plt.show() # the result is a positive correlation between engine-size and price and seems like a good predictor

# Linear relationship between highway-mpg and price using regplot
print(auto1[["highway-mpg", "price"]].corr())
sns.regplot(x="highway-mpg", y="price", data=auto1)
plt.ylim(0,)
plt.show() # the result is a negative correlation and seems like a good predator

# Linear relationship between peak-rpm and price
print(auto1[["peak-rpm", "price"]].corr())
sns.regplot(x="peak-rpm", y="price", data=auto1)
plt.ylim(0,)
plt.show() # peak-rpm doesn't look like a good predictor

# Linear relationship between stroke and price
print(auto1[["stroke", "price"]].corr())
sns.regplot(x="stroke", y="price", data=auto1)
plt.ylim(0,)
plt.show() # stroke doesn't look like a good predictor either

### Linear relationship of categorical variables
# Linear relationship between body-style and price
sns.boxplot(x="body-style", y="price", data=auto1)
plt.show() # Not a good predictor

# Linear relationship between engine-location and price
sns.boxplot(x="engine-location", y="price", data=auto1)
plt.show() # Potentially a good predictor

# Linear relationship between drive-wheels and price
sns.boxplot(x="drive-wheels", y="price", data=auto1)
plt.show() # kPotentially a good predictor

### Descriptive Statistics Analysis
print(auto1.describe(include=["object"]))

# Value counts on drive-wheels then convert it to frame
# Saving the result in the dataframe drive_wheels_counts then rename the column drive_wheels to value_counts
drive_wheels_counts = auto1["drive-wheels"].value_counts().to_frame()
drive_wheels_counts.rename(columns={"drive-wheels":"value_counts"}, inplace=True)
#print(drive_wheels_counts)

# Rename the index to drive-wheels
drive_wheels_counts.index.name = "drive-wheels"
print(drive_wheels_counts)

# Value counts of engine-location then replace the index to engine-location
engine_location_counts = auto1["engine-location"].value_counts().to_frame()
engine_location_counts.rename(columns={"engine-location": "value_counts"}, inplace=True)
engine_location_counts.index.name = "engine-location"
print(engine_location_counts) # not a good predictor since the result is skewed

### Grouping 
# Grouping drive-wheels, body-style and price and find the average price of drive-wheels
# different categories 
group_drive = auto1[["drive-wheels", "body-style", "price"]]
group_drive = group_drive.groupby(["drive-wheels"], as_index=False).mean()
print(group_drive)

# Finding the average price of body-style
group_style = auto1[["drive-wheels", "body-style", "price"]]
group_style = group_style.groupby(["body-style"], as_index=False).mean()
print(group_style)

# Finding the average price of drive-wheels and body-style the display
#u using pivot for better visualization
group2 = auto1[["drive-wheels", "body-style", "price"]]
group_drive_style = group2.groupby(["drive-wheels", "body-style"], as_index=False).mean()
#print(group_drive_style)
group_drive_style_pivot = group_drive_style.pivot(index="drive-wheels", columns="body-style")
#print(group_drive_style_pivot)

# Filling the nan with 0
group_drive_style_pivot = group_drive_style_pivot.fillna(0)
print(group_drive_style_pivot)

# Visualizing the relationship between body-style and price 
#using heatmap
fig, ax = plt.subplots()
im = ax.pcolor(group_drive_style_pivot, cmap="RdBu")

# Labels
row_labels = group_drive_style_pivot.columns.levels[1]
col_labels = group_drive_style_pivot.index

# Move ticks and labels to the center
ax.set_xticks(np.arange(group_drive_style_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(group_drive_style_pivot.shape[0]) + 0.5, minor=False)

# Insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()

### Pearson Correlation and P-value
# Pearson correlation and P-value of wheel-base and price
pearson_coef, p_value = stats.pearsonr(auto1["wheel-base"], auto1["price"])
print("The pearson correlation coefficient is ", pearson_coef, "with a P-value of P =", p_value)

# Pearson correlation and P-value of horsepower and price
pearson_coef, p_value = stats.pearsonr(auto1["horsepower"], auto1["price"])
print("The horsepower's pearson correlation coefficient is ", pearson_coef, "with a P-value of P =", p_value)

# Pearson correlation and P-value of length and price
pearson_coef, p_value = stats.pearsonr(auto1["length"], auto1["price"])
print("The length's pearson correlation coefficient is ", pearson_coef, "with a P-value of P =", p_value)

# Pearson correlation and P-value of width and price
pearson_coef, p_value = stats.pearsonr(auto1["width"], auto1["price"])
print("The width's pearson correlation coefficient is ", pearson_coef, "with a P-value of P =", p_value)

# Pearson correlation and P-value of curb-weight and price
pearson_coef, p_value = stats.pearsonr(auto1["curb-weight"], auto1["price"])
print("The curb-weight's pearson correlation coefficient is ", pearson_coef, "with a P-value of P =", p_value)

# Pearson correlation and P-value of engine-size and price
pearson_coef, p_value = stats.pearsonr(auto1["engine-size"], auto1["price"])
print("The engine-size's pearson correlation coefficient is ", pearson_coef, "with a P-value of P =", p_value)

# Pearson correlation and P-value of bore and price
pearson_coef, p_value = stats.pearsonr(auto1["bore"], auto1["price"])
print("The bore's pearson correlation coefficient is ", pearson_coef, "with a P-value of P =", p_value)

# Pearson correlation and P-value of city-mpg and price
pearson_coef, p_value = stats.pearsonr(auto1["city-mpg"], auto1["price"])
print("The city-mpg's pearson correlation coefficient is ", pearson_coef, "with a P-value of P =", p_value)

# Pearson correlation and P-value of highway-mpg and price
pearson_coef, p_value = stats.pearsonr(auto1["highway-mpg"], auto1["price"])
print("The highway-mpg's pearson correlation coefficient is ", pearson_coef, "with a P-value of P =", p_value)

### ANOVA
group3 = group2[["drive-wheels", "price"]].groupby(["drive-wheels"])
#print(group3.head(3))

# Finding the f_val and p_val of all drive-wheels categories
# using f_oneway method
f_val, p_val = stats.f_oneway(group3.get_group("fwd")["price"], group3.get_group("rwd")["price"], group3.get_group("4wd")["price"])
print("fwd, rwd and 4wd's Anova results: F-value = ", f_val, "P-value = ", p_val)

# Finding the f_val and p_val of fwd and rwd drive-wheels categories
f_val, p_val = stats.f_oneway(group3.get_group("fwd")["price"], group3.get_group("rwd")["price"])
print("Anova results: F-value = ", f_val, "P-value = ", p_val)

# Finding the f_val and p_val of 4wd and rwd drive-wheels categories
f_val, p_val = stats.f_oneway(group3.get_group("rwd")["price"], group3.get_group("4wd")["price"])
print("4wd and rwd's Anova results: F-value = ", f_val, "P-value = ", p_val)

# Finding the f_val and p_val of 4wd and fwd drive-wheels categories
f_val, p_val = stats.f_oneway(group3.get_group("fwd")["price"], group3.get_group("4wd")["price"])
print("4wd and fwd's Anova results: F-value = ", f_val, "P-value = ", p_val)

### Important variables
# Continuous numerical variables: Length, width, curb-weight, engine-size
# horsepower, city-mpg, highway-mpg, wheel-base and bore

# Categorical variables: drive-wheels





























