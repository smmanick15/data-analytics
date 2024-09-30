import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


print("test print statement")

dataset = pd.read_csv("UberDataset.csv")
dataset.head()
dataset.shape

print(dataset.head())
print("\nNow printing the shape of the data set:")
print(dataset.shape)
print(dataset.info())

# Pandas fillna() method replaces NULL values with specified value
# In this case, we replace NULL with "NOT"
dataset['PURPOSE'].fillna("NOT", inplace=True)

print("data set before date_time format")
print(dataset.head())

# Change START_DATE and END_DATE to date_time format
dataset['START_DATE'] = pd.to_datetime(dataset['START_DATE'], 
    errors = 'coerce')

dataset['END_DATE'] = pd.to_datetime(dataset['END_DATE'],
    errors = 'coerce')

print("data set after date_time formatting")
print(dataset.head())



# Split START_DATE to date and time column and convert time into 
# 4 different categories: morning, afternoon, evening, night
# dataset['date'] creates a new column called date
# dataset['time'] creates a new column called time

dataset['date'] = pd.DatetimeIndex(dataset['START_DATE']).date
dataset['time'] = pd.DatetimeIndex(dataset['START_DATE']).hour

print("did we make it this far?")
print(dataset.head())


# change categories into category of day and night
dataset['day-night'] = pd.cut(x=dataset['time'],
                              bins = [0, 10, 15, 19, 24],
                              labels =
                              ['Morning', 'Afternoon', 'Evening', 'Night'])


# after creating new columns, drop rows with NULL values
# dataset.dropna(inplace=True)


# drop duplicate rows from dataset
# dataset.drop_duplicates(inplace=True)




# DATA VISUALIZATION
# check unique values in dataset of columns with object datatype
obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)

unique_values = {}
for col in object_cols:
    unique_values[col] = dataset[col].unique().size
unique_values

print("printing unique values from the object columns")
print(unique_values)
print("this means that for:")
for col in unique_values:
    print("In the column: ", col, ", there are ", 
          unique_values[col], "unique values")

print()
print()


# Using matplotlib and seaborn library for countplot for
# CATEGORY and PURPOSE columns

# figure() creates a new figure
# parameters figsize() accepts (float, float) meaning width and height in inches
fig = plt.figure(figsize=(10,5))

# subplot() creates a subplot in a grid of subplots
# subplot - the first 1 is the # of rows in the grid
# subplot - the second 2 is the # of columns in the grid
# subplot - the third 1 is the index of the current subplot
# countplot() with seaborn library creates a histogram (represents categorical frequency)

"""
plt.subplot(2,1,1)
sns.countplot(dataset['CATEGORY'])
plt.xticks(rotation=90)

plt.subplot(2,1,2)
sns.countplot(dataset['PURPOSE'])
plt.xticks(rotation=90)
"""


# Create the first subplot
plt.subplot(2, 1, 1)
sns.countplot(x='CATEGORY', data=dataset)  # It's better to use x= to specify the column

# Rotate the x-axis labels
plt.xticks(rotation=90)
plt.tight_layout()  # Adjusts subplot parameters to give more room for labels


# Create the second subplot
plt.subplot(2,1,2)
sns.countplot(x = 'PURPOSE', data=dataset)
plt.xticks(rotation=90)
plt.tight_layout()

plt.show()