import numpy as np
import pandas as pd
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns


# Load Arable.xls into a Pandas DataFrame
arable_df = pd.read_excel(r'D:\Hertfordshire\Applied data science\Moe 2nd\Arable.xlsx')

# Load Forest.xls into a Pandas DataFrame
forest_df = pd.read_excel(r'D:\Hertfordshire\Applied data science\Moe 2nd\Forest.xlsx')

# Now you can use arable_df and forest_df to perform further data analysis
# using Pandas and NumPy libraries


# Print column names of Arable.xls file
print("Columns in Arable.xls:")
print(arable_df.columns)

# Print column names of Forest.xls file
print("\nColumns in Forest.xls:")
print(forest_df.columns)

# Transpose the DataFrames to create one with years as columns and one with countries as columns
arable_by_year = arable_df.set_index('Country Name').T
forest_by_year = forest_df.set_index('Country Name').T

# Remove any NaN or Null values from the DataFrames
arable_by_year.dropna(inplace=True)
forest_by_year.dropna(inplace=True)


# Use .describe() method to explore the data in arable_by_year DataFrame
print("Arable land area in hectares (ha) per country over years:")
print(arable_by_year.describe())

# Calculate the mean of each country's arable land area for each year
arable_by_year_mean = arable_by_year.mean(axis=0)
print("\nMean arable land area in hectares (ha) per year for all countries:")
print(arable_by_year_mean)

# Calculate the correlation between arable land area and forest area for each year
correlation = arable_by_year.corrwith(forest_by_year, axis=0)
print("\nCorrelation between arable land area and forest area per year:")
print(correlation)


def country_stats(df1, df2, country):
    # Check if the selected country is in both dataframes
    if country not in df1.index or country not in df2.index:
        print(f"{country} not found in both dataframes")
        return None
    
    # Get the data for the selected country and drop missing values
    arable_data = df1.loc[country].dropna()
    forest_data = df2.loc[country].dropna()
    
    # Calculate the mean, median, and skewness for each dataframe
    arable_mean = arable_data.mean()
    arable_median = arable_data.median()
    arable_skew = arable_data.skew()
    
    forest_mean = forest_data.mean()
    forest_median = forest_data.median()
    forest_skew = forest_data.skew()
    
    # Create a results dataframe
    results_df = pd.DataFrame({
        'Arable Land': [arable_mean, arable_median, arable_skew],
        'Forest Area': [forest_mean, forest_median, forest_skew]},
        index=['Mean', 'Median', 'Skewness']
    )
    
    return results_df



# Call the function to calculate the statistics for a different country
stats_df = country_stats(arable_by_year, forest_by_year, 'Aruba')

# Print the results
print(stats_df)




# Manually enter the data for forest and arable land analysis for 7 countries
data = {
    'country': ['USA', 'China', 'India', 'Brazil', 'Russia', 'Australia', 'Canada'],
    'arable_land': [10, 20, 30, 40, 50, 60, 70],
    'forest_land': [5, 15, 25, 35, 45, 55, 65]
}

# Create dataframes from the data
arable_df = pd.DataFrame(data, columns=['country', 'arable_land'])
forest_df = pd.DataFrame(data, columns=['country', 'forest_land'])

# Calculate the mean, median, and skewness for each country
results = []
for i in range(len(data['country'])):
    country = data['country'][i]
    arable_mean = arable_df.loc[arable_df['country'] == country, 'arable_land'].mean()
    arable_median = arable_df.loc[arable_df['country'] == country, 'arable_land'].median()
    arable_skewness = skew(arable_df.loc[arable_df['country'] == country, 'arable_land'])
    forest_mean = forest_df.loc[forest_df['country'] == country, 'forest_land'].mean()
    forest_median = forest_df.loc[forest_df['country'] == country, 'forest_land'].median()
    forest_skewness = skew(forest_df.loc[forest_df['country'] == country, 'forest_land'])
    results.append((country, arable_mean, arable_median, arable_skewness, forest_mean, forest_median, forest_skewness))

# Create a dataframe from the results
results_df = pd.DataFrame(results, columns=['country', 'arable_mean', 'arable_median', 'arable_skewness', 'forest_mean', 'forest_median', 'forest_skewness'])




# Histogram of arable land and forest land
plt.hist([arable_df['arable_land'], forest_df['forest_land']], bins=10, color=['green', 'brown'], alpha=0.5)
plt.legend(['Arable land', 'Forest land'])
plt.title('Distribution of Arable and Forest Land')
plt.xlabel('Land area')
plt.ylabel('Frequency')
plt.show()


fig, ax = plt.subplots()
ax.scatter(arable_df['arable_land'], [results_df['arable_skewness'], results_df['forest_skewness']], color=['green', 'brown'])
ax.set_title('Skewness of Arable and Forest Land')
ax.set_xlabel('Land area')
ax.set_ylabel('Skewness')


# Bar plot of arable land and forest land means
plt.bar(results_df['country'], results_df['arable_mean'], color='green', alpha=0.5)
plt.bar(results_df['country'], results_df['forest_mean'], color='brown', alpha=0.5)
plt.legend(['Arable land', 'Forest land'])
plt.title('Mean Land Area of Arable and Forest Land')
plt.xlabel('Country')
plt.ylabel('Land area')
plt.show()

