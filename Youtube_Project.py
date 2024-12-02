import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import Dataset
df = pd.read_csv('GlobalYouTubeStatistics.csv', encoding='unicode escape')

# Data Understanding
print(df.head())
print(df.info())
print(df.shape)
print(df.isnull().sum())

# Data Cleaning
print(df.duplicated().sum())
print(df[-df.duplicated()].shape[0])

NaNnumerical = df.select_dtypes(include=['float64', 'int64']).columns
NaNcategorical = df.select_dtypes(include='object').columns
df[NaNnumerical] = df[NaNnumerical].fillna(0)
df[NaNcategorical] = df[NaNcategorical].fillna('Others')
print(df.isnull().sum())

# Overview on Numerical columns
print(df.describe())

# Converting Subscribers, Video Views, Population, and Urban Population to millions
df['subscribers'] = (df['subscribers'] / 1000000).round(2)
df['video views'] = (df['video views'] / 1000000).round(2)
df['Population'] = (df['Population'] / 1000000).round(2)
df['Urban_population'] = (df['Urban_population'] / 1000000).round(2)
print(df[['subscribers', 'video views', 'Population', 'Urban_population']].tail())

# Rename columns
df.rename(columns={'Unemployment rate': 'Unemployment rate(%)'}, inplace=True)
print(df.head())

# Filter the rows where video views are not equal to 0
video = df[df['video views'] != 0]
print(video)

# Create a new column "AverageYearEarning"
df["AverageYearEarning"] = (df['lowest_yearly_earnings'] + df['highest_yearly_earnings'])
df["AverageYearEarning"] = (df["AverageYearEarning"] / 1000000).round(2)
print(df)

# 1. Top 10 YouTube Channels Based on the Number of Subscribers
top_10_channels = df[['Youtuber', 'subscribers']].sort_values(by='subscribers', ascending=False).head(10)
print("Top 10 YouTube Channels Based on the Number of Subscribers:")
print(top_10_channels)

# 2. Category with the Highest Average Number of Subscribers
category_avg_subs = df.groupby('category')['subscribers'].mean().sort_values(ascending=False).head(1)
print("Category with the Highest Average Number of Subscribers:")
print(category_avg_subs)

# 3. Average Number of Videos Uploaded by YouTube Channels in Each Category
avg_videos_per_category = df.groupby('category')['uploads'].mean()
print("Average Number of Videos Uploaded by YouTube Channels in Each Category:")
print(avg_videos_per_category)

# 4. Top 5 Countries with the Highest Number of YouTube Channels
top_5_countries = df['Country'].value_counts().head(5)
print("Top 5 Countries with the Highest Number of YouTube Channels:")
print(top_5_countries)

# 5. Distribution of Channel Types Across Different Categories
channel_type_distribution = df.groupby('category')['channel_type'].value_counts().unstack().fillna(0)
print("Distribution of Channel Types Across Different Categories:")
print(channel_type_distribution)

# 6. Correlation Between Number of Subscribers and Total Video Views for YouTube Channels
correlation_subs_views = df[['subscribers', 'video views']].corr()
print("Correlation Between Number of Subscribers and Total Video Views for YouTube Channels:")
print(correlation_subs_views)

# 7. Monthly Earnings Variation Across Different Categories
monthly_earnings_category = df.groupby('category')['highest_monthly_earnings'].mean()
print("Monthly Earnings Variation Across Different Categories:")
print(monthly_earnings_category)

# 8. Overall Trend in Subscribers Gained in the Last 30 Days Across All Channels
if 'subscribers_for_last_30_days' in df.columns:
    subscribers_for_last_30_days = df['subscribers_for_last_30_days'].sum()
    print("Overall Trend in Subscribers Gained in the Last 30 Days Across All Channels:")
    print(subscribers_for_last_30_days)
else:
    print("The column 'subscribers_for_last_30_days' does not exist in the dataset.")

# 9. Outliers in Terms of Yearly Earnings from YouTube Channels
yearly_earnings = df['highest_yearly_earnings']
Q1 = yearly_earnings.quantile(0.25)
Q3 = yearly_earnings.quantile(0.75)
IQR = Q3 - Q1
outliers = df[(yearly_earnings < (Q1 - 1.5 * IQR)) | (yearly_earnings > (Q3 + 1.5 * IQR))]
print("Outliers in Terms of Yearly Earnings from YouTube Channels:")
print(outliers[['Youtuber', 'highest_yearly_earnings']])

# 10. Distribution of Channel Creation Dates and Trend Over Time
creation_date_distribution = pd.to_datetime(df['created_date']).dt.year.value_counts().sort_index()
print("Distribution of Channel Creation Dates and Trend Over Time:")
print(creation_date_distribution)


# 11. Relationship Between Gross Tertiary Education Enrollment and Number of YouTube Channels in a Country
tertiary_education_vs_channels = df.groupby('Country')[['Gross tertiary education enrollment (%)', 'Youtuber']].count().reset_index()
print("Relationship Between Gross Tertiary Education Enrollment and Number of YouTube Channels in a Country:")
print(tertiary_education_vs_channels)

# 12. Unemployment Rate Among the Top 10 Countries with the Highest Number of YouTube Channels
top_10_countries = df['Country'].value_counts().head(10).index
unemployment_top_10_countries = df[df['Country'].isin(top_10_countries)].groupby('Country')['Unemployment rate(%)'].mean()
print("Unemployment Rate Among the Top 10 Countries with the Highest Number of YouTube Channels:")
print(unemployment_top_10_countries)

# 13. Average Urban Population Percentage in Countries with YouTube Channels
avg_urban_population_percentage = df.groupby('Country')['Urban_population'].mean()
print("Average Urban Population Percentage in Countries with YouTube Channels:")
print(avg_urban_population_percentage)


# 14. Patterns in Distribution of YouTube Channels Based on Latitude and Longitude Coordinates
latitude_longitude_distribution = df[['Latitude', 'Longitude']].dropna()
print("Patterns in Distribution of YouTube Channels Based on Latitude and Longitude Coordinates:")
print(latitude_longitude_distribution)

# 15. Correlation Between Number of Subscribers and Population of a Country
correlation_subs_population = df[['subscribers', 'Population']].corr()
print("Correlation Between Number of Subscribers and Population of a Country:")
print(correlation_subs_population)

# 16. Comparison of Top 10 Countries with Highest Number of YouTube Channels in Terms of Their Total Population
top_10_countries_population = df[df['Country'].isin(top_10_countries)].groupby('Country')['Population'].sum()
print("Comparison of Top 10 Countries with Highest Number of YouTube Channels in Terms of Their Total Population:")
print(top_10_countries_population)

# 17. Correlation Between Number of Subscribers Gained in the Last 30 Days and Unemployment Rate in a Country
correlation_subs_30days_unemployment = df[['subscribers_for_last_30_days', 'Unemployment rate(%)']].corr()
print("Correlation Between Number of Subscribers Gained in the Last 30 Days and Unemployment Rate in a Country:")
print(correlation_subs_30days_unemployment)


# 18. Distribution of Video Views for the Last 30 Days Across Different Channel Types
video_views_for_the_last_30_days_channel_types = df.groupby('channel_type')['video_views_for_the_last_30_days'].sum()
print("Distribution of Video Views for the Last 30 Days Across Different Channel Types:")
print(video_views_for_the_last_30_days_channel_types)


# 19. Seasonal Trends in Number of Videos Uploaded by YouTube Channels
upload_dates = pd.to_datetime(df['uploads'])
upload_dates_trend = upload_dates.dt.month.value_counts().sort_index()
print("Seasonal Trends in Number of Videos Uploaded by YouTube Channels:")
print(upload_dates_trend)


# 20. Average Number of Subscribers Gained Per Month Since Creation of YouTube Channels Till Now
df['created_date'] = pd.to_datetime(df['created_date'])
df['months_since_creation'] = (pd.to_datetime('today') - df['created_date']).dt.days // 30
df['avg_subs_per_month'] = df['subscribers'] / df['months_since_creation']
avg_subs_per_month = df['avg_subs_per_month'].mean()
print("Average Number of Subscribers Gained Per Month Since Creation of YouTube Channels Till Now:")
print(avg_subs_per_month)

# Plotting the top 10 YouTube channels based on the number of subscribers
plt.figure(figsize=(10, 6))
plt.barh(top_10_channels['Youtuber'], top_10_channels['subscribers'], color='orange')
plt.xlabel('Subscribers (in millions)')
plt.ylabel('YouTube Channel')
plt.title('Top 10 YouTube Channels by Subscribers')
plt.gca().invert_yaxis() 
plt.show()
