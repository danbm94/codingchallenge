import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os


# load and investigate the data here:


df= pd.read_csv("/Users/danielbustillo/Documents/GitHub/codingchallenge/tennis_ace_starting/tennis_stats.csv")


print(df.head())



print(df.info())




# perform exploratory analysis here:

#Create a scatter plot of numerical values to see the relationship between the variables
for column in df.columns:
  if df[column].dtype == "int64" or df[column].dtype == "float64":
    plt.scatter(df[column], df['Wins'])
    plt.title(f'Winnings vs {column}')
    plt.xlabel(column)
    plt.ylabel('Winnings')
    plt.show()







## perform single feature linear regressions here:

#Split the data
train, test = train_test_split(df, train_size=0.8,random_state=42)


regression = LinearRegression()
model = regression.fit(train[['FirstServeReturnPointsWon']], train['Winnings'])

print(model.coef_)

model2 = regression.fit(train[['Aces']], train['Winnings'])

print(model2.coef_)

model3 = regression.fit(train[['BreakPointsOpportunities']], train['Winnings'])

print(model3.coef_)





















## perform two feature linear regressions here:


features = ['BreakPointsOpportunities' , 'FirstServeReturnPointsWon']

model4 = regression.fit(train[features], train['Winnings'])

print(model4.coef_)



















## perform multiple feature linear regressions here:
features = [['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon',
'SecondServePointsWon','SecondServeReturnPointsWon','Aces',
'BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities',
'BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon',
'ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon',
'TotalServicePointsWon']]
model5 = regression.fit(train[features], train['Winnings'])

print(model5.coef_)

