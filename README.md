https://nbviewer.jupyter.org/github/ElmoLee822/AI_Trader/blob/master/trader.ipynb

# AI Trader
### Concept
Cauculate "previous 10 days average" price and add it in a new column as feature when checking one row in testing data. Use "open", "high", "low", "close" and "10 days average" as the five features to predict the next day 'open' price by using polynomial regression. If the predict value is greater than '10 days average' 10 then buy the stock and the action is 1. If the predict value is small than '10 days average' 10 then buy the stock and the action is -1.

### Moving Average
In statistics, a moving average (rolling average or running average) is a calculation to analyze data points by creating series of averages of different subsets of the full data set.  
The average price for a period of time represents the average cost of buying stocks over a period of time. This can be used to judge the trend of stock price development and is the most popular technical indicator.
Simple Moving Average(SMA), is the unweighted mean of the previous n data.

![image](https://github.com/ElmoLee822/AI_Trader/blob/master/moving_average_formula.PNG)

### 10 days line
It represents the average purchase cost for the past 10 days.

### Method
Use Polynomial Features, which we can ask to convert our values of X into all possible polynomials change the value of degree to 5 to train the model. And use the Linear Regression to predict the stock value in next day.

### Result Plot
blue line : predict value

red line : actual value

green line : 10 days average

![image](https://github.com/ElmoLee822/AI_Trader/blob/master/predict_stock.png)
