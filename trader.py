
# coding: utf-8

# In[1]:

import pandas  as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import argparse
import csv
#import matplotlib.pyplot as plt #matplotlib==2.2.0

class Trader():
    def drawPlot(self, result_df):
        predict_line = result_df['predict']
        actual_line = result_df['actual']
        mean_line = result_df['10days_mean']
        plt.plot(predict_line, 'b', actual_line, 'r', mean_line, 'g')
        plt.xlabel('days', fontsize=20)
        plt.ylabel('price', fontsize=15)
        plt.legend(['predict','actual', '10 days average'], loc = 2)
        fig = plt.gcf()
        fig.set_size_inches(20, 20)
        plt.show()

    def train(self, training_data, poly, regression_model):
        feature_name = ["open", "high", "low", "close"]
        training_data.columns = feature_name
        data_length = len(training_data['open'])

        training_data['10days_mean'] = pd.Series(np.zeros(data_length), index=training_data.index)
        training_data['tomorrow'] = pd.Series(np.zeros(data_length), index=training_data.index)

        #5 days average
        train_mean_values = []
        #train_mean_values[0:5] = [0]*4

        for i in range(0, data_length):
            if i == 0:
                train_mean_values.append(training_data['open'].iloc[0])
            elif i > 0 :
                train_mean_values.append(np.mean(training_data['open'].iloc[0:i]))
            else:
                train_mean_values.append(np.mean(training_data['open'].iloc[i-4:i]))


        training_data['10days_mean'] = train_mean_values

        #tomorrow open
        training_data['tomorrow'].iloc[data_length-1] = training_data['open'][data_length-1]   #last data
        for i in range(0, data_length-1):
            training_data['tomorrow'].iloc[i] = training_data['open'].iloc[i+1]

        features_train = training_data[['open', 'high', 'low', 'close', '10days_mean']]
        label_train = training_data[['tomorrow']]

        #poly.fit(features_train, label_train.values.ravel())

        poly_x = poly.fit_transform(features_train)


        regression_model.fit(features_train, label_train)

        #print(training_data)

    def predict(self, testing_data, regression_model):
        result_df = testing_data
        features_test = result_df[['open', 'high', 'low', 'close']]
        test_data_length = len(result_df['open'])
        result_df['10days_mean'] = pd.Series(np.zeros(test_data_length), index=testing_data.index)

        action_list = []
        money = 0
        pred_list = []
        actual_list = []


        for i in range(0, test_data_length-1):
            action = 0

            #cauculate 10 days mean line
            if i == 0:
                result_df['10days_mean'].iloc[i] = result_df['close'].iloc[0]
            elif i < 10 :
                result_df['10days_mean'].iloc[i] = np.mean(result_df['close'].iloc[0:i])
            elif i == test_data_length-2:
                result_df['10days_mean'].iloc[i] = np.mean(result_df['close'].iloc[i-10:i])
                result_df['10days_mean'].iloc[test_data_length-1] = np.mean(result_df['close'].iloc[i-9:i+1])
            else:
                result_df['10days_mean'].iloc[i] = np.mean(result_df['close'].iloc[i-10:i])


            #predict next day price
            pred = regression_model.predict(result_df.iloc[i].values.reshape([1,-1]))

            pred_list.append(pred[0][0])


            if sum(action_list) == 0:  #no stock
                if pred - result_df['10days_mean'].iloc[i] > 10:   #(buy) predict next day > 10 days mean
                    action = 1

                    money = money - result_df['open'].iloc[i+1]     #buy in next day open price

                    # print("position" + repr(i))
                    # print("action" + repr(action))
                    # print("buy:" + repr(result_df['open'].iloc[i+1]))
                    # print("-----------------------")


            elif sum(action_list) == 1:    #one stock
                if result_df['10days_mean'].iloc[i] - pred > 10:  #(sell) predict next day < 10 days mean
                    action = -1

                    money = money + result_df['open'].iloc[i+1]

                    # print("position" + repr(i))
                    # print("action" + repr(action))
                    # print("sell:" + repr(result_df['open'].iloc[i+1]))
                    # print("-----------------------")
                else:
                    action = 0

            elif sum(action_list) == -1:    #sell short
                if result_df['10days_mean'].iloc[i] - pred > 10:
                    action = 1
                    money = money - result_df['open'].iloc[i+1]

                    # print("position" + repr(i))
                    # print("action" + repr(action))
                    # print("buy:" + repr(result_df['open'].iloc[i]))
                    # print("-----------------------")
                else:
                    action = 0


            action_list.append(action)

        # print(action_list)
        # print(len(action_list))

        #in the last day if hold/short stock, then sell the stock
        if sum(action_list) != 0:
            money += result_df['open'].iloc[test_data_length-1]

        print(money)

        #actual tomorrow value
        result_df['actual'] = pd.Series(np.zeros(test_data_length), index=testing_data.index)

        result_df['actual'].iloc[test_data_length-1] = result_df['open'].iloc[test_data_length-1]
        for i in range(0, test_data_length-1):
            result_df['actual'].iloc[i] = result_df['open'].iloc[i+1]

        pred_list.append(result_df['open'].iloc[test_data_length-1])
        result_df['predict'] = pred_list
        #print(result_df)

        #draw plot
        #self.drawPlot(result_df)

        # print(action_list)
        # print(len(action_list))



        with open('output.csv', 'w') as output_file:
            wf = csv.writer(output_file,lineterminator='\n')
            for val in action_list:
                wf.writerow([val])

        #action_list.append(0)
        #result_df['action'] = action_list
        #result_df.to_csv('result_df.csv')


# In[3]:

if __name__ == '__main__':
    feature_name = ["open", "high", "low", "close"]
    parser = argparse.ArgumentParser()

    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')

    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')

    args = parser.parse_args()
    train = args.training
    test = args.testing

    training_data = pd.read_csv(train, header=None)
    testing_data = pd.read_csv(test, header=None)
    testing_data.columns = feature_name

    #training_data = load_data(args.training)

    trader = Trader()

    poly = PolynomialFeatures(degree=5)
    regression_model = LinearRegression()

    #training
    trader.train(training_data, poly, regression_model)
    #testing
    trader.predict(testing_data, regression_model)
