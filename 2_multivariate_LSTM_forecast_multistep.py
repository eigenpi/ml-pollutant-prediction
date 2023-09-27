# this code builds an LSTM a model and then evaluates it for a number
# of datasets; each dataset is used separately by commenting/uncommenting 
# reading it (see code below...); 
# this is only for SINGLE step out, by design of the code!!!
# input can be single step or multistep multivariate 
# NOTE: normalization is of whole array/matrix not column wise

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import TimeDistributed
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from numpy import array
from numpy import mean
import pandas as pd


# function to convert series to supervised learning format
# initial dataset is of the form:
# ...
# t-n: NO2,O3,CO,PM2.5,PM10
# ...
# t-1: NO2,O3,CO,PM2.5,PM10
#   t: NO2,O3,CO,PM2.5,PM10 <--- am interested to predict NO2(t) based on one or multiple previous steps data
# t+1: NO2,O3,CO,PM2.5,PM10 
# ...
# t+m: NO2,O3,CO,PM2.5,PM10
# ...
# cases:
# 1) n_in=0, n_out=1
#    predict NO2(t) as function of O3,CO,PM2.5,PM10(t)
# 2) n_in=n>=1, n_out=1
#    predict NO2(t) as function of NO2,O3,CO,PM2.5,PM10(t-n,...,t-1) + O3,CO,PM2.5,PM10(t)
# 3) n_in=0, n_out=m>=1
#    predict NO2(t,...,t+m) as function of O3,CO,PM2.5,PM10(t)
# 4) n_in=n>=1, n_out=m>=1

def series_to_supervised(data, n_in=1, m_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    
	# (1) input sequence (t-n,...,t-1,t)
    # for i in range(n_in, -1, -1): # n, n-1,...,0
    #     if i == 0: 
    #         # this is the case of (t)           
    #         # I want to use as input all variables except the one I will predict,
    #         # which is always in 1st column of dataset; for example, in the case of:
    #         # ...
    #         # t-1: NO2,O3,CO,PM2.5,PM10
    #         #   t: NO2,O3,CO,PM2.5,PM10
    #         # I want to include in the input "O3,CO,PM2.5,PM10" from (t) and predict
    #         # NO2(t); if n_in>0, then, we also include in the input NO2,O3,CO,PM2.5,PM10
    #         # from time (t-n,...,t-1)
    #         cols.append( df[df.columns[1:n_vars]] )
    #         names += [('var%d(t)' % (j+1)) for j in range(1,n_vars)]
    #     else:
    #         # this is the case of (t-n,...,t-1) time points;
    #         # we include all columns of dataset, including the 1st that we will predict;
    #         # shift() function in Pandas that will push all values in a series 
    #         # down by a specified number places; pushed-down series will have 
    #         # a new position at the top with no value;
    #         cols.append(df.shift(i))
    #         names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    for i in range(n_in, 0, -1): # n, n-1,...,1
        # this is the case of (t-n,...,t-1) time points;
        # we include all columns of dataset, including the 1st that we will predict;
        # shift() function in Pandas that will push all values in a series 
        # down by a specified number places; pushed-down series will have 
        # a new position at the top with no value;
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

            
    # (2) forecast sequence (t, t+1, ... t+m)
    # we will actually forecast only current time (t) even though
    # here we can set it such that we can predict multiple steps m_out 
    # ahead in time;
    # also we only focus on the variable in the 1st column of the dataset NO2;
    # we do not predict all variables in a row of the initial dataset;
    # no multivariate output; just single variate output single (t) or 
    # multi-step (t, t+1, ... t+m)
    for i in range(0, m_out):
        # take only the 1st column and append shifted back in time as many times as needed;
        cols.append( df[df.columns[0]].shift(-i) )       
        if i == 0:
            names += ['var1(t)']
        else:
            names += ['var1(t+%d)' % i]
	
    # (3) put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
	# (4) drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# load cleaned dataset
#dataset = read_csv('dataset1_NO2.csv',
#dataset = read_csv('dataset1_PM25.csv',                                 
#dataset = read_csv('dataset2_NO2.csv',
#dataset = read_csv('dataset2_PM25.csv', 
#dataset = read_csv('dataset3_NO2.csv',
#dataset = read_csv('dataset3_PM25.csv',
dataset = read_csv('dataset4_NO2.csv',
#dataset = read_csv('dataset4_PM25.csv',                   
                   header= 0, 
                   infer_datetime_format=True, 
                   parse_dates=['Datetime'], 
                   index_col=['Datetime'])

values = dataset.values
print(values)
print(values.shape)

# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# how many steps back in time to use as input features, starting with
# current (t) but without the 1st column variable, whcih we predict,
# and past times (...t-2,t-1) all columns
n_input_steps = 2 # 6, 0 for only current (t), 1 for (t-1,t), 2 for (t-2.t-1,t), etc. 
# how many steps ahead to predict, starting with current (t)
n_output_steps = 1 # 1 for current step (t), 2 for (t,t+1), etc.
n_variables = 6 # 5 all columns in dataset are the multiple variables (multivariate) features or inputs; 
# frame dataset as for supervised learning; comments above for this function;
reframed = series_to_supervised(scaled, n_input_steps, n_output_steps) 
print('Shape of reframed:')
print(reframed)
print(reframed.shape)


# split into train and test sets
values = reframed.values
print('Length of values:')
print(len(values))
split_index = int(len(values) * 0.9) 
print('Split index:', split_index)
train = values[:split_index, :]
test = values[split_index:, :]


# split columns into input and outputs pairs;
n_observations = n_input_steps * n_variables # total number of data values or observations used as input
n_result = 1 # we only predict one time step ahead as a result 
train_X, train_y = train[:, :n_observations], train[:, -n_result]
test_X, test_y = test[:, :n_observations], test[:, -n_result]
print('train_X, length of train_X, train_y:')
print(train_X.shape, len(train_X), train_y.shape)


# reshape input to be 3D [samples, timesteps, features]
# NOTE: I changed this compared to MLM because n_obs is not an integer number
# of num of columns; here I use as input the columns 1:n_features at time (t)
# also as input, not only all-columns of previous times (...,t-1);
# NOTE: here I can use:
# 1) timesteps, features == n_res( =1), n_obs
# 2) timesteps, features == n_obs, n_res( =1)
# #1 gives better results; i.e., when I consider all values from past times and
# current values at (t) of all pollutants as a features in one big timestep
# rather than treat them as multiple time steps of a single feature variable;
# this needs more investigation!

# In terms of how we FRAME the (input,output) pairs, which must be fed to the model
# dureing training, the input must be presented as ("n_timesteps", "n_features"); 
# we can look at the n_observations inputs in several ways:
# 1) 1 "timestep", n_observations "features" <--- gives better results than 2) below
#    I treat NO2,O3,CO,PM2.5,PM10 from times (t-n,...,t-1) as one larger 
#    number of features fed into the model in one step;
# 2) n_observations timesteps, 1 feature
#    I feed all combined NO2,O3,CO,PM2.5,PM10 from times (t-n,...,t-1)
#    fed each in a different timestep
# 3) n_input timesteps, n_features features
#    this is how the actual data is!
#
#n_timesteps, n_features = 1, n_observations # <--- must use for DNN
#n_timesteps, n_features = n_observations, 1
n_timesteps, n_features = n_input_steps, n_variables

train_X = train_X.reshape((train_X.shape[0], n_timesteps, n_features))
test_X = test_X.reshape((test_X.shape[0], n_timesteps, n_features))
train_y = train_y.reshape((train_y.shape[0], n_result)) # we only have 1 output
test_y = test_y.reshape((test_y.shape[0], n_result)) # we only have 1 output
print('train_X, train_y after 3D reshape, test_X, test_y:')
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# MODEL --- Simple LSTM
model = Sequential()
model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='sgd') #mae, adam are best


# let's add a callback, "EarlyStopping"; this one of the techiniques 
# to prevent overfitting; and save runtime;
early_stop = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(train_X, train_y, 
                    epochs=200, 
                    batch_size=30, 
                    #validation_data=(test_X, test_y), 
                    validation_split=0.2,
                    verbose=2, shuffle=False,
                    callbacks=[early_stop])

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.xlabel('Epoch During Training')   
pyplot.ylabel('Train and Test Loss')
pyplot.legend()
pyplot.show()
# evaluate model using test true test data that has actual measured NO2/SO2 values
# inside test portion too! 
# print("Evaluate model on test data")
# results = model.evaluate(test_X, test_y, batch_size=10)
# print("test loss, test acc:", results)


# =============================================================================
# # > Version 1:
# # make predictions on the test portion, in one shot
# # for each new prediction the input has NO2 as the actual measurements
# # taht were part of the dataset - but, in reality if NO2 sensor is to be replaced
# # withthis LSTM model, then, the input value of NO2 should be the previous
# # estimations/predicyions of NO2! the question is at the start of the test sequence, what 
# # do I substitute NO2 measurements with?
# # I will try the average value of all NO2 measurements from the train portion;
# # see version 2 for that!
# yhat = model.predict(test_X)
# yhat = yhat.reshape((yhat.shape[0], n_result)) 
# =============================================================================

# > Version 2:
avg_value_from_train_portion = mean(train_y[-24:]) 
# buffer that will always shift left and then place new prediction in the last entry of the buffer;
# thus the buffer always has the last "n" predictions as a moving window, and its
# values need to be used to replace the NO2/SO2 values from the test_X accordingly
# so that we feed into model durig evaluation past predictions and not real measurements;
# comparison will be done against the ground truth values, i.e., reals measurements; 
my_buffer_prev_predictions = list() 
for j in range(0, n_input_steps):
    my_buffer_prev_predictions.append(avg_value_from_train_portion)
# walk-forward validation/testing
yhat = list()
for i in range(len(test_X)):
    # (1) replace the actual real measurements from this entry of the test data
    # with the values from rotating buffer;
    for j in range(0, n_input_steps):
        test_X[i, j, 0] = my_buffer_prev_predictions[j]
    # (2) then do the prediction for this text_X[i] that has NO2/SO2 repleced with past predictions
    # forecast the next day, as 24 values
    input_x = test_X[i]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    yhat_i = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat_i_0 = yhat_i[0,0] #np.float32(yhat_i[0])
    yhat.append(yhat_i_0)
    # (3) update rotating buffer with last predicted value;
    for j in range(1, n_input_steps):
        my_buffer_prev_predictions[j-1] = my_buffer_prev_predictions[j]
    my_buffer_prev_predictions[n_input_steps-1] = yhat_i_0
#reshape yhat
yhat = array(yhat)
yhat = yhat.reshape((yhat.shape[0], n_result)) 
    

# invert scaling for forecast
test_X = test_X.reshape((test_X.shape[0], n_timesteps*n_features)) 
# take only the last (n_variables-1) columns from test_X
# and concatenate with yhat to construct an array similar to the
# original one of 5 columns, where the 1st one is NO2 or SO2 and then
# inverse scalar to get to values before scaling;
inv_yhat = concatenate((yhat, test_X[:, -(n_variables-1):]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -(n_variables-1):]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]


#calculate MSE
mse = mean_squared_error(inv_y, inv_yhat)
print('Test MSE: %.3f' % mse)

# calculate RMSE
rmse = sqrt(mse)
print('Test RMSE: %.3f' % rmse)

# calculate mean absolute percentage error (MAPE)
# as relative errors expressed as percentages
# 100 * (predicted - actual)/actual
# in other  words, report mean absolute percentage error (MAPE)
# which is commonly used to measure the predictive accuracy of models
def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] == 0:
            actual[j] += 1E-6          
        res[j] = 100 * abs(predicted[j] - actual[j]) / actual[j]
    return res

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) 

test_MAPE = mean_absolute_percentage_error( inv_y, inv_yhat)
print('Test MAPE: %.3f' % (test_MAPE))

mae = mean_absolute_error(inv_y,inv_yhat)
print('Test MAE: %.3f' % (mae))


# plot actual and predictions of pollution for test portion of dataset
num_points_to_plot = 100 #200
pyplot.rc('font', size=14)
pyplot.rc('axes', titlesize=14)
pyplot.plot(inv_y[0:num_points_to_plot], label='Actual', linewidth=4.0) 
pyplot.plot(inv_yhat[0:num_points_to_plot], label='Predicted', linewidth=2.0) 
pyplot.legend()
pyplot.xlabel('Time Steps [hours]')   
pyplot.ylabel('NO2 [ug/m^3]') #'NO2 [ug/m^3]' d1,d2,d4 and 'NO2 [ppb]' for d3
pyplot.title('Dataset 4')
pyplot.savefig('fig_prediction_d4_NO2_2steps_mse_sgd_repl.png', dpi=300) # <--- change y label too!!!
pyplot.show()


# plot error for the testing portion of dataset;
# pyplot.plot( percentage_error(np.asarray(inv_y), np.asarray(inv_yhat))[0:200] ) 
# pyplot.xlabel('Time Step')
# pyplot.ylabel('Absolute Error')
# pyplot.show()

#df = DataFrame({"Predicted SO2 dataset1" : inv_y, "Actual SO2 dataset1" : inv_yhat})
#df.to_csv("results.csv", index=False)
