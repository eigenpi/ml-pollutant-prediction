# this code builds a model and then evaluates it for a number
# of possible values for a selected hyperparameter;
# the hyperparameter that is investigated is selected 
# by hard-coding the argument passed to run();
# for investigating the number of past-time-steps to use, a 
# different (from run) function is used; that is run_search_for_n_input_steps()
# credits: some portions of this code was developed starting from code
# from machinelearningmastery.com; thanks Jason!

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

import numpy as np
from numpy import array
from numpy import mean
from numpy import concatenate

import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from pandas import Series
from pandas import datetime

from matplotlib import pyplot
from math import sqrt


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
    
    # commented code is not currently used;
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

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df = df.drop(0)
    return df

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, yhat):
    new_row = [x for x in X] + [yhat]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

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

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model

# run a repeated experiment
def run_experiment(repeats, a_epochs, a_batch_size, a_units_count, a_optimizer,
                   a_n_input_steps, n_timesteps, n_features, n_variables, 
                   train_X, train_y, test_X, test_y, scaler):

    # run experiment
    error_scores = list()
    for r in range(repeats):
        print('r =', r)
        # fit the model
        
        # MODEL #1 --- Simple LSTM
        model = Sequential()
        model.add(LSTM(a_units_count, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1))

        model.compile(loss='mae', optimizer=a_optimizer) 
        history = model.fit(train_X, train_y, 
                            epochs=a_epochs, batch_size=a_batch_size, 
                            validation_data=(test_X, test_y), verbose=2, shuffle=False)
        # plot history
        #pyplot.plot(history.history['loss'], label='train')
        #pyplot.plot(history.history['val_loss'], label='test')
        #pyplot.xlabel('Epoch During Training')   
        #pyplot.ylabel('Train and Test Loss')
        #pyplot.legend()
        #yplot.show()        
               
        
        # >>> Version 1:
        # make predictions on the test portion, in one shot
        # for each new prediction the input has NO2 as the actual measurements
        # taht were part of the dataset - but, in reality if NO2 sensor is to be replaced
        # withthis LSTM model, then, the input value of NO2 should be the previous
        # estimations/predicyions of NO2! the question is at the start of the test sequence, what 
        # do I substitute NO2 measurements with?
        # I will try the average value of all NO2 measurements from the train portion;
        # see version 2 for that!
        yhat = model.predict(test_X)
        yhat = yhat.reshape((yhat.shape[0], 1)) # 1 was n_result

        
        # >>> Version 2:
# =============================================================================
#         avg_value_from_train_portion = mean(train_y[-24:]) # last 24 hours avg.? 
#         # buffer that will always shift left and then place new prediction in the last entry of the buffer;
#         # thus the buffer always has the last "n" predictions as a moving window, and its
#         # values need to be used to replace the NO2 values from the test_X accordingly
#         # so that we feed into model durig evaluation past predictions and not real measurements;
#         # comparison will be done against the ground truth values, i.e., reals measurements; 
#         my_buffer_prev_predictions = list() 
#         for j in range(0, a_n_input_steps):
#             my_buffer_prev_predictions.append(avg_value_from_train_portion)
#         # walk-forward validation/testing
#         yhat = list()
#         for i in range(len(test_X)):
#             # (1) replace the actual real measurements from this entry of the test data
#             # with the values from rotating buffer;
#             for j in range(0, a_n_input_steps):
#                 test_X[i, j, 0] = my_buffer_prev_predictions[j]
#             # (2) then do the prediction for this text_X[i] that has NO2/SO2 repleced with past predictions
#             # forecast the next day, as 24 values
#             input_x = test_X[i]
#             input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
#             yhat_i = model.predict(input_x, verbose=0)
#             # we only want the vector forecast
#             yhat_i_0 = yhat_i[0,0] #np.float32(yhat_i[0])
#             yhat.append(yhat_i_0)
#             # (3) update rotating buffer with last predicted value;
#             for j in range(1, a_n_input_steps):
#                 my_buffer_prev_predictions[j-1] = my_buffer_prev_predictions[j]
#             my_buffer_prev_predictions[a_n_input_steps-1] = yhat_i_0
#         #reshape yhat
#         yhat = array(yhat)
#         yhat = yhat.reshape((yhat.shape[0], 1)) # 1 was n_result
# =============================================================================
   

        # invert scaling for forecast
        r_test_X = test_X.reshape((test_X.shape[0], n_timesteps*n_features)) 
        # take only the last (n_variables-1) columns from test_X
        # and concatenate with yhat to construct an array similar to the
        # original one of 5 columns, where the 1st one is NO2 or SO2 and then
        # inverse scalar to get to values before scaling;
        inv_yhat = concatenate((yhat, r_test_X[:, -(n_variables-1):]), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]
        # invert scaling for actual
        r_test_y = test_y.reshape((len(test_y), 1))
        inv_y = concatenate((r_test_y, r_test_X[:, -(n_variables-1):]), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,0]

        #calculate MSE
        mse = mean_squared_error(inv_y, inv_yhat)
        print('Test MSE: %.3f' % mse)

        # calculate RMSE
        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
        print('%d) Test RMSE: %.3f' % (r+1, rmse))
        error_scores.append(rmse)
        
        mape = mean_absolute_percentage_error(inv_y, inv_yhat)
        print('Test MAPE: %.3f' % (mape))

        mae = mean_absolute_error(inv_y,inv_yhat)
        print('Test MAE: %.3f' % (mae))
    return error_scores


# what_to_vary = 'epoch', 'batch_size', 'units_count', 'optimizer', 'nothing'
def run(what_to_vary='nothing'):
    # load one dataset
    dataset = read_csv('dataset1_NO2.csv',
    #dataset = read_csv('dataset1_PM25.csv',                                 
    #dataset = read_csv('dataset2_NO2.csv',
    #dataset = read_csv('dataset2_PM25.csv', 
    #dataset = read_csv('dataset3_NO2.csv',
    #dataset = read_csv('dataset3_PM25.csv',
    #dataset = read_csv('dataset4_NO2.csv',
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
    n_input_steps = 2 # 0 for only current (t), 1 for (t-1,t), 2 for (t-2.t-1,t), etc. 
    # how many steps ahead to predict, starting with current (t)
    n_output_steps = 1 # 1 for current step (t), 2 for (t,t+1), etc.
    n_variables = 6 # all columns in dataset are the multiple variables (multivariate) features or inputs; 
    # frame dataset as for supervised learning; comments above for this function;
    reframed = series_to_supervised(scaled, n_input_steps, n_output_steps) 
    print('Shape of reframed:')
    print(reframed)
    print(reframed.shape)

    # split into train and test sets
    values = reframed.values
    print('Length of values:')
    print(len(values))
    split_index = int(len(values) * 0.7) 
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

    #n_timesteps, n_features = 1, n_observations # <--- must use for DNN
    #n_timesteps, n_features = n_observations, 1
    n_timesteps, n_features = n_input_steps, n_variables
    
    train_X = train_X.reshape((train_X.shape[0], n_timesteps, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_timesteps, n_features))
    train_y = train_y.reshape((train_y.shape[0], n_result)) # we only have 1 output
    test_y = test_y.reshape((test_y.shape[0], n_result)) # we only have 1 output
    print('train_X, train_y after 3D reshape, test_X, test_y:')
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    
    # experiments
    repeats = 5 # 30
    results = DataFrame()  
    if (what_to_vary == 'epoch'):
        batch_size=30
        units_count=30
        optimizer='Adam'
        # vary training epochs
        epochs = [20, 50, 100, 200, 500, 1000] 
        for e in epochs:
            results[str(e)] = run_experiment(repeats, e, batch_size, units_count, optimizer,
                                             n_input_steps, n_timesteps, n_features, n_variables,
                                              train_X, train_y, test_X, test_y, scaler)
        # summarize results
        print(results.describe())
        # save boxplot
        pyplot.rc('font', size=14)
        pyplot.rc('axes', titlesize=14)
        pyplot.xlabel('epoch')   
        pyplot.ylabel('RMSE')
        pyplot.boxplot(results, labels=epochs)
        pyplot.savefig('d1_search_epoch.png', dpi=300)
        pyplot.show()                                                                                 
    elif (what_to_vary == 'batch_size'):
        epoch=20
        units_count=30
        optimizer='Adam'
        # vary training batch_sizes
        batch_sizes = [10, 30, 50, 70, 100, 200] 
        for bs in batch_sizes:
            results[str(bs)] = run_experiment(repeats, epoch, bs, units_count, optimizer,
                                              n_input_steps, n_timesteps, n_features, n_variables,
                                              train_X, train_y, test_X, test_y, scaler)   
        # summarize results
        print(results.describe())
        # save boxplot
        pyplot.rc('font', size=14)
        pyplot.rc('axes', titlesize=14)
        pyplot.xlabel('batch_size')   
        pyplot.ylabel('RMSE')
        pyplot.boxplot(results, labels=batch_sizes)
        pyplot.savefig('d1_search_batch_size.png', dpi=300)
        pyplot.show()
    elif (what_to_vary == 'units_count'):
        epoch=20
        batch_size=30
        optimizer='Adam'
        # vary training units per layer
        units_counts = [10, 20, 30, 40, 50, 100] # units per layer
        for uc in units_counts:
            results[str(uc)] = run_experiment(repeats, epoch, batch_size, uc, optimizer,
                                              n_input_steps, n_timesteps, n_features, n_variables,
                                              train_X, train_y, test_X, test_y, scaler)   
        # summarize results
        print(results.describe())
        # save boxplot
        pyplot.rc('font', size=14)
        pyplot.rc('axes', titlesize=14)
        pyplot.xlabel('units/layer')   
        pyplot.ylabel('RMSE')
        pyplot.boxplot(results, labels=units_counts)
        pyplot.savefig('d1_search_units_count.png', dpi=300)
        pyplot.show()
    elif (what_to_vary == 'optimizer'):
        epoch=20
        batch_size=30
        units_count=30
        # try different optimizers
        optmizer_list = ['Adam', 'sgd', 'Adagrad', 'RMSprop', 'Adadelta'] 
        for opt in optmizer_list:
            results[str(opt)] = run_experiment(repeats, epoch, batch_size, units_count, opt, 
                                              n_input_steps, n_timesteps, n_features, n_variables,
                                              train_X, train_y, test_X, test_y, scaler)   
        # summarize results
        print(results.describe())
        # save boxplot
        pyplot.rc('font', size=14)
        pyplot.rc('axes', titlesize=14)
        pyplot.xlabel('optimizer')   
        pyplot.ylabel('RMSE')
        pyplot.boxplot(results, labels=optmizer_list)
        pyplot.savefig('d1_search_optimizer.png', dpi=300)
        pyplot.show()
    else: # do not vary any hyperparameter, but, just repeat the experiment
        epoch=20
        batch_size=30 
        units_count=30 
        optimizer='Adam'
        results['fixed_params'] = run_experiment(repeats, epoch, batch_size, units_count, optimizer,
                                                 n_input_steps, n_timesteps, n_features, n_variables,
                                                 train_X, train_y, test_X, test_y, scaler)   
        # summarize results
        print(results.describe())
        # save boxplot
        pyplot.rc('font', size=14)
        pyplot.rc('axes', titlesize=14)
        #pyplot.xlabel('batch_size')   
        pyplot.ylabel('RMSE')
        pyplot.boxplot(results)
        pyplot.savefig('d1_fixed_params_repeated.png', dpi=300)
        pyplot.show()
        
        
def run_search_for_n_input_steps():
    # load dataset
    dataset = read_csv('dataset1_NO2.csv',
    #dataset = read_csv('dataset1_PM25.csv',                                 
    #dataset = read_csv('dataset2_NO2.csv',
    #dataset = read_csv('dataset2_PM25.csv', 
    #dataset = read_csv('dataset3_NO2.csv',
    #dataset = read_csv('dataset3_PM25.csv',
    #dataset = read_csv('dataset4_NO2.csv',
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
    
    # how many steps ahead to predict, starting with current (t)
    n_output_steps = 1 # 1 for current step (t), 2 for (t,t+1), etc.
    n_variables = 6 # all columns in dataset are the multiple variables (multivariate) features or inputs;  
    
    # experiments
    repeats = 20 
    results = DataFrame()  
    epoch=20
    batch_size=30
    units_count=30
    optimizer='Adam'    
    
    # how many steps back in time to use as input features, starting with
    # current (t) but without the 1st column variable, whcih we predict,
    # and past times (...t-2,t-1) all columns
    # n_input_steps = 2 # 0 for only current (t), 1 for (t-1,t), 2 for (t-2.t-1,t), etc.    
    # vary number of input steps used
    n_input_steps_values = [2, 4, 6, 8, 12, 24] 
    for n_input_steps in n_input_steps_values:
       
        # frame dataset as for supervised learning; comments above for this function;
        reframed = series_to_supervised(scaled, n_input_steps, n_output_steps) 
        print('Shape of reframed:')
        print(reframed)
        print(reframed.shape)

        # split into train and test sets
        values = reframed.values
        print('Length of values:')
        print(len(values))
        split_index = int(len(values) * 0.7) 
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

        #n_timesteps, n_features = 1, n_observations # <--- must use for DNN
        #n_timesteps, n_features = n_observations, 1
        n_timesteps, n_features = n_input_steps, n_variables
        
        train_X = train_X.reshape((train_X.shape[0], n_timesteps, n_features))
        test_X = test_X.reshape((test_X.shape[0], n_timesteps, n_features))
        train_y = train_y.reshape((train_y.shape[0], n_result)) # we only have 1 output
        test_y = test_y.reshape((test_y.shape[0], n_result)) # we only have 1 output
        print('train_X, train_y after 3D reshape, test_X, test_y:')
        print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
        results[str(n_input_steps)] = run_experiment(repeats, epoch, batch_size, units_count, optimizer,
                                                     n_input_steps, n_timesteps, n_features, n_variables,
                                                     train_X, train_y, test_X, test_y, scaler)


 
    # summarize results
    print(results.describe())
    # save boxplot
    pyplot.rc('font', size=14)
    pyplot.rc('axes', titlesize=14)
    pyplot.xlabel('input steps')   
    pyplot.ylabel('RMSE')
    pyplot.boxplot(results, labels=n_input_steps_values)
    pyplot.savefig('d1_search_n_steps.png', dpi=300)
    pyplot.show()  





# entry point
# (1) vary one of the following hyperparameters;
# what_to_vary = 'epoch', 'batch_size', 'units_count', 'optimizer', 'nothing'
#run('optimizer')

# (2) or vary number of time steps used as input into model;
run_search_for_n_input_steps()
