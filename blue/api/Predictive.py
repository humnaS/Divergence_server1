# load libraries

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
import requests, pandas as pd, numpy as np
from pandas import DataFrame
from io import StringIO
import time, json
from datetime import date
import statsmodels
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error

# Load and visualize the time series data
#  Run the following to convert the Pandas DataFrame into a time series with daily frequency:

import sys
import types

# @hidden_cell
import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell

df_fx_data['Date'] = pd.to_datetime(df_fx_data['Date'], format = '%Y-%m-%d')
indexed_df = df_fx_data.set_index('Date')

ts_euro = indexed_df['Value']
ts_euro.head(5)

# Visualize the raw data
# Visualize the time series to see how the Euro is trending against the US dollar over time:

plt.plot(ts_euro.index.to_pydatetime(), ts_euro.values)

#  Resample the data
# Using daily data for your time series contains too much variation, so you must first resample the time series
# data by week. Then use this resampled time series to predict the Euro exchange rates against the US Dollar:

ts_euro_week = ts_euro.resample('W').mean()
plt.plot(ts_euro_week.index.to_pydatetime(), ts_euro_week.values)


# Check for stationarity

# What is meant by checking the stationarity of a time series and why do you care about it?
#
# In a stationary time series, the statistical properties over time must be constant and autoconvariance
# must be time independent. Normally, when running a regular regression, you expect the observations to be
# independent of each other. In a time series, however, you know that the observations are time dependent.
# In order to use regression techniques on time dependent variables, the data must be stationary.
# The techniques that apply to independent random variables also apply to stationary random variables.
#
# There are two ways to check the stationarity of a time series. The first is plot the moving variance and observe
# if it remains constant over time. However, you might not always be able to make such visual inferences.
# The second way is to use the Dickey-Fuller test, a statistical test with the null hypothesis that the time series
# is non-stationary. If the test results in the test statistic significantly less than the critical values,
# we can reject the null hypothesis in favor of time series stationarity.
#
# Calculate the moving variances, plot the results and apply the Dickey-Fuller test on the time series:

def check_stationarity(timeseries):
    # Determing rolling statistics
    rolling_mean = timeseries.rolling(window=52, center=False).mean()
    rolling_std = timeseries.rolling(window=52, center=False).std()

    # Plot rolling statistics:
    original = plt.plot(timeseries.index.to_pydatetime(), timeseries.values, color='blue', label='Original')
    mean = plt.plot(rolling_mean.index.to_pydatetime(), rolling_mean.values, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std.index.to_pydatetime(), rolling_std.values, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dickey_fuller_test = adfuller(timeseries, autolag='AIC')
    dfresults = pd.Series(dickey_fuller_test[0:4],
                          index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dickey_fuller_test[4].items():
        dfresults['Critical Value (%s)' % key] = value
    print(dfresults)

    check_stationarity(ts_euro_week)

    # Because the test statistic is more than the 5% critical value and the p-value is larger than 0.05,
    # the moving average is not constant over time and the null hypothesis of the Dickey-Fuller test cannot be rejected.
    # This shows that the weekly time series is not stationary.
    #
    # Before you can apply ARIMA models for forecasting, you need to transform this time series into a stationary time series.

    # Stationarize the time series

    # If your time series reveals a trend or seasonality, this is an indication that it is non-stationary. You can stationarize
    # the time series by calculating the trend and seasonality and removing these factors from the model
    # Apply a nonlinear log transformation
    # Begin by applying a simple, nonlinear log transformation and checking for stationarity:



    ts_euro_week_log = np.log(ts_euro_week)
    check_stationarity(ts_euro_week_log)


    # The Dickey-Fuller test results confirm that the series is still non-stationary.
    # Again the test statistic is larger than the 5% critical value and the p-value larger than 0.05.
    # Remove trend and seasonality with decomposition
    # Next, decompose the time series to remove trend and seasonality from the data. Decomposition results
    # show an increasing trend and seasonal effect of approximately a 12 month cycle for the more recent weeks in the data set.

    decomposition = seasonal_decompose(ts_euro_week)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Select the most recent weeks
    ts_euro_week_log_select = ts_euro_week_log[-100:]

    plt.subplot(411)
    plt.plot(ts_euro_week_log_select.index.to_pydatetime(), ts_euro_week_log_select.values, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(ts_euro_week_log_select.index.to_pydatetime(), trend[-100:].values, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(ts_euro_week_log_select.index.to_pydatetime(), seasonal[-100:].values, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(ts_euro_week_log_select.index.to_pydatetime(), residual[-100:].values, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()

    # Now that you have stationarized your time series, you could go on and model residuals (fit lines between values
    # in the plot). However, as the patterns for the trend and seasonality information extracted from the series
    # that are plotted after decomposition are still not consistent and cannot be scaled back to the original values,
    # you cannot use this approach to create reliable forecasts.

    # 4.3 Remove trend and seasonality with differencing¶
    # Differencing is one of the most common methods of dealing with both trend and seasonality. In first order differencing,
    # you compute the differences between consecutive observations in the time series. This usually improves the
    # stationarity of the time series. In the code, this is confirmed by running the Dickey-Fuller test.

    ts_euro_week_log_diff = ts_euro_week_log - ts_euro_week_log.shift()
    plt.plot(ts_euro_week_log_diff.index.to_pydatetime(), ts_euro_week_log_diff.values)

    ts_euro_week_log_diff.dropna(inplace=True)
    check_stationarity(ts_euro_week_log_diff)

    # The results show that the test statistic is significantly less than the 1% critical value.
    #
    # This shows you that your time series is now stationary with 99% confidence. Now you can begin
    # to apply statistical models like ARIMA to forecast future Euro exchange rates using this stationarized time series.


    # 5. Find optimal parameters and build an ARIMA model
    # To apply an ARIMA model to your time series, you need to find optimal values for the
    # following three model parameters (p,d,q):
    #
    # The number of autoregressive (AR) terms (p): AR terms are just lags of the dependent variable,
    # the euro rate, in this case. So, if p=2, it means that predictors of x(t) will be x(t-1) and x(t-2).
    # The number of moving average (MA) terms (q): MA terms are lagged forecast errors in the prediction
    # equation. For instance, if q=2, the predictors for x(t) will be e(t-1) and e(t-2) where e(i) is the difference
    # between the moving average at i-th instant and the actual value.
    # The number of differences (d): These are the number of non-seasonal differences. In your case,
    # d=1, as you are modeling using the first order differenced time series.


    # 5.1 Plot the autocorrelation function (ACF) and partial autocorrelation function (PACF)¶
    # Run the next cell to plot the ACF and PACF, and determine the p, d and q
    # model parameters which you will need later as input for the ARIMA model:

    # ACF and PACF plots

    lag_auto_corr = acf(ts_euro_week_log_diff, nlags=10)
    lag_par_auto_corr = pacf(ts_euro_week_log_diff, nlags=10, method='ols')

    # Plot ACF:
    plt.subplot(121)
    plt.plot(lag_auto_corr)
    plt.axhline(y=0, linestyle='--', color='black')
    plt.axhline(y=-1.96 / np.sqrt(len(ts_euro_week_log_diff)), linestyle='--', color='black')
    plt.axhline(y=1.96 / np.sqrt(len(ts_euro_week_log_diff)), linestyle='--', color='black')
    plt.title('Autocorrelation Function')

    # Plot PACF:
    plt.subplot(122)
    plt.plot(lag_par_auto_corr)
    plt.axhline(y=0, linestyle='--', color='black')
    plt.axhline(y=-1.96 / np.sqrt(len(ts_euro_week_log_diff)), linestyle='--', color='black')
    plt.axhline(y=1.96 / np.sqrt(len(ts_euro_week_log_diff)), linestyle='--', color='black')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()



    # In this plot, the 'p' and 'q' values can be determined as follows:
    #
    # p: The lag value where the PACF cuts off (drops to 0) for the first time. If you look closely, p=2.
    # q: The lag value where the ACF chart crosses the upper confidence interval for the first time. If you look closely, q=2.
    # This means that the optimal values for the ARIMA(p,d,q) model are (2,1,2).
    #
    # If your assessment of the ACF and PACF plots differs from the values suggested by the arma_order_select_ic
    # function, you should plug in different values for the p and q terms and use the model fit results to study the
    # AIC values and proceed with the model with a lower AIC value
    #
    # Run the next code cell to plot the ARIMA model using the values (2,1,2):

    model = ARIMA(ts_euro_week_log, order=(2, 1, 1))
    results_ARIMA = model.fit(disp=-1)
    plt.plot(ts_euro_week_log_diff.index.to_pydatetime(), ts_euro_week_log_diff.values)
    plt.plot(ts_euro_week_log_diff.index.to_pydatetime(), results_ARIMA.fittedvalues, color='red')
    plt.title('RSS: %.4f' % sum((results_ARIMA.fittedvalues - ts_euro_week_log_diff) ** 2))



    # 5.2 Measure the variance between the data and the values predicted by the model
    # You can measure whether the results of your model fit the underlying data by using the residual
    # sum of squares (RSS) metric. A small RSS indicates that the model fits tightly to the data.
    #
    # Yet another approach to validate the ARIMA model appropriateness is by performing residual analysis.
    #
    # Print the results of the ARIMA model and plot the residuals. A density plot of the residual error values
    # indicates a normal distribution centered around zero mean. Also, the residuals do not violate the assumptions of
    # constant location and scale with most values in the range (-1,1).



print(results_ARIMA.summary())
# plot residual errors
residuals = DataFrame(results_ARIMA.resid)
residuals.plot(kind='kde')
print(residuals.describe())


# 5.3 Scale predictions¶
# Now that the model is returning the results you want to see, you
# can scale the model predictions back to the original scale. For this, you will remove the first order
# differencing and take exponent to restore the predictions back to their original scale.
#
# The lower the root mean square error (RMSE) and the closer it is to 0, the better are the model
# predictions in being closer to actual values.

euro_predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print (euro_predictions_ARIMA_diff.head())



euro_predictions_ARIMA_diff_cumsum = euro_predictions_ARIMA_diff.cumsum()
euro_predictions_ARIMA_log = pd.Series(ts_euro_week_log.iloc[0], index=ts_euro_week_log.index)
euro_predictions_ARIMA_log = euro_predictions_ARIMA_log.add(euro_predictions_ARIMA_diff_cumsum,fill_value=0)


euro_predictions_ARIMA = np.exp(euro_predictions_ARIMA_log)
plt.plot(ts_euro_week.index.to_pydatetime(), ts_euro_week.values)
plt.plot(ts_euro_week.index.to_pydatetime(), euro_predictions_ARIMA.values)
plt.title('RMSE: %.4f'% np.sqrt(sum((euro_predictions_ARIMA-ts_euro_week)**2)/len(ts_euro_week)))

# 6. Perform and visualize time series forecasting
# What you have achieved in this notebook so far is in-sample forecasting using ARIMA as you trained
# the model on the entire time series data. Now you need to split the data set into a training and
# testing data sets. You will use the training data set to train the ARIMA model and perform out-of-sample
# forecasting. Then you will compare the results of your out-of-sample predictions for Euro rates with the
# actual values from the test data set.
#
# You will use the forecast function forecast and perform a rolling one-step forecast with ARIMA. A rolling
# forecast is required given the dependence on observations during differencing and the AR model.
# You will re-create the ARIMA model after each new prediction is received. And you will manually keep
# track of all observations in a list called history, which is seeded with the training data and to which
# new predictions are appended in each iteration.



size = int(len(ts_euro_week_log) - 15)
train, test = ts_euro_week_log[0:size], ts_euro_week_log[size:len(ts_euro_week_log)]
historical = [x for x in train]
predictions = list()

print('Printing Predicted vs Expected Values...')
print('\n')
for t in range(len(test)):
    model = ARIMA(historical, order=(2,1,1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(float(yhat))
    observed = test[t]
    historical.append(observed)
    print('Predicted Euro Rate = %f, Expected Euro Rate = %f' % (np.exp(yhat), np.exp(observed)))

error = mean_squared_error(test, predictions)

print('\n')
print('Printing Mean Squared Error of Predictions...')
print('Test MSE: %.6f' % error)

euro_predictions_series = pd.Series(predictions, index = test.index)

# You validated the model by comparing its out-of-sample predictions for Euro rates with actual values
# from the test data set and calculating the mean squared error. Now plot the rolling forecast
# predictions against the observed values. You will see that the predictions are in the correct
# scale and are picking up the trend in the original time series.



fig, ax = plt.subplots()
ax.set(title='Spot Exchange Rate, Euro into USD', xlabel='Date', ylabel='Euro into USD')
ax.plot(ts_euro_week[-50:], 'o', label='observed')
ax.plot(np.exp(euro_predictions_series), 'g', label='rolling one-step out-of-sample forecast')
legend = ax.legend(loc='upper left')
legend.get_frame().set_facecolor('w')





