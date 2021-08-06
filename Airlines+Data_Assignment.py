# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 13:07:43 2021

@author: ASUS
"""

# Importing the necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.tsa.holtwinters import Holt
from pandas.tseries.offsets import DateOffset
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

# Loading the dataset
file=pd.read_excel("D:\Data Science Assignments\Python-Assignment\Forecasting\Airlines+Data.xlsx")

# Exploring the dataset
file.columns
file.head()
file["Months"]=file["Month"].dt.month_name()
file["Months1"]=file["Month"].dt.month_name()
file["Years"]=file["Month"].dt.year
file=pd.get_dummies(file,columns=["Months1"],prefix="")
file['t']=np.arange(1,97)
file['t_square']=file['t']*file['t']
file['log_Pass']=np.log(file["Passengers"])

# Data Visualisation
sns.boxplot(x="Years",y="Passengers",data=file)
sns.lineplot(x="Years",y="Passengers",data=file)
sns.boxplot(x="Months",y="Passengers",data=file)
sns.lineplot(x="Months",y="Passengers",data=file,sort=False)
sns.violinplot(x="Years",y="Passengers",data=file) 
sns.violinplot(x="Months",y="Passengers",data=file) 
heatmap=pd.pivot_table(data=file,values="Passengers",columns="Months",index="Years",aggfunc="mean",fill_value=0)  
sns.heatmap(heatmap,annot=True,fmt="d")
# Rolling Average Plot
file["Passengers"].plot(label="Original")
for i in range(1,24,4):
    file["Passengers"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)
# Autocorrelation Plot
autocorrelation_plot(file["Passengers"])

# Splitting dataframe into training and testing dataset
train=file.iloc[0:84,]
test=file.iloc[84:,]

# RMSE Function
def rmse(predict):
    return (np.sqrt(np.mean((np.array(test["Passengers"])-np.array(predict))**2)))

# MAPE Function
def mape(predict):
    temp=np.abs((predict-test["Passengers"])/test["Passengers"])*100
    return np.mean(temp)

# Augmented Dickey-Fuller Test
def adfull(dataset):
    dftest=adfuller(dataset)
    dfoutput=pd.Series(dftest[0:4],
                       index=['Test Statistic',
                              'P-Value',
                              'Lags Used',
                              'Number of Observation Used'])
    return dfoutput    

##################################################################################
# Model Driven Approaches

# Linear Model
lin_mod=smf.ols("Passengers~t",data=train).fit()
pred_lin=pd.Series(lin_mod.predict(test))
rmse_lin=rmse(pred_lin)

# Exponential Model
exponent_mod=smf.ols("log_Pass~t",data=train).fit()
pred_exponent=pd.Series(np.exp(exponent_mod.predict(test)))
rmse_exponent=rmse(pred_exponent)

# Quadratic Model
qua_mod=smf.ols("Passengers~t+t_square",data=train).fit()
pred_qua=pd.Series(qua_mod.predict(test))
rmse_qua=rmse(pred_qua)

# Additive Seasonality
adse_mod=smf.ols("Passengers~_January+_February+_March+_April+_May+_June+_July+_August+_September+_October+_November",data=train).fit()
pred_adse=pd.Series(adse_mod.predict(test))
rmse_adse=rmse(pred_adse)

# Additive Seasonality with Quadratic Trend
adsequ_mod=smf.ols("Passengers~t+t_square+_January+_February+_March+_April+_May+_June+_July+_August+_September+_October+_November",data=train).fit()
pred_adsequ=pd.Series(adsequ_mod.predict(test))
rmse_adsequ=rmse(pred_adsequ)

# Multiplicative Model
mul_mod=smf.ols("log_Pass~_January+_February+_March+_April+_May+_June+_July+_August+_September+_October+_November",data=train).fit()
pred_mul=pd.Series(np.exp(mul_mod.predict(test)))
rmse_mul=rmse(pred_mul)

#Multiplicative Model with Additive Seasonality
mulad_mod=smf.ols("log_Pass~t+_January+_February+_March+_April+_May+_June+_July+_August+_September+_October+_November",data=train).fit()
pred_mulad=pd.Series(np.exp(mulad_mod.predict(test)))
rmse_mulad=rmse(pred_mulad)
pred_mulad

# Creating a table of RMSE Values
RMSE_table=pd.DataFrame({'Model_Names':["Linear Model","Exponential Model",
                                        "Quadratic Model","Additive Seasonality",
                                        "Additive Seasonality with Quadratic Trend",
                                        "Multiplicative Model",
                                        "Multiplicative Model with Additive Seasonality"],
                         'RMSE_Values':[rmse_lin,rmse_exponent,rmse_qua,rmse_adse,
                                        rmse_adsequ,rmse_mul,rmse_mulad]})
RMSE_table #Analysing the model accuracies

# Lowest RMSE Value is for Multiplicative Model with Additive Seasonality 
# Applying Multiplicative Model with Additive Seasonality on entire dataset
mu_ad_mod=smf.ols("log_Pass~t+_January+_February+_March+_April+_May+_June+_July+_August+_September+_October+_November",data=file).fit()
pred_mu_ad=pd.Series(np.exp(mu_ad_mod.predict(test)))
rmse_mu_ad=rmse(pred_mu_ad)
# Prediction Dataset Visualisation
plt.xticks(rotation=40)
sns.lineplot(x="Months",y=pred_mu_ad,data=test,sort=False)
sns.lineplot(x="Months",y="Passengers",data=test,sort=False)
plt.legend(['Prediction','Original'])

#Also Trying with Additive Seasonality with Quadratic Trend 
adsequ=smf.ols("Passengers~t+t_square+_January+_February+_March+_April+_May+_June+_July+_August+_September+_October+_November",data=file).fit()
pred_adsequad=pd.Series(adsequ.predict(test))
rmse_adsequad=rmse(pred_adsequad)
# Prediction Dataset Visualisation
plt.xticks(rotation=40)
sns.lineplot(x="Months",y=pred_adsequad,data=test,sort=False)
sns.lineplot(x="Months",y="Passengers",data=test,sort=False)
plt.legend(['Prediction','Original'])

################################################################################
# Data Driven Approaches

# Time-Series Decomposition plot
decompose_add=seasonal_decompose(file.Passengers,model="additive",freq=12)
decompose_add.plot(); # For Additive Type Model
decompose_mul=seasonal_decompose(file.Passengers,model="multiplicative",freq=12)
decompose_mul.plot(); # For Multiplicative Type Model

# Plotting ACF and PACF plot
plot_acf(file.Passengers,lags=10); #ACF Plot
plot_pacf(file.Passengers,lags=10);

# Models
#Simple Exponential Smoothing Model
ses_mod=SimpleExpSmoothing(train["Passengers"]).fit()
pred_ses=ses_mod.predict(start=test.index[0],end=test.index[-1])
mape_ses=mape(pred_ses)

#Holt Model
hol_mod=Holt(train["Passengers"]).fit()
pred_hol=hol_mod.predict(start=test.index[0],end=test.index[-1])
mape_hol=mape(pred_hol)

#Holt-Winter Exponential Smoothing Method with Additive Seasonality and Additive Trend
hol_ad_mod=ExponentialSmoothing(train["Passengers"],seasonal="additive",trend="additive",seasonal_periods=12).fit()
pred_hol_ad_mod=hol_ad_mod.predict(start=test.index[0],end=test.index[-1])
mape_hol_ad=mape(pred_hol_ad_mod)

#Holt-Winter Exponential Smoothing Method with Multiplicative Seasonality and Additive Trend
hol_mul_ad_mod=ExponentialSmoothing(train["Passengers"],seasonal="multiplicative",trend="additive",seasonal_periods=12).fit()
pred_hol_mul_ad_mod=hol_mul_ad_mod.predict(start=test.index[0],end=test.index[-1])
mape_hol_mul_ad=mape(pred_hol_mul_ad_mod)

# Creating a table with MAPE Values
MAPE_table=pd.DataFrame({'Model_Names':["Simple Exponential Smoothing Model","Holt Model",
                                        "Holt-Winter Exponential Smoothing Method with Additive Seasonality and Additive Trend",
                                        "Holt-Winter Exponential Smoothing Method with Multiplicative Seasonality and Additive Trend"],
                         'MAPE_Values': [mape_ses,mape_hol,mape_hol_ad,mape_hol_mul_ad]})
MAPE_table

# Lowest MAPE Value is for Holt-Winter Exponential Smoothing Method with Additive Seasonality and Additive Trend
# Applying Holt-Winter Exponential Smoothing Method with Additive Seasonality and Additive Trendy on entire dataset
#Holt-Winter Exponential Smoothing Method with Additive Seasonality and Additive Trend
final_hol_ad_mod=ExponentialSmoothing(file["Passengers"],seasonal="additive",trend="additive",seasonal_periods=12).fit()
final_pred_hol_ad_mod=final_hol_ad_mod.predict(start=test.index[0],end=test.index[-1])
final_mape_hol_ad=mape(final_pred_hol_ad_mod)

##########################################################################################
# ARIMA and SARIMAX

#Augmented Dickey-Fuller Test
adftest=adfull(file["Passengers"])
adftest #P-Value=0.99

#First Seasonal Differencing
file["First Seasonal Difference"]=file["Passengers"]-file["Passengers"].shift(12)
adftest1=adfull(file["First Seasonal Difference"].dropna())
adftest1

#Second Seasonal Differencing
file["Second Seasonal Difference"]=file["Passengers"]-file["Passengers"].shift(24)
adftest2=adfull(file["Second Seasonal Difference"].dropna())
adftest2

#Third Seasonal Differencing
file["Third Seasonal Difference"]=file["Passengers"]-file["Passengers"].shift(36)
adftest3=adfull(file["Third Seasonal Difference"].dropna())
adftest3

# Plotting ACF and PACF plot
fig=plt.figure(figsize=(14,8))
ax1=fig.add_subplot(211)
fig=plot_acf(file["Third Seasonal Difference"].dropna(),lags=20,ax=ax1)
ax2=fig.add_subplot(212)
fig=plot_pacf(file["Third Seasonal Difference"].dropna(),lags=20,ax=ax2)

# As the data is seasonal, we will be choosing SARIMAX over ARIMA model
smx_mod=sm.tsa.statespace.SARIMAX(file["Passengers"],order=(3,2,3),seasonal_order=(4,1,3,12)).fit()
file["Forecast"]=smx_mod.predict(start=84,end=96,dynamic=True)
file[["Passengers","Forecast"]].plot(figsize=(12,8))

# Creating a new Dataframe for forecasting
file.set_index('Month',inplace=True)
future=[file.index[-1]+ DateOffset(months=x)for x in range(1,25)]
future_dataset=pd.DataFrame(index=future,columns=file.columns)
future_dataset["Months"]=future_dataset.index
future_dataset["Years"]=future_dataset["Months"].dt.year
future_dataset["Months"]=future_dataset["Months"].dt.month_name()

# Forecasting the future values
future_dataframe=pd.concat([file,future_dataset])
future_dataframe["Forecast"]=smx_mod.predict(start=94,end=120,dynamic=True)
future_dataframe[["Passengers","Forecast"]].plot(figsize=(12,8))
plt.xlabel("Years")
plt.ylabel("Passengers")