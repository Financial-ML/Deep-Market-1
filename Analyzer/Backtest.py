from Features import *
import fxcmpy
import pandas as pd
import numpy as np
from joblib import load
from keras.models import model_from_yaml
import os
import sys
#------------------------------------------------------------------------------

masterFrameCleaned = pd.read_csv('calculated1.csv')
Distance = 20
exchang_rate = 110.00
#------------------------------------------------------------------------------
#load the models
columns = ['momentum3close','momentum4close'
            ,'momentum5close','momentum8close','momentum9close','momentum10close'
            ,'stoch3K','stoch3D','stoch4K','stoch4D'
            ,'stoch5K','stoch5D','stoch8K','stoch8D'
            ,'stoch9K','stoch9D','stoch10K'
            ,'stoch10D','will6R','will7R','will8R'
            ,'will9R','will10R','proc12close','proc13close'
            ,'proc14close','proc15close','wadl15close','adosc2AD'
            ,'adosc3AD','adosc4AD','adosc5AD','macd1530','cci15close'
            ,'bollinger15upper','bollinger15mid','bollinger15lower','paverage2open'
            ,'paverage2high','paverage2low','paverage2close','slope3high','slope4high','slope5high'
            ,'slope10high','slope20high','slope30high'
            ,'fourier10a0','fourier10a1','fourier10b1','fourier10w','fourier20a0'
            ,'fourier20a1','fourier20b1','fourier20w','fourier30a0'
            ,'fourier30a1','fourier30b1','fourier30w','sine5a0','sine5b1','sine5w'
            ,'sine6a0','sine6b1','sine6w','open','high','low','close']


# load YAML and create model
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

df = masterFrameCleaned[list(columns)].values

#------------------------------------------------------------------------------    
i = 0
sum_profit = 0
sum_profit_current = 0
count = 0
win = 0
lose = 0
price_list_sell = []#for max draw all
price_list_buy = []#for max draw all
price_list_sell_close = []#for max draw all
price_list_buy_close = []#for max draw all
sum_draw_down_profit_max = 0#for max draw all
min_account_balance_max = 0
while i < len(masterFrameCleaned):
    a=[]
    a1 = []
    profit = 0
    newfeatures = df[i].reshape(1, 69) 

    predicted_rf = loaded_model.predict_classes(newfeatures)
    pro_rf = loaded_model.predict_proba(newfeatures)
    a1 = np.append(a1,pro_rf) 
    a = np.append(a,predicted_rf)
#------------------------------------------------------------------------------ 
    if i < len(masterFrameCleaned)-Distance:
        print('Propa :',a1[0])
        print('Prediction :',a[0])
        if a1[0] > 0.75 or a1[0] < 0.25:
            count = count +1
            if a[0]==1:
                profit = ((masterFrameCleaned.close[i + Distance] - masterFrameCleaned.close[i] - 0.04)*1000)/exchang_rate
                price_list_buy_close.append(profit)#for max draw all
                price_list_buy.append(masterFrameCleaned.close[i])#for max draw all
                #max down for 1 order
            elif a[0]==0:
                profit = ((masterFrameCleaned.close[i] - masterFrameCleaned.close[i + Distance] - 0.04)*1000)/exchang_rate
                price_list_sell_close.append(profit)#for max draw all
                price_list_sell.append(masterFrameCleaned.close[i])#for max draw all
            if profit > 0:
                win = win + 1
            else:
                lose = lose + 1                
    #max down for all order
    #----------------
    #sell
    pop_list = []
    for y in range(len(price_list_sell)):
        draw_down_profit = 0
        draw_down_profit = ((price_list_sell[y] - masterFrameCleaned.close[i] - 0.04)*1000)/exchang_rate
        if draw_down_profit == price_list_sell_close[y]:
            sum_profit_current = sum_profit_current + draw_down_profit
            pop_list.append(y)
    c=0
    for y in range(len(pop_list)):
        price_list_sell.pop(pop_list[y]-c)
        price_list_sell_close.pop(pop_list[y]-c)
        c=c+1
    #buy
    pop_list = []
    for y in range(len(price_list_buy)):
        draw_down_profit = 0
        draw_down_profit = ((masterFrameCleaned.close[i] - price_list_buy[y] - 0.04)*1000)/exchang_rate
        if draw_down_profit == price_list_buy_close[y]:
            sum_profit_current = sum_profit_current + draw_down_profit
            pop_list.append(y)
    c=0
    for y in range(len(pop_list)):
        price_list_buy.pop(pop_list[y]-c)
        price_list_buy_close.pop(pop_list[y]-c)
        c=c+1
    #all
    sum_draw_down_profit_sell = 0
    for y in range(len(price_list_sell)):
        draw_down_profit = 0
        draw_down_profit = ((price_list_sell[y] - masterFrameCleaned.close[i] - 0.04)*1000)/exchang_rate
        sum_draw_down_profit_sell = sum_draw_down_profit_sell + draw_down_profit
    sum_draw_down_profit_buy = 0
    for y in range(len(price_list_buy)):
        draw_down_profit = 0
        draw_down_profit = ((masterFrameCleaned.close[i] - price_list_buy[y] - 0.04)*1000)/exchang_rate
        sum_draw_down_profit_buy = sum_draw_down_profit_buy + draw_down_profit
    if(sum_draw_down_profit_sell + sum_draw_down_profit_buy < sum_draw_down_profit_max):
        sum_draw_down_profit_max = sum_draw_down_profit_sell + sum_draw_down_profit_buy
    #----------------
    sum_profit = sum_profit + profit
    min_account_balance = sum_profit_current + sum_draw_down_profit_sell + sum_draw_down_profit_buy
    if (min_account_balance < min_account_balance_max):
        min_account_balance_max = min_account_balance
    i = i + 1
print('profit of 1 lots:',sum_profit)
print('Total number of trades:',count)
print('Total number of wining trades:',win)
print('Total number of lose trades:',lose)
print('Max draw Dawn:',sum_draw_down_profit_max)
print('min account balance:',min_account_balance_max)
#------------------------------------------------------------------------------
