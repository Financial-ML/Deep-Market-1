from Features import *
import fxcmpy
import pandas as pd
import datetime as dt
import pickle
#------------------------------------------------------------------------------
#defult data
Sympol = 'AUD/JPY'
Period = 'D1'
start_traning = dt.datetime(2003, 1, 1) 
stop_traning = dt.datetime(2017, 1, 1) 
stop_testing = dt.datetime(2017, 6, 30) 
Distance = 20 # MUST BE DISTANCE > 0  
#------------------------------------------------------------------------------
#Get Data From Market

TOKEN = "fea322e8370d263a0556673b882dfdf8d9065699"
        

con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error')

if con.is_connected() == True:
    print("Data retrieved...")
    print(' ')
    df = con.get_candles(Sympol, period = Period,start = start_traning, stop = stop_testing)
    df = df.drop(columns=['bidopen', 'bidclose','bidhigh','bidlow'])
            
    df = df.rename(columns={"askopen": "open", "askhigh": "high","asklow": "low", "askclose": "close","tickqty": "volume"})
    df = df[['open','high','low','close','volume']]
    df = df[~df.index.duplicated()]
    prices = df.copy()
else:
    print('No connection with fxcm')
    print(' ')

prices = prices.drop_duplicates(keep=False)

con.close()
#------------------------------------------------------------------------------
#Process Data

momentumKey = [3,4,5,8,9,10] 
stochasticKey = [3,4,5,8,9,10] 
williamsKey = [6,7,8,9,10] 
procKey = [12,13,14,15] 
wadlKey = [15] 
adoscKey = [2,3,4,5] 
macdKey = [15,30] 
cciKey = [15] 
bollingerKey = [15] 
paverageKey = [2] 
slopeKey = [3,4,5,10,20,30] 
fourierKey = [10,20,30] 
sineKey = [5,6] 
marketKey = [0]


keylist = [momentumKey,stochasticKey,williamsKey,procKey,wadlKey,adoscKey,macdKey,cciKey,bollingerKey
            ,paverageKey,slopeKey,fourierKey,sineKey,marketKey] 

momentumDict = momentum(prices,momentumKey) 
stochasticDict = stochastic(prices,stochasticKey) 
williamsDict = williams(prices,williamsKey)
procDict = proc(prices,procKey) 
wadlDict = wadl(prices,wadlKey) 
adoscDict = adosc(prices,adoscKey)
macdDict = macd(prices,macdKey) 
cciDict = cci(prices,cciKey) 
bollingerDict = bollinger(prices,bollingerKey,2) 
paverageDict = pavarage(prices,paverageKey) 
slopeDict = slopes(prices,slopeKey) 
fourierDict = fourier(prices,fourierKey) 
sineDict = sine(prices,sineKey) 
marketDict = Market(prices,marketKey,Distance) 

# Create list of dictionaries 

dictlist = [momentumDict.close,stochasticDict.close,williamsDict.close
            ,procDict.proc,wadlDict.wadl,adoscDict.AD,macdDict.line
            ,cciDict.cci,bollingerDict.bands,paverageDict.avs
            ,slopeDict.slope,fourierDict.coeffs,sineDict.coeffs,marketDict.slope] 

# list of column name on csv

colFeat = ['momentum','stoch','will','proc','wadl','adosc','macd',
            'cci','bollinger','paverage','slope','fourier','sine','market']

masterFrame = pd.DataFrame(index = prices.index) 
for i in range(0,len(dictlist)): 
    if colFeat[i] == 'macd':
        colID = colFeat[i] + str(keylist[6][0]) + str(keylist[6][1]) 
        masterFrame[colID] = dictlist[i] 
    else: 
        for j in keylist[i]: 
            for k in list(dictlist[i][j]):
                colID = colFeat[i] + str(j) + str(k)
                masterFrame[colID] = dictlist[i][j][k]
                    
threshold = round(0.7*len(masterFrame)) 
masterFrame[['open','high','low','close']] = prices[['open','high','low','close']]

masterFrameCleaned = masterFrame.copy() 
masterFrameCleaned = masterFrameCleaned.dropna(axis=1,thresh=threshold)
masterFrameCleaned = masterFrameCleaned.dropna(axis=0)



#------------------------------------------------------------------------------
#for traning
s = 0 
masterFrameCleaned['date']=pd.to_datetime(masterFrameCleaned.index)
for i in range(0,len(masterFrameCleaned)):
    if masterFrameCleaned.date.iloc[i] >= stop_traning:
        s = i
        break
masterFrameCleaned = masterFrameCleaned.drop(['date'], axis=1)

train = masterFrameCleaned.iloc[:s]

train.to_csv('calculated.csv')

#for backtesting
test = masterFrameCleaned.iloc[s:]
test.to_csv('calculated1.csv')
#------------------------------------------------------------------------------    

print('Complete procrss the features...')
print(' ')
#------------------------------------------------------------------------------