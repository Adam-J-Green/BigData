import os
import pandas as pd
import yfinance as yf
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup as bs
import requests
from pyspark.sql.functions import sum,max,min,mean,count
import datetime as dt
from os.path import abspath
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.model_selection import ParameterGrid
import yfinance as yf
import datetime as dt
from keras.saving.legacy.save import load_model
from sklearn.preprocessing import MinMaxScaler
import dateparser
#function to retrieve headlines for desired period
headers = {'User-Agent':
	'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:108.0) Gecko/20100101 Firefox/108.0'}
def gather_headlines(company_name, ticker):
    headlines = []
    dates = []
    for i in range(10, 300, 10):    # Running for-loop
        info_url = "https://financialpost.com/search/?search_text="+company_name +"&date_range=-365d&sort=desc&from="+str(i)
        page = requests.get(info_url, headers = headers)
        parser = bs(page.content, "html.parser" )
        date = parser.body.find_all('div', attrs={'class': 'article-card__meta-bottom'})
        for span in date:
            dates.append(span.text.split("   ")[1])
        headline = parser.body.find_all('h3', class_ = 'article-card__headline text-size--extra-large--sm-up')
        for x in headline:
            headlines.append(x.text)
    dates = dates[:len(headlines)]
    file = {'date' : dates, "headline" : headlines}
    file = pd.DataFrame(file)
    file['ticker'] = ticker
    return file

#calculate sentiment scores for each headlines and append to dataset
def analyze_sent(company, ticker):
    data = gather_headlines(company, ticker)
    dates = []
    for index, row in data.iterrows():
        date = dateparser.parse(row['date'], date_formats = ["%d-%m-%y"])
        dates.append(date.date())
    data['date'] = dates
    analyze_obj = SentimentIntensityAnalyzer()
    data['sentiment']=data['headline'].apply(lambda headline: analyze_obj.polarity_scores(str(headline))['compound'])
    data.fillna(0, inplace = True)
    headline_count = data.groupby(by = ['date'])['headline'].count()
    mean_sent = data.groupby(by = ['date'])['sentiment'].mean()
    df = {'sentiment' : mean_sent, 'headline_count':headline_count}
    df = pd.DataFrame(df)
    df["score"] = (df['sentiment']*(df['headline_count']**2))
    df.drop(['sentiment', 'headline_count'], axis = 1, inplace = True)
    print(df)
    return df


def prophet_data(ticker, start_date):
    delta = dt.timedelta(days = 400)
    data = yf.download(ticker, (start_date - delta)).reset_index()
    data = data.rename(columns = {'Date':'ds', 'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Adj Close': 'y', 'Volume':'volume'})
    data = data.loc[:,['ds', 'y']]
    return data

def get_data(ticker, start_day):
  Best_parameters={'changepoint_prior_scale': 0.1, 'seasonality_mode': 'multiplicative', 'seasonality_prior_scale': 1.0}
  today = dt.date.today()
  final_model = Prophet(**Best_parameters)
  final_model.fit(prophet_data(ticker, start_day))
  df = cross_validation(model=final_model, initial='380 days', horizon='100 days', period='100 days')
  df = df.set_index('ds').loc[str((today - dt.timedelta(days = 30))):today,]
  print(df)
  return df


def get_financials(ticker, start):
    time_delt = dt.timedelta(days = 300)
    start_day = start - time_delt
    data = yf.download(str(ticker), start_day)
    data['ticker'] = ticker
    data = data.reset_index()
    data = data.rename(columns = {'Date':'date', 'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Adj Close': 'adj_close', 'Volume':'volume'})
    print(data)
    print('success!')
    return data
                       
                       
def EWMA(data, ndays): 
    EMA = pd.Series(data['close'].ewm(span = ndays, min_periods = ndays - 1).mean(), 
                 name = 'EWMA_' + str(ndays)) 
    data = data.join(EMA) 
    return data

def rsi(close, periods = 14):
    
    close_delta = close.diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()

    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi

def BBANDS(data, window):
    MA = data.close.rolling(window).mean()
    SD = data.close.rolling(window).std()
    data['MiddleBand'] = MA
    data['UpperBand'] = MA + (2 * SD) 
    data['LowerBand'] = MA - (2 * SD)
    return data

def prep_financials(df):
    df = pd.DataFrame(df)
    #df.set_index('date')
    df['target'] = (df['close'])
    df['tenmda'] = df['close'].rolling(10).mean()
    df['twentymda'] = df['close'].rolling(20).mean()
    df['fiftymda'] = df['close'].rolling(50).mean()
    df['hundredmda'] = df['close'].rolling(100).mean()
    df = EWMA(df, 20)
    df = EWMA(df, 50) 
    df = EWMA(df, 100)
    df['rsi'] = rsi(df['close'])
    df = BBANDS(df, 40)
    df.dropna(inplace = True)
    df = df.iloc[-20:, ]
    df.reset_index()
    return df

def lstm_split(data,target,steps):
    X = []
    y = []
      # Creating a data structure with 10 time-steps and 1 output
    for i in range(10, steps):
        X.append(data[i-10:i])
        y.append(target[i])  
    return np.array(X),np.array(y)


#User Inputs
def set_inputs():
    company_name = input('please enter the target company name: ').lower()
    target_company = input('Please input the ticker of your target stock: ').upper()
    backtest = input("Would you like to predict tomorrow's share price or backtest model accuracy? Type 'predict' to predict or 'backtest' to backtest: ")
    if backtest == 'backtest':
        pass
        #start_date = 
        #stop_date = 
    if backtest == 'predict':
        delt = dt.timedelta(days = 20)
        today = dt.date.today()
        period = (today - delt)
        financials = prep_financials(get_financials(target_company, period)).set_index('date')
        prophet_df= get_data(target_company, dt.date(2017,1,1))
        sentiment = analyze_sent(company_name, target_company).reset_index().rename(columns = {'ds':'date'}).set_index('date')
        training_data = financials.join(sentiment, how = 'left')
        training_data = training_data.join(prophet_df['yhat'])
        print(training_data)
        training_data.fillna(0.5, inplace = True)
    else:
        print('Error in input, please try again and select either backtest or predict')
    return training_data, prophet_df

def data_prep1(training_data):
    target1 = training_data['target']
    training_data.drop(['target', 'ticker'], axis = 1, inplace = True)
    target_scaler = MinMaxScaler()
    feature_scaler = MinMaxScaler()
    target = target_scaler.fit_transform([target1]).flatten()
    for col in training_data.columns:
        training_data[col] = feature_scaler.fit_transform(training_data[[col]])
    prophet = np.array(training_data['yhat'])
    training_data.drop(['yhat'], axis = 1, inplace = True)
    x, y = lstm_split(training_data, target, len(training_data))
    return x, y, target_scaler, prophet, target1

def hybrid_trainer(training_data):
    preds_dict = {}
    path = 'sub_models/'
    x_train, y_train, targ_scaler, prophet, target1 = data_prep1(training_data)
    prophet = prophet[10:]
    for num, model in enumerate(os.listdir(path)):
        model = load_model(path+model)
        prediction = model.predict(x_train).flatten()
        preds_dict[model] = prediction
    preds_df = pd.DataFrame(preds_dict)
    preds_df['prophet'] = prophet
    preds_df = np.array(preds_df)
    ex_predict = preds_df.reshape((1,10,5))
    return ex_predict, targ_scaler, target1

def final_prediction(prediction_data, target_scaler, y_train):
    model = load_model('main_models/hybrid_expanded.h5')
    prediction = model.predict(prediction_data)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(np.array(y_train).reshape(-1,1))
    final = scaler.inverse_transform(prediction)
    print(f"The prediction of tomorrows stock close price for your target company is {final[0]}")


def main():
    train_data, prophet_data = set_inputs()
    preds_df, scaler, y_train = hybrid_trainer(train_data)
    final_prediction(preds_df, scaler, y_train)
main()