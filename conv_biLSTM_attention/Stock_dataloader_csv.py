import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import os
from sklearn import preprocessing


class stock_csv_read:
    def __init__(self, data, x_frames, y_frames):
        self.data = data
        self.x_frames = x_frames
        self.y_frames = y_frames
        self.stock_data = self.data_loader()
        
    def data_loader(self):
        stock_data = pd.read_csv("C:/Users/lab/Desktop/conv_biLSTM_attention/data/kdd17/price_long_50/"+ self.data ,header=0,index_col="Date")

        open = stock_data.loc[:,"Open"]
        high = stock_data.loc[:,"High"]
        low = stock_data.loc[:,"Low"]
        close = stock_data.loc[:,"Adj Close"]
        volume = stock_data.loc[:,"Volume"]

        upndown = (stock_data.loc[:, "Adj Close"] - stock_data.loc[:, "Adj Close"].shift(periods=-1, axis=0))
        change = (((stock_data.loc[:, "Adj Close"] - stock_data.loc[:, "Adj Close"].shift(periods=-1, axis=0))/stock_data.loc[:, "Adj Close"].shift(periods=-1, axis=0))*100)
        tgt = np.where(stock_data.loc[:, "Adj Close"] >= stock_data.loc[:, "Adj Close"].shift(periods=-1, axis=0), 1.0, 0.0)

        df = np.column_stack((open, high, low, close, volume, upndown, change))



        scaler  = preprocessing.StandardScaler().fit(df)
        scaled_df = scaler.transform(df)
        data_len = len(tgt)
        data = np.column_stack((scaled_df,tgt))[:data_len-1]
        
        return pd.DataFrame(data)
    
    def spliter(self, data):
        self.dd = data
        data_list = []
        for i in range(len(self.dd)-self.x_frames-self.y_frames+1):
            xy = []
            X = self.dd.iloc[i : i +self.x_frames, 0:7]
            y = self.dd.iloc[i +self.x_frames : i +self.x_frames +self.y_frames, 7:]
            xy.append(X)
            xy.append(y)
            data_list.append(xy)
        return data_list
    
    def cv_split(self):
        stock_data = self.stock_data
        
        data_len = len(stock_data)
        mok = data_len//19
        
        adder = 0
        data_list = []
        for i in range(10):
            sp_data = stock_data.iloc[0+adder:10*mok+adder,:]
            
            tvt = []
            train_sp_data = sp_data[0:8*mok]
            validation_sp_data = sp_data[8*mok:9*mok]
            test_sp_data = sp_data[9*mok:10*mok]
            
            train_sp_data_ = self.spliter(train_sp_data)
            validation_sp_data_ = self.spliter(validation_sp_data)
            test_sp_data_ = self.spliter(test_sp_data)

            tvt.append(train_sp_data_)
            tvt.append(validation_sp_data_)
            tvt.append(test_sp_data_)
            adder += mok
            data_list.append(tvt)
            
        return data_list


