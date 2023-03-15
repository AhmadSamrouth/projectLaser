from sklearn.preprocessing import MinMaxScaler
import pickle
import pandas as pd
import pandas_ta as ta
import numpy as np
import tensorflow as tf

def BuyOrSell():
    modelloaded = tf.keras.models.load_model('bitcoinModel.h5')
    filename = 'scaler.pkl'
    with open(filename, 'rb') as f:
        sc = pickle.load(f)
    # close the file
    f.close()

    #scaler of Target Next Close
    filename = 'scaler_target_next_close.pkl'
    with open(filename, 'rb') as f:
        scaler_target_next_close = pickle.load(f)
    # close the file
    f.close()

    #load data
    data = pd.read_csv('C:\\Users\\ahmad\\downloads\\data.csv', header = None, names = ['Open','High','Low','Close','Volume'])

    #preprocess data

    # Adding indicators

    data['RSI']=ta.rsi(data.Close, length=14)
    data['EMAF']=ta.ema(data.Close, length=20)
    data['EMAM']=ta.ema(data.Close, length=20)
    data['EMAS']=ta.ema(data.Close, length=20)

    data['Target'] = data['Close']-data.Open
    data['Target'] = data['Target'].shift(-1)

    data = data.iloc[20:]

    data['TargetClass'] = [1 if data.iloc[i].Target>0 else 0 for i in range(len(data))]

    data['TargetNextClose'] = data['Close'].shift(-1)

    ##################################################################################################
    ##################################################################################################
    ##################################################################################################
    ##################################################################################################
    ############### TO BE ROMOVED AFTER GETTING A LARGER DATASET #####################################
    #data = pd.concat([data,data,data,data], axis = 0)

    data.dropna(inplace=True)
    data.reset_index(inplace = True,drop=True)
    data.drop(['Volume'], axis=1, inplace=True)

    data_set = data.iloc[:, 0:11]#.values
    pd.set_option('display.max_columns', None)

    #scale
    data_set_scaled = sc.transform(data_set)


    # multiple feature from data provided to the model
    X = []
    #print(data_set_scaled[0].size)
    #data_set_scaled=data_set.values
    backcandles = 30
    #print(data_set_scaled.shape[0])
    for j in range(8):#2 columns are target not X
        X.append([])
        for i in range(backcandles, data_set_scaled.shape[0]):#backcandles+2
            X[j].append(data_set_scaled[i-backcandles:i, j])

    import numpy as np
    #move axis from 0 to position 2
    X=np.moveaxis(X, [0], [2])

    #Erase first elements of y because of backcandles to match X length
    #del(yi[0:backcandles])
    #X, yi = np.array(X), np.array(yi)
    # Choose -1 for last column, classification else -2...
    X, yi =np.array(X), np.array(data_set_scaled[backcandles:,-1])
    #y=np.reshape(yi,(len(yi),1))

    X = np.array([data_set_scaled[i-backcandles:i,:4].copy() for i in range(backcandles,len(data_set_scaled))])

    y=modelloaded.predict(X)
    #print(y)

    yinv=scaler_target_next_close.inverse_transform(y)
    #print(yinv)

    y=yinv[-1]-yinv[-2]
    if(y> 0):
        print('Buy!')
    else:
        print('Sell!')


