import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam, Nadam, Adamax
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error


data = pd.read_csv('train.csv')
data = data.drop('Id', axis=1)
data['Cover_Type'] = data['Cover_Type'].astype('category')

features = data.drop('Cover_Type', axis=1)
types = data['Cover_Type']

train_features, test_features, train_types, test_types = train_test_split(features, types, test_size = 0.15, random_state = 42)


def NNmodel():
    
    myInputs = Input(shape=(54,))

    x = Dense(108)(myInputs)
    x = Activation('sigmoid')(x)
    x = Dropout(0.26)(x)

    x = Dense(162)(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.39)(x)

    x = Dense(54)(myInputs)
    x = Activation('sigmoid')(x)
    x = Dropout(0.13)(x)
    
    x = Dense(51)(myInputs)
    x = Activation('sigmoid')(x)
    x = Dropout(0.12)(x)
    
    x = Dense(12)(myInputs)
    x = Activation('sigmoid')(x)
    x = Dropout(0.03)(x)
    
    x = Dense(10)(myInputs)
    x = Activation('sigmoid')(x)
    x = Dropout(0.03)(x)
    
    x = Dense(8)(myInputs)
    x = Activation('sigmoid')(x)
    x = Dropout(0.02)(x)

    myPredicts = Dense(1)(x)
    myModel = Model(inputs=myInputs, outputs=myPredicts)
    
    return myModel


model = NNmodel()
myRMS = RMSprop(lr=0.0005, rho=0.9, epsilon=None, decay=0.0)
model.compile(optimizer=myRMS, loss='mse', metrics=['mse'])

scaler1 = StandardScaler()
scaler1 = scaler1.fit(train_features)
xScaled0 = scaler1.transform(train_features)

scaler2 = MinMaxScaler(feature_range=(0,1))
scaler2 = scaler2.fit(xScaled0)
xScaled = scaler2.transform(xScaled0)

model.fit(xScaled, train_types, epochs=25, verbose=0)

xTestScale0 = scaler1.transform(test_features)
xTestScale = scaler2.transform(xTestScale0)

pred_types = model.predict(xTestScale)
print(pred_types)
print(type(pred_types))

round_pred_types = np.round(pred_types)
df_round_pred_types = pd.DataFrame(data=round_pred_types, columns=['Cover_Type'])
df_round_pred_types['Cover_Type'] = df_round_pred_types['Cover_Type'].astype('category')

# myScore = mean_squared_error(test_types, pred_types)
# print(myScore)

# myScore1 = mean_absolute_error(test_types, pred_types)
# print(myScore1)

myScore = accuracy_score(test_types, df_round_pred_types)
print(myScore)