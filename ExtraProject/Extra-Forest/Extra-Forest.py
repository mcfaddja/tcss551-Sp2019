import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam, Nadam, Adamax
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical


data = pd.read_csv('train.csv')
data = data.drop('Id', axis=1)
data['Cover_Type'] = data['Cover_Type'].astype('category')

features = data.drop('Cover_Type', axis=1)
types = data['Cover_Type']

train_features, test_features, train_types, test_types = train_test_split(features, types, test_size = 0.15, random_state = 42)


model = Sequential()

model.add(Dense(54,
				input_dim=54,
				kernel_initializer='uniform',
				activation='relu'))
model.add(Dropout(0.125))

model.add(Dense(108,
				kernel_initializer='uniform',
				activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(162,
				kernel_initializer='uniform',
				activation='relu'))
model.add(Dropout(0.375))

model.add(Dense(108,
				kernel_initializer='uniform',
				activation='sigmoid'))
model.add(Dropout(0.25))

model.add(Dense(54,
                kernel_initializer='uniform',
                activation='sigmoid'))
model.add(Dropout(0.125))

# model.add(Dense(51,
#                 kernel_initializer='uniform',
#                 activation='relu'))
# model.add(Dropout(0.118))

# model.add(Dense(51,
#                 kernel_initializer='uniform',
#                 activation='sigmoid'))
# model.add(Dropout(0.118))

model.add(Dense(12,
                kernel_initializer='uniform',
                activation='relu'))
model.add(Dropout(0.028))

model.add(Dense(12,
                kernel_initializer='uniform',
                activation='sigmoid'))
model.add(Dropout(0.028))

model.add(Dense(10,
                kernel_initializer='uniform',
                activation='relu'))
model.add(Dropout(0.028))

model.add(Dense(10,
                kernel_initializer='uniform',
                activation='sigmoid'))
model.add(Dropout(0.028))

# model.add(Dense(8,
#                 kernel_initializer='uniform',
#                 activation='relu'))
# model.add(Dropout(0.021))

# model.add(Dense(8,
#                 kernel_initializer='uniform',
#                 activation='sigmoid'))
# model.add(Dropout(0.021))

model.add(Dense(8,kernel_initializer='uniform',activation='sigmoid'))


# myRMS = RMSprop(lr=0.0005, rho=0.9, epsilon=None, decay=0.0)
model.compile(optimizer='rmsprop',
				loss='categorical_crossentropy',
				metrics=['accuracy'])


binary_train_types = to_categorical(train_types)


scaler1 = StandardScaler()
scaler1 = scaler1.fit(train_features)
xScaled0 = scaler1.transform(train_features)

scaler2 = MinMaxScaler(feature_range=(0,1))
scaler2 = scaler2.fit(xScaled0)
xScaled = scaler2.transform(xScaled0)

model.fit(xScaled, binary_train_types, epochs=100)

xTestScale0 = scaler1.transform(test_features)
xTestScale = scaler2.transform(xTestScale0)

pred_types = model.predict(xTestScale)
print(pred_types)
print(type(pred_types))

round_pred_types = np.round(pred_types)
df_round_pred_types = pd.DataFrame(data=round_pred_types, columns=['Cover_Type'])


# myScore = accuracy_score(test_types, pred_types)
# print(myScore)