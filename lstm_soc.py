import os
from math import sqrt
from numpy import concatenate, random
from matplotlib import pyplot as plt
from pandas import read_csv, DataFrame, concat
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout

random.seed(7)
 
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

file = str(os.getcwd())+'/socfix.txt'
print(file)

# load dataset
dataset = read_csv(file)
values = dataset.values
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[3], axis=1, inplace=True)
print(reframed.head())

# split into train and test sets
values = reframed.values
n_train = int(len(dataset)*0.75)
train = values[:n_train, :]
test = values[n_train:, :]
# split into input and outputs
train_x, train_y = train[:, :-1], train[:, -1]
test_x, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

# design network
model = Sequential()
model.add((LSTM(150, input_shape=(train_x.shape[1], train_x.shape[2]))))
model.add(Dense(units = 1))

model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_x, train_y, epochs=100, batch_size=32, validation_data=(test_x, test_y), verbose=1, shuffle=False)
# plot history
plt.figure(1)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.show()

# make a prediction
yhat0 = model.predict(train_x)
train_x = train_x.reshape((train_x.shape[0],train_x.shape[2]))
# invert scaling for forecast
inv_yhat0 = concatenate((yhat0,train_x[:,1:]),axis=1)
inv_yhat0 = scaler.inverse_transform(inv_yhat0)
inv_yhat0 = inv_yhat0[:,0]
# invert scaling for actual
train_y0 = train_y.reshape((len(train_y),1))
inv_y0 = concatenate((train_y0,train_x[:,1:]),axis=1)
inv_y0 = scaler.inverse_transform(inv_y0)
inv_y0 = inv_y0[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y0, inv_yhat0))
print('Train RMSE: %.3f' % rmse)

# make a prediction
yhat = model.predict(test_x)
test_x = test_x.reshape((test_x.shape[0],test_x.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat,test_x[:,1:]),axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y),1))
inv_y = concatenate((test_y,test_x[:,1:]),axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y,inv_yhat))
print('Test RMSE: %.3f' % rmse)

#pyplot.plot(dataset[0])

x0 = []
x1 = []

for i in range(len(train_y)):
	x0.append(i)

for i in range(len(test_y)):
	x1.append(i + len(train_y) + 1)

ydata = read_csv(file, usecols=[0], engine='python')

plt.figure(2)
plt.plot(ydata, 'b', label='Actual value')
plt.plot(x1,inv_yhat, 'r', label='Testing')
plt.xlabel('Series')
plt.ylabel('SoC (%)')
plt.title('State of Charge')
plt.legend()
plt.xlim(len(train_y)+1,)
plt.show()

plt.figure(3)
plt.plot(ydata, 'b', label='Actual value')
plt.plot(x0,inv_yhat0, 'g', label='Training')
plt.plot(x1,inv_yhat, 'r', label='Testing')
plt.xlabel('Series')
plt.ylabel('SoC (%)')
plt.title('State of Charge')
plt.xlim(0,)
plt.legend()
plt.show()

plt.figure(4)
plt.plot(ydata, 'b', label='Actual value')
plt.plot(x0,inv_yhat0, 'g', label='Training')
plt.xlabel('Series')
plt.ylabel('SoC (%)')
plt.title('State of Charge')
plt.legend()
plt.xlim(0,len(train_y))
plt.show()
