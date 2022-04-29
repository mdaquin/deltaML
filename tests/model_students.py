t1 = time.time()
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(len(X_train.iloc[0]),)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=P["epochs"], batch_size=128, verbose=2)
model_time = time.time() - t1
