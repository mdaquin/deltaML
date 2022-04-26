n = P["nbnn_test"]
y_pred = dl.predict(model, np.array(X_test), neighbors=n)
