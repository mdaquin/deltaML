
y_pred = dl.predict(model, np.array(X_test), neighbors=P["nbnn_test"], selection="nearest")
