# parameter: nbnn_test - number of nearest neighbours for testing (averaged)
n = P["nbnn_test"]
nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree').fit(X_train)
distances, indices = nbrs.kneighbors(X_test)

# compute the differences, and train_example prices
fdiffs = []
rtrain = []
rtest = []
for i, inds in enumerate(indices): 
  org = X_test[i]
  for ind in inds:
    sec = X_train[ind]
    fdiffs.append(org-sec)
    rtrain.append(y_train[ind])
    # rtest.append(y_test[i])
X_test_d = np.array(fdiffs)
y_test_source = np.array(rtrain)

y_pred_d = model.predict(X_test_d).flatten()
y_pred_r = y_test_source - y_pred_d

y_pred_ds = (y_pred_r * df_m.price.std()) + df_m.price.mean()
y_test_ds = (y_test * df_m.price.std()) + df_m.price.mean()

y_pred_ds_m = np.mean(y_pred_ds.reshape(-1, n), axis=1)
