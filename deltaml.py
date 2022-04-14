import numpy as np
from sklearn.neighbors import NearestNeighbors
from skimage.measure import block_reduce

class DiffLearning:
    def __init__(self, X_train, y_train, neighbors=1, context=False, selection="nearest"):
      self.X_train = np.array(X_train)
      self.y_train = np.array(y_train)
      self.n = neighbors
      self.context = context
      self.selection = selection
      self.buildDiffX()
    def buildDiffX(self):
      indices = []
      if self.selection=="nearest":
        nbrs = NearestNeighbors(n_neighbors=self.n+1, algorithm='ball_tree').fit(self.X_train)
        distances, indices = nbrs.kneighbors(self.X_train)
      else: 
         for v in range(len(self.X_train)):
            nv = np.random.randint(len(self.X_train), size=(1,self.n))
            indices.append([v]+list(nv.flatten()))
      lddf = []
      lddr = []
      for inds in indices:
        org = self.X_train[inds[0]].copy()
        for i in range(1,len(inds)):
          sec = self.X_train[inds[i]].copy()
          na = org-sec
          if self.context:
            na = np.concatenate([org,na])
          lddf.append(na)
          lddr.append(self.y_train[inds[0]]-self.y_train[inds[i]])
      self.X_train_d = np.array(lddf)
      self.y_train_d = np.array(lddr)
    def diffDataset(self): return self.X_train_d, self.y_train_d
    def predict(self, model, X_test, neighbors=3, selection="nearest"):
      indices = []
      if selection == "nearest": 
        nbrs = NearestNeighbors(n_neighbors=neighbors, algorithm='ball_tree').fit(self.X_train)
        distances, indices = nbrs.kneighbors(X_test)
      else: 
         for v in range(len(X_test)):
            nv = np.random.randint(len(self.X_train), size=(1,neighbors))
            indices.append(list(nv.flatten()))
      fdiffs = []
      rtrain = []
      for i, inds in enumerate(indices): 
        org = X_test[i]
        for ind in inds:
          sec = self.X_train[ind]
          na = org-sec
          if self.context:
            na = np.concatenate([org,na])
          fdiffs.append(na)
          rtrain.append(self.y_train[ind])
      X_test_d = np.array(fdiffs)
      y_train_source = np.array(rtrain)
      y_pred_d = model.predict(X_test_d)
      if len(y_train_source.shape) == 1:
        y_pred_d = y_pred_d.flatten()
      y_pred_r = y_train_source - y_pred_d
      block_s = (neighbors,)
      if len(y_pred_r.shape) == 2:
        block_s = (neighbors,1)
      y_pred_m = block_reduce(y_pred_r, block_size=block_s, func=np.mean, cval=np.mean(y_pred_r))
      return y_pred_m
