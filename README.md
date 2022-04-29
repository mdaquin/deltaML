# deltaML - learning from differences

deltaML is a simple library to help using the approach of learning from differences in machine learning. 
The idea of learning from differences originates from Case-Based Reasoning where it is called the Case Difference Heuristic. 
In short, instead of applying a machine learning method on the features of the items considered directly to predict a raw value, 
we apply it to differences between pairs of feature vectors to predict differences in values. 

The advantage of doing include that the results might be more explainable (the predicted price of a used car for example is easier to 
explain than if previded in reference to the real price of a similar used car). Through a number of tests (see the testing tool test.py and the tests folder)
we also showed that: 
   - A neural network learning from difference can sometimes outperform a neural network trained in the usual way
   - Learning by differences help exploit the data more, as multiple case differences can be created for every item in the original training data
   - A neural network learning from differences might require significantly less training (in number of epochs) than a neural networked trained in the usual way.
  
## How to use deltaML

Below is extract of code using deltaML, in addition to a machine learning library, to train a model from differences. 
As can be seen from the provided tests, so far, deltaML has been tested with keras (neural networks) for regression. 
It should however be adaptable to other libraries and methods. 

```python
...
from deltaml import DiffLearning
...
# X_train, y_train, X_test and y_test have been created in the usual way
dl = DiffLearning(X_train, y_train, neighbors=2)
X_train_d, y_train_d = dl.diffDataset()
...
# a machine learning model has been created and trained on X_train_d and y_train_d
y_pred = dl.predict(model, X_test, neighbors=3)
...
```

## More details

deltaML mainly contains two functions: one to create a difference dataset, and one to apply the predict function 
of a model, predicting differences, and applying those differences to the actual target value of the used similar 
cases from the training set. 

To create a difference training set, an instance of the class DiffLearning will first be created, so the diffDataset method can be called. 
diffDataset does not take any parameter, and will create a new dataset based on the differences of feature vectors
based on X_train, and the differences of target values based on y_train using the parameters entered when building the instance. 
The creation of this difference-based dataset is therefore driven by the parameters of the constructor of the class DiffLearning:

```python
 DiffLearning(X_train, y_train, neighbors=1, context=False, selection="nearest")
```
- X-train: The feature vectors of the original training set
- y_train: The target values of the original training set
- neighbors (default: 1): The number of similar items to use for each item in X_train/y_train to create differences
- context (default: False): Wether or not to include the context in addition to the differences, i.e. the vector of features of the original item
- selection (default: nearest): How to select pairs of items of creating difference-bases cases. "nearest" selects similar items and "random" select random items

Once the difference-based training set is created and the machine learning model has been trained on it, the predict function 
uses this model to predict differences between the items of X_test and similar (or random) items of X_train, and then apply those
differences on the target values for the used cases of X_train. It returns y_pred, the vector of predicted values. 
```
predict(model, X_test, neighbors=3, selection="nearest")
```
- model: the machine learning model to use for prediction. This model must have been trained on a difference-based training set
- X_test: The vector of items (feature vevtors) for which a target value should be predicted
- neighbors (default: 3): the number of items to retrieve and use for prediction. The predicted differences from all retrieved items will be averaged before being applied 
- selection (default: nearest): How to select items for which to predict differences. "nearest" selects similar items and "random" select random items
