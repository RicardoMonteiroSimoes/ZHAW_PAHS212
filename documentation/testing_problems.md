# Non-Trivial testing problems

To create a usefull comparison, we want to find non-trivial ML problems that can be solved in, preferably, multitude of ways (i.e. log. regression, SVM, NN), and compare it to a hybrid based solution.



---

## IRIS Dataset

The [IRIS Dataset](https://www.kaggle.com/uciml/iris) is a collection of data on the iris plant, and the goal is to train a model so that it can correctly identify the class, given the following attributes:
- sepal length
- sepal width
- petal length
- petal width

The output is measured as in which *species* the sample belongs to

---

## Red Wine Quality Data Set

This [dataset](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009) contains data on wine with the following attributes:

1 - fixed acidity
2 - volatile acidity
3 - citric acid
4 - residual sugar
5 - chlorides
6 - free sulfur dioxide
7 - total sulfur dioxide
8 - density
9 - pH
10 - sulphates
11 - alcohol

The output is a numerical classification of the quality of the wine

---

### Heart Failure Prediction

The following [dataset](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data) is a collection of patient data that can be used to train a model to predict if a patient will suffer from heart failure

---


## Helpfull ideas

[_PCA_](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) could be used to transform the data into a lower dimension, so that it becomes feasible to create a hybrid based solution.
