# Non-Trivial testing problems

To create a usefull comparison, we want to find non-trivial ML problems that can be solved in, preferably, multitude of ways (i.e. log. regression, SVM, NN), and compare it to a hybrid based solution

---

## IRIS Dataset

The [IRIS Dataset](https://archive.ics.uci.edu/ml/datasets/iris) is a collection of data on the iris plant, and the goal is to train a model so that it can correctly identify the class, given the following attributes:
- sepal length
- sepal width
- petal length
- petal width

It's a classic example of machine learning and the origin paper of it by [_Fisher et al._](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1469-1809.1936.tb02137.x) is still in this day often cited. There are multiple solutions to this problem, with varrying results:

[_exceedingly simple domain_](https://archive.ics.uci.edu/ml/datasets/iris)

#### SVM based approach

_Support Vector Machines_ offer a form of unsupervised learning, that can predict the result according to the inputs. One of the problems of a SVM is that the data has to be linearly differentiable data, and that the there can only be 2 attributes, as opposed to the 4 given. The special part of the IRIS dataset is that 2 of the 3 represented classes _cannot_ be linearly separated trough a classical SVM. Using a kernel trick helps in this matter and can [help achieve acceptable results](https://towardsdatascience.com/multiclass-classification-with-support-vector-machines-svm-kernel-trick-kernel-functions-f9d5377d6f02)

#### NN based approach

_Neural networks_ offer great flexibility when it comes to the problem theyre applied to. One can also create a NN to be trained on the IRIS dataset and achieve satysfing results in terms of [accuracy](https://www.kaggle.com/azzion/iris-data-set-classification-using-neural-network)

---

## Wine Quality Data Set

The wine dataset offers 11 attributes with a quality score, defining the quality of the wine. 
https://towardsdatascience.com/red-wine-quality-prediction-using-regression-modeling-and-machine-learning-7a3e2c3e1f46
https://www.kaggle.com/gauravduttakiit/red-wine-quality-linear-regression

[_These datasets can be viewed as classification or regression tasks. The classes are ordered and not balanced (e.g. there are many more normal wines than excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent or poor wines. Also, we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods._](https://archive.ics.uci.edu/ml/datasets/wine+quality)

---

## Helpfull ideas

[_PCA_](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) could be used to transform the data into a lower dimension, so that it becomes feasible to create a hybrid based solution.