
#!/usr/local/anaconda3/bin/python3

# imports
import os
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
from pennylane.templates import AmplitudeEmbedding
import matplotlib.pyplot as plt

from qiskit.visualization import plot_state_qsphere

import warnings
warnings.simplefilter('ignore')

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))


##################################
# settings
# num_qubits => 2 or 4 will be used
num_layers = 20
iterations = 250

# batch size and optimizer
opt_2qubits = NesterovMomentumOptimizer(0.01)
opt_4qubits = NesterovMomentumOptimizer(0.01)
batch_size = 5

# var_init = (0.01 * np.random.randn(num_layers, num_qubits, 3), 0.0) # initial weights
var_init_2qubits = (0.01 * np.random.randn(num_layers, 2, 3), 0.0)  # initial weights
var_init_4qubits = (0.01 * np.random.randn(num_layers, 4, 3), 0.0)  # initial weights

##################################
# quantum devices
dev_2qubits = qml.device("default.qubit", wires=2)
dev_4qubits = qml.device("default.qubit", wires=4)
#dev_2qubits = qml.device("qiskit.aer", wires=2)
#dev_4qubits = qml.device("qiskit.aer", wires=4)

##################################
# define functions

def get_angles(x):
    beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(
        np.sqrt(x[2] ** 2 + x[3] ** 2)
        / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
    )
    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    return loss

def accuracy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)
    return loss

# 2 qubits
def layer_2qubits(weights):
    qml.Rot(weights[0, 0], weights[0, 1], weights[0, 2], wires=0)
    qml.Rot(weights[1, 0], weights[1, 1], weights[1, 2], wires=1)
    qml.CNOT(wires=[0, 1])

# amplitude embedding
def statepreparation_2qubits(f):
    qml.RY(f[0], wires=0)

    qml.CNOT(wires=[0, 1])
    qml.RY(f[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(f[2], wires=1)

    qml.PauliX(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(f[3], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(f[4], wires=1)
    qml.PauliX(wires=0)

    # other possibility
    # AmplitudeEmbedding(features=f, wires=range(2), pad_with=0., normalize=True)

@qml.qnode(dev_2qubits)
def circuit_2qubits(weights, features):
    statepreparation_2qubits(features)
    for W in weights:
        layer_2qubits(W)
    return qml.expval(qml.PauliZ(0))

def variational_classifier_2qubits(var, features):
    weights = var[0]
    bias = var[1]
    return circuit_2qubits(weights, features) + bias

def cost_2qubits(weights, features, labels):
    predictions = [variational_classifier_2qubits(weights, f) for f in features]
    return square_loss(labels, predictions)

# 4 qubits
def layer_4qubits(weights):
    qml.Rot(weights[0, 0], weights[0, 1], weights[0, 2], wires=0)
    qml.Rot(weights[1, 0], weights[1, 1], weights[1, 2], wires=1)
    qml.Rot(weights[2, 0], weights[2, 1], weights[2, 2], wires=2)
    qml.Rot(weights[3, 0], weights[3, 1], weights[3, 2], wires=3)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])

# repeated amplitude embedding
def statepreparation_4qubits(f):
    # qubits 0,1
    qml.RY(f[0], wires=0)

    qml.CNOT(wires=[0, 1])
    qml.RY(f[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(f[2], wires=1)

    qml.PauliX(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(f[3], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(f[4], wires=1)
    qml.PauliX(wires=0)

    # same thing for qubits 2,3
    qml.RY(f[0], wires=2)

    qml.CNOT(wires=[2, 3])
    qml.RY(f[1], wires=3)
    qml.CNOT(wires=[2, 3])
    qml.RY(f[2], wires=3)

    qml.PauliX(wires=2)
    qml.CNOT(wires=[2, 3])
    qml.RY(f[3], wires=3)
    qml.CNOT(wires=[2, 3])
    qml.RY(f[4], wires=3)
    qml.PauliX(wires=2)

    # other possibility
    # AmplitudeEmbedding(features=f, wires=range(4), pad_with=0., normalize=True)

@qml.qnode(dev_4qubits)
def circuit_4qubits(weights, features):
    statepreparation_4qubits(features)
    for W in weights:
        layer_4qubits(W)
    return qml.expval(qml.PauliZ(0))

def variational_classifier_4qubits(var, features):
    weights = var[0]
    bias = var[1]
    return circuit_4qubits(weights, features) + bias

def cost_4qubits(weights, features, labels):
    predictions = [variational_classifier_4qubits(weights, f) for f in features]
    return square_loss(labels, predictions)

##################################
# load data
data = np.loadtxt(os.path.join(__location__, "data/iris_classes1and2_scaled.txt"))
X = data[:, 0:2]  # get X data
Y = data[:, -1]  # get Y data
print("First X sample (original)  :", X[0])

# padding
padding = 0.3 * np.ones((len(X), 1))
X_pad = np.c_[np.c_[X, padding], np.zeros((len(X), 1))]
print("First X sample (padded)    :", X_pad[0])

# normalization
normalization = np.sqrt(np.sum(X_pad ** 2, -1))
X_norm = (X_pad.T / normalization).T
print("First X sample (normalized):", X_norm[0])

# features
features = np.array([get_angles(x) for x in X_norm])
print("First features sample      :", features[0])

##################################
# train and validation dataset
np.random.seed(0)
num_data = len(Y)
num_train = int(0.75 * num_data)

index = np.random.permutation(range(num_data))

# feats_train = X[index[:num_train]] # use X
# feats_train = X_pad[index[:num_train]]  # use X_pad
feats_train = features[index[:num_train]]  # use features
Y_train = Y[index[:num_train]]

# feats_val = X[index[num_train:]] # use X
# feats_val = X_pad[index[num_train:]]  # use X_pad
feats_val = features[index[num_train:]]  # use features
Y_val = Y[index[num_train:]]

# We need these later for plotting
X_train = X[index[:num_train]]
X_val = X[index[num_train:]]

##################################
##################################
# train the variational classifier
var_2qubits = var_init_2qubits
var_4qubits = var_init_4qubits

weightChanges_2qubits = np.append([], var_2qubits, axis=0)
weightChanges_4qubits = np.append([], var_4qubits, axis=0)

costs_2qubits = np.zeros(iterations)  # initialize costs array
costs_4qubits = np.zeros(iterations)  # initialize costs array

# 2 qubits
# Do some iterations
for it in range(iterations):

    # Update the weights by one optimizer step
    batch_index = np.random.randint(0, num_train, (batch_size,))
    feats_train_batch = feats_train[batch_index]
    Y_train_batch = Y_train[batch_index]

    var_2qubits = opt_2qubits.step(lambda v: cost_2qubits(v, feats_train_batch, Y_train_batch), var_2qubits)

    # Compute predictions on train and validation set
    predictions_train = [np.sign(variational_classifier_2qubits(var_2qubits, f)) for f in feats_train]
    predictions_val = [np.sign(variational_classifier_2qubits(var_2qubits, f)) for f in feats_val]

    # Compute accuracy on train and validation set
    acc_train = accuracy(Y_train, predictions_train)
    acc_val = accuracy(Y_val, predictions_val)

    # gather informations for plotting
    weightChanges_2qubits = np.append(weightChanges_2qubits, var_2qubits, axis=0)
    costs_2qubits[it] = cost_2qubits(var_2qubits, features, Y)

    print(
        "2 Qubits => Iter: {:5d} | Overall cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
        "".format(it + 1, costs_2qubits[it], acc_train, acc_val)
    )

if dev_2qubits.short_name == "qiskit.aer":
    # display(dev_2qubits._circuit_2qubits.draw(output="mpl"))
    print(dev_2qubits._circuit.draw())
else:
    # if device is "default.qubit"
    print(circuit_2qubits.draw())

# 4 qubits
# Do some iterations
for it in range(iterations):

    # Update the weights by one optimizer step
    batch_index = np.random.randint(0, num_train, (batch_size,))
    feats_train_batch = feats_train[batch_index]
    Y_train_batch = Y_train[batch_index]

    var_4qubits = opt_4qubits.step(lambda v: cost_4qubits(v, feats_train_batch, Y_train_batch), var_4qubits)

    # Compute predictions on train and validation set
    predictions_train = [np.sign(variational_classifier_4qubits(var_4qubits, f)) for f in feats_train]
    predictions_val = [np.sign(variational_classifier_4qubits(var_4qubits, f)) for f in feats_val]

    # Compute accuracy on train and validation set
    acc_train = accuracy(Y_train, predictions_train)
    acc_val = accuracy(Y_val, predictions_val)

    # gather informations for plotting
    weightChanges_4qubits = np.append(weightChanges_4qubits, var_4qubits, axis=0)
    costs_4qubits[it] = cost_4qubits(var_4qubits, features, Y)

    print(
        "4 Qubits => Iter: {:5d} | Overall cost: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
        "".format(it + 1, costs_4qubits[it], acc_train, acc_val)
    )

if dev_4qubits.short_name == "qiskit.aer":
    #display(dev_4qubits._circuit_4qubits.draw(output="mpl"))
    print(dev_4qubits._circuit.draw())
else:
    # if device is "default.qubit"
    print(circuit_4qubits.draw())

##################################

## calc weightchanges
# matrix_2qubits = [[[[] for x in range(3)] for y in range(2)] for u in range(num_layers)]
# matrix_4qubits = [[[[] for x in range(3)] for y in range(4)] for u in range(num_layers)]
# # print(matrix_2qubits)
# # print(matrix_4qubits)

# weights_2qubits = weightChanges_2qubits[::2]
# weights_4qubits = weightChanges_4qubits[::2]

# # num iterations
# for i, o in enumerate(weights_2qubits):
#     # num layers
#     for l in range(num_layers):
#         # num qubits
#         for j in range(2):
#             # print("[{}] {}, {} ,{}".format(i, o[l, j, 0], o[l, j, 1], o[l, j, 2]))
#             # print("---")
#             matrix_2qubits[l][j][0].append(o[l, j, 0].item())
#             matrix_2qubits[l][j][1].append(o[l, j, 1].item())
#             matrix_2qubits[l][j][2].append(o[l, j, 2].item())

# # num iterations
# for i, o in enumerate(weights_4qubits):
#     # num layers
#     for l in range(num_layers):
#         # num qubits
#         for j in range(4):
#             # print("[{}] {}, {} ,{}".format(i, o[l, j, 0], o[l, j, 1], o[l, j, 2]))
#             # print("---")
#             matrix_4qubits[l][j][0].append(o[l, j, 0].item())
#             matrix_4qubits[l][j][1].append(o[l, j, 1].item())
#             matrix_4qubits[l][j][2].append(o[l, j, 2].item())


# fig, (ax1, ax2, ax3) = plt.subplots(3)
fig, (ax1) = plt.subplots(1)
# fig.suptitle('Costs and weight changes')
fig.suptitle('Costs')

ax1.plot(range(iterations), costs_2qubits, label='2 qubits')
ax1.plot(range(iterations), costs_4qubits, label='4 qubits')
ax1.set_xlabel('iteration number')
ax1.set_ylabel('cost objective function')
ax1.legend(loc="upper right")

## plot weightchanges
# for l, wv in enumerate(matrix_2qubits):
#   for j in range(2):
#     for i in range(3):
#       label_txt = 'l['+ str(l) +'], q[' + str(j) + '], w['+ str(i) + ']'
#       # print("matrix[j][i]", wv[j][i])
#       ax2.plot(np.arange(0, iterations+1,1), wv[j][i], label=label_txt)

# ax2.title.set_text('weights 2 qubits')
# ax2.set_xlabel('iteration number')
# ax2.set_ylabel('weight value')
# ax2.legend(loc="upper right")

# for l, wv in enumerate(matrix_4qubits):
#   for j in range(4):
#     for i in range(3):
#       label_txt = 'l['+ str(l) +'], q[' + str(j) + '], w['+ str(i) + ']'
#       # print("matrix[j][i]", wv[j][i])
#       ax3.plot(np.arange(0, iterations+1,1), wv[j][i], label=label_txt)

# ax3.title.set_text('weights 4 qubits')
# ax3.set_xlabel('iteration number')
# ax3.set_ylabel('weight value')
# ax3.legend(loc="upper right", ncol=len(ax3.lines))

plt.show()

##################################
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Train vs. validation plots')
cm = plt.cm.RdBu

# make data for decision regions
xx, yy = np.meshgrid(np.linspace(0.0, 1.5, 20), np.linspace(0.0, 1.5, 20))
X_grid = [np.array([x, y]) for x, y in zip(xx.flatten(), yy.flatten())]

# preprocess grid points like data inputs above
padding = 0.3 * np.ones((len(X_grid), 1))
X_grid = np.c_[np.c_[X_grid, padding], np.zeros((len(X_grid), 1))]  # pad each input
normalization = np.sqrt(np.sum(X_grid ** 2, -1))
X_grid = (X_grid.T / normalization).T  # normalize each input
features_grid = np.array(
    [get_angles(x) for x in X_grid]
)  # angles for state preparation are new features

predictions_grid_2qubits = [variational_classifier_2qubits(var_2qubits, f) for f in features_grid]
# predictions_grid = [variational_classifier_2qubits(var, f) for f in X_grid]

predictions_grid_4qubits = [variational_classifier_4qubits(var_4qubits, f) for f in features_grid]
# predictions_grid = [variational_classifier_4qubits(var, f) for f in X_grid]

Z2 = np.reshape(predictions_grid_2qubits, xx.shape)
Z4 = np.reshape(predictions_grid_4qubits, xx.shape)

# plot decision regions
cnt = ax1.contourf(
    xx, yy, Z2, levels=np.arange(-1, 1.1, 0.1), cmap=cm, alpha=0.8, extend="both"
)
ax1.contour(
    xx, yy, Z2, levels=[0.0], colors=("black",), linestyles=("--",), linewidths=(0.8,)
)

# plot data
ax1.scatter(
    X_train[:, 0][Y_train == 1],
    X_train[:, 1][Y_train == 1],
    c="b",
    marker="o",
    edgecolors="k",
    label="class 1 train",
)
ax1.scatter(
    X_val[:, 0][Y_val == 1],
    X_val[:, 1][Y_val == 1],
    c="b",
    marker="^",
    edgecolors="k",
    label="class 1 validation",
)
ax1.scatter(
    X_train[:, 0][Y_train == -1],
    X_train[:, 1][Y_train == -1],
    c="r",
    marker="o",
    edgecolors="k",
    label="class -1 train",
)
ax1.scatter(
    X_val[:, 0][Y_val == -1],
    X_val[:, 1][Y_val == -1],
    c="r",
    marker="^",
    edgecolors="k",
    label="class -1 validation",
)
ax1.legend(loc="upper right")
ax1.title.set_text('2 Qubits')

# plot decision regions
cnt = ax2.contourf(
    xx, yy, Z4, levels=np.arange(-1, 1.1, 0.1), cmap=cm, alpha=0.8, extend="both"
)
ax2.contour(
    xx, yy, Z4, levels=[0.0], colors=("black",), linestyles=("--",), linewidths=(0.8,)
)
#ax2.colorbar(cnt, ticks=[-1, 0, 1])

# plot data
ax2.scatter(
    X_train[:, 0][Y_train == 1],
    X_train[:, 1][Y_train == 1],
    c="b",
    marker="o",
    edgecolors="k",
    label="class 1 train",
)
ax2.scatter(
    X_val[:, 0][Y_val == 1],
    X_val[:, 1][Y_val == 1],
    c="b",
    marker="^",
    edgecolors="k",
    label="class 1 validation",
)
ax2.scatter(
    X_train[:, 0][Y_train == -1],
    X_train[:, 1][Y_train == -1],
    c="r",
    marker="o",
    edgecolors="k",
    label="class -1 train",
)
ax2.scatter(
    X_val[:, 0][Y_val == -1],
    X_val[:, 1][Y_val == -1],
    c="r",
    marker="^",
    edgecolors="k",
    label="class -1 validation",
)
ax2.legend(loc="upper right")
ax2.title.set_text('4 Qubits')

plt.colorbar(cnt, ax=ax1)
plt.colorbar(cnt, ax=ax2)
plt.show()

