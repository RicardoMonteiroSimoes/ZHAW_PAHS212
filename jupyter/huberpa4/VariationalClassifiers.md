# Variational Classifier

Relevant papers
- Example of repeated amplitude embedding: https://www.mdpi.com/2073-431X/10/6/71/htm
- Extensive paper about kernel methods, basis- , amplitude- and angle embedding by Maria Schuld: https://arxiv.org/pdf/2101.11020.pdf
- Paper "Transformation of quantum states using uniformly controlled rotations": https://arxiv.org/pdf/quant-ph/0407010.pdf

## Pennylane Tutorial
Link to the original Pennylane tutorial:
- https://pennylane.ai/qml/demos/tutorial_variational_classifier.html

Jupyter notebook with the Pennylane tutorial:
- [pennylane_iris](pennylane_iris.ipynb)

### State preparation (amplitude embedding)
As a first step we need to encode the data into the qubits.
Depending on the chosen technique (basis-, amplitude-, angle embedding, etc.) the original data needs to be pre-processed.

The VC uses amplitude embedding thus the iris dataset is alread pre–processed as mentioned [here](https://discuss.pennylane.ai/t/inquiries-on-state-preparation-in-variaitonal-classifier-example/248/2):
- dataset file: [iris_classes1and2_scaled.txt](data/iris_classes1and2_scaled.txt) (preprocessed: zero mean, unit deviation)

In the variational classifier the `get_angles()` function is use to calculate the angles for the amplitude embedding using the pre-processed data

**Example steps for one dataset (first line from the file):**

The first line explained:
```
sepal length             sepal width              petal length             petal width              class (-1=setosa,1=versicolor)
3.999999999999999112e-01 7.500000000000000000e-01 1.999999999999999556e-01 5.000000000000000278e-02 -1.000000000000000000e+00
```

| step                                                                                      | data                                                                                                                            |
| :---------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------ |
| first line from the file                                                                  | `3.999999999999999112e-01 7.500000000000000000e-01 1.999999999999999556e-01 5.000000000000000278e-02 -1.000000000000000000e+00` |
| we only use the first two features (sepal length and with)                                | `3.999999999999999112e-01 7.500000000000000000e-01`                                                                             |
| First sample as array                                                                     | `[0.4  0.75]`                                                                                                                   |
| We pad the array with `0.3` and `0.0`                                                     | `[0.4  0.75 0.3  0.  ]`                                                                                                         |
| We normalize the array                                                                    | `[0.44376016 0.83205029 0.33282012 0.        ]`                                                                                 |
| Finally we use the `get_angles([0.44376016 0.83205029 0.33282012 0.        ])` to receive | `[ 0.67858523 -0.          0.         -1.080839    1.080839  ]`                                                                 |


#### `get_angles()` function

The calculation for the formula for `get_angles()` is described in the following paper; https://arxiv.org/pdf/quant-ph/0407010.pdf, read section II & III.
Additionaly there seems to be a slightly modificated version of the formula in the ebook; https://link.springer.com/book/10.1007/978-3-319-96424-9 (Schuld and Petruccione (2018))

Related link to forum question about the `get_angles()` function: https://discuss.pennylane.ai/t/data-encoding-for-a-real-datasets/1090

```python
def get_angles(x):
    beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(
        np.sqrt(x[2] ** 2 + x[3] ** 2)
        / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
    )
    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])
```

The following screenshots are form the paper: https://arxiv.org/pdf/quant-ph/0407010.pdf  which should explain the algorithm to create the `get_angles()` function:
|                                                                                                                     screenshot section II                                                                                                                      |                                                                                                                      screenshot section III                                                                                                                      |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| <a href="images/Transformation%20of%20quantum%20states%20using%20uniformly%20controlled%20rotations%20-%20II.png"><img src="images/Transformation%20of%20quantum%20states%20using%20uniformly%20controlled%20rotations%20-%20II.png" alt="" width="50%" /></a> | <a href="images/Transformation%20of%20quantum%20states%20using%20uniformly%20controlled%20rotations%20-%20III.png"><img src="images/Transformation%20of%20quantum%20states%20using%20uniformly%20controlled%20rotations%20-%20III.png" alt="" width="50%" /></a> |

### Repeated Amplitude Embedding (special case)

| embedding method   |                                                          circuit example                                                          |
| :----------------- | :-------------------------------------------------------------------------------------------------------------------------------: |
| amplitude          | <img src="https://www.mdpi.com/computers/computers-10-00071/article_deploy/html/images/computers-10-00071-g006-550.jpg" alt="" /> |
| repeated amplitude | <img src="https://www.mdpi.com/computers/computers-10-00071/article_deploy/html/images/computers-10-00071-g007-550.jpg" alt="" /> |
The pictures above do not resemble the variational classifier for the "DATA ENCODING" part.

#### Examples from the current implementation of the VC (Variational Classifier)
1 Layer with 2 Qubits (single amplitude embedding):
```
    ╭──────────────────────── DATA ENCODING ─────────────────────────╮  ╭──────────── LAYER 1 ───────────────╮

 0: ──RY(0.712)──╭C──────────╭C──X──────╭C──────────────╭C──X───────── | ─Rot(0.0102, 1.29, -4.32e-06)────╭C──┤ ⟨Z⟩
 1: ─────────────╰X──RY(-0)──╰X──RY(0)──╰X──RY(-0.519)──╰X──RY(0.519)─ | ─Rot(0.00231, -1.58, -7.91e-05)──╰X──┤
```
The layers have `2x3=6` weights and all qubits are entangled with a CNOT.

1 Layer with 4 Qubits (**repeated** amplitude embedding):
```
    ╭──────────────────────── DATA ENCODING ─────────────────────────╮  ╭────────────────── LAYER 1 ──────────────────────╮

 0: ──RY(0.712)──╭C──────────╭C──X──────╭C──────────────╭C──X───────── | ─Rot(0.00108, -0.963, 0.000439)───╭C──────────╭X──┤ ⟨Z⟩
 1: ─────────────╰X──RY(-0)──╰X──RY(0)──╰X──RY(-0.519)──╰X──RY(0.519)─ | ─Rot(-0.00117, -1.04, -0.00308)───╰X──╭C──────│───┤
 2: ──RY(0.712)──╭C──────────╭C──X──────╭C──────────────╭C──X───────── | ─Rot(0.00106, -0.246, -0.000862)──────╰X──╭C──│───┤
 3: ─────────────╰X──RY(-0)──╰X──RY(0)──╰X──RY(-0.519)──╰X──RY(0.519)─ | ─Rot(-0.00351, -0.489, 0.00503)───────────╰X──╰C──┤
```
The layers have `4x3=12` weights and all qubits are entangled with a CNOT.




## VC Script and Plots
I created and modified the VC script ([VC_with_pennylane.py](VC_with_pennylane.py)) to contain amplitude embedding and repeated amplitude embedding for comparison.

Change `num_layers` and `iterations` in the script
```python
num_layers = 1
iterations = 50
```

**Settings user for the plots below:**
Iterations: `iterations=250`
Quantum device used: `default.qubit` (pennylane default)
Optimizers: `NesterovMomentumOptimizer` (step size: `0.01` for 2 qubits and 4 qubits)

The following plots show the differences for this two approachs

| Layers |                         costs                         |                      train/validation plot                       |
| :----- | :---------------------------------------------------: | :--------------------------------------------------------------: |
| 1      |  <img src="images/vc_250_layer1_costs.png" alt="" />  |  <img src="images/vc_250_layer1_train_validation.png" alt="" />  |
| 2      | <img src="images/vc_250_layers2_costs.png" alt="" />  | <img src="images/vc_250_layers2_train_validation.png" alt="" />  |
| 3      | <img src="images/vc_250_layers3_costs.png" alt="" />  | <img src="images/vc_250_layers3_train_validation.png" alt="" />  |
| 5      | <img src="images/vc_250_layers5_costs.png" alt="" />  | <img src="images/vc_250_layers5_train_validation.png" alt="" />  |
| 10     | <img src="images/vc_250_layers10_costs.png" alt="" /> | <img src="images/vc_250_layers10_train_validation.png" alt="" /> |
| 20     | <img src="images/vc_250_layers20_costs.png" alt="" /> | <img src="images/vc_250_layers20_train_validation.png" alt="" /> |

Repeated amplitude embedding leads to faster training times and better predictions but uses the double amount of qubits for the same amount of features

### VC: Try to use 3 classes and all features from iris
Naive approach to classify the whole iris datset:
- [pennylane_iris2](pennylane_iris2.ipynb)

The idea is to classify the data depending on the Y rotation between -1, 1:
- classX between: -1 and  -1+(2/3)
- classY between: 0 - (1/3) and 0 + (1/3)
- classZ between: 1-(2/3) and 1

This approach assumes one can measure the Z axis value precisely

## More on embedding methods:

Quantum Data embedding Methods for Quantum Machine Learning:
- https://medium.datadriveninvestor.com/all-about-data-encoding-for-quantum-machine-learning-2a7344b1dfef
- [pennylane embedding examples](pennylane_embedding.ipynb)
