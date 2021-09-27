# Setting up the environment

## Virtual environments

Install conda to help you manage your environments. This makes sure you don't have any conflicting installs/plugins that might result in headaches. To find out how to install conda for your flavour of OS, go [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

The IDE isn't given, but we suggest using [VSCode](https://code.visualstudio.com/) for it, as it has plenty of usefull plugins and works well with jupyter notebooks.

Depending on the steps you take and if you use a virtual environment or not, you might have to install the .ipykernel somewhere in between:

```
pip install ipykernel
```

It's also good to have numpy installed, as many libraries rely on it and it offers good tools for mathematical calculations

```
pip install numpy
```

---


## qiskit

Using the installed conda, we first create a virtual environment. For qiskit, python 3 is a prerequisite:
```
conda create -n NAME python=3
```

Accept any installs, as it makes sure your python setup is correct and up to the required version

To switch to it, we can use
```
conda activate NAME
```

_After turning on your virtual environment, you can install the libraries referenced above!_

To turn it off, we simply write
```
conda deactivate NAME
```

To finally install qiskit, we have two packages that we will use. The main package:

```
pip install qiskit
```

And a secondary package that we can use to visualize circuits
```
pip install qiskit[visualization]
```

Some usefull plugins if you are using VSCode are:
- https://github.com/Microsoft/vscode-jupyter
- https://github.com/microsoft/pylance-release
- https://github.com/Microsoft/vscode-python
- https://github.com/qiskit-community/qiskit-vscode

To test if your installation was completed correctly, there is a file you can try and run [here](https://github.com/RicardoMonteiroSimoes/ZHAW_PAHS212/blob/main/jupyter/qiskit_test.ipynb)

### IBM Quantum

To have the ability to test our circuits on a real quantum device, you need an account on [IBM Quantum](https://quantum-computing.ibm.com/) as well as a token. The token can be saved locally so you don't have to write it down every time.

#### Setup:
- Create account on [IBM Quantum](https://quantum-computing.ibm.com/)
- Copy the token on your profile
- Open your console
- Write the following commands:
```python
$ python
>>> from qiskit import IBMQ
>>> IBMQ.save_account("your token")
>>>
```
- Your token should now be saved locally and usable in a script with `IBMQ.load_account()`

If you have any problems with this setup, consult the [help page](https://quantum-computing.ibm.com/lab/docs/iql/manage/account/ibmq)
---


# PennyLane

[PennyLane](https://pennylane.ai/)

## Setup
https://github.com/PennyLaneAI/pennylane#installation


## jupyter notebooks

To test if your installation was completed correctly, there is a file you can try and run:
- [pennylane_test](jupyter/pennylane_test.ipynb)
- [pennylane and qiskit](jupyter/pennylane_qiskit.ipynb)


