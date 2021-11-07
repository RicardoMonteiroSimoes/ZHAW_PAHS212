# Measurments of QNN when using qiskit

Qiskit offers with `CircuitQNN` and `NeuralNetwrokClassifier` a method of optimizing weights in a quantum circuit using comparable algorithms as in classical machine learning. This "front" hides most of the underlying logic, and to truly understand the what is happening and allowing for this combination of theories to work, we have to take a look _under_ the hood. All research and observations has been made on the following source files [`neural_network_classifier.py`](https://github.com/Qiskit/qiskit-machine-learning/blob/main/qiskit_machine_learning/algorithms/classifiers/neural_network_classifier.py) and [`circuit_qnn.py`](https://github.com/Qiskit/qiskit-machine-learning/blob/main/qiskit_machine_learning/neural_networks/circuit_qnn.py).

## `CircuitQNN`

`CircuitQNN` is a class that allows to define the circuit, the input parameters (also referred to as features) and the (optimizable) weight parameters. In our examples, we have used a  `parity` function that maps an `integer` to a class by using `mod(x, classes)`. This opens the suggestion, that the measurment returns a given integer, which can be confirmed trough the documentation:

```py
interpret: A callable that maps the measured integer to another unsigned integer or
                tuple of unsigned integers. These are used as new indices for the (potentially
                sparse) output array. If this is used, and sampling==False, the output shape of
                the output needs to be given as a separate argument. If no interpret function is
                passed, then an identity function will be used by this neural network.
```

A sparse matrix is a matrix that is primarly populated with zeroes. Another point is the `output_shape` that we have, until now, supplied with the amount of possible classis.

```py
output_shape: The output shape of the custom interpretation, only used in the case
                where an interpret function is provided and sampling==False. Note that in the
                remaining cases, the output shape is automatically inferred by: 2^num_qubits if
                sampling==False and interpret==None, (num_samples,1)
                if sampling==True and interpret==None, and
                (num_samples, interpret_shape) if sampling==True and an interpret function
                is provided.
```

As off now (07.11.2021), the code _removes_ any existing measurments in the given circuit first:


```py
 rad_circuit.remove_final_measurements()  # ideally this would not be necessary
```

It seems that even the maintainers are not happy with this, which is understandable: it limits the power users have over their constructions and can be troublesome when trying to understand what is happening. Further down the line we see what the code does: it measures _all_ qubits.

```py
if self._quantum_instance is not None:
    # add measurements in case none are given
    if self._quantum_instance.is_statevector:
        if len(self._circuit.clbits) > 0:
            self._circuit.remove_final_measurements()
    elif len(self._circuit.clbits) == 0:
        self._circuit.measure_all()
```

This shows that we are limited and cannot, for example, use additional qubits to summarize all states and do, as example, binary classification. The measurment is then done the usualy way and turns, depending on set parameters, a different matrix(i.e. sparse).

### `_construct_gradient_circuit`

During initialization of the class, the ends with a coll to `_construct_gradient_circuit`. Some points have been already mentioned before, but the circuit goes as follows:

- copy the given circuit
- remove all measurments
- create parameters
```py
if self._input_gradients:
    params = self._input_params + self._weight_params
else:
    params = self._weight_params
``` 
- create the gradient circuit
```py
self._gradient_circuit = self._gradient.convert(StateFn(grad_circuit), params)
```

`_gradient` is, in this case, an instance of qiskits `Gradient()` class. By definition, it _"Convert an operator expression to the first-order gradient"_. Lets dissect the function.

As we know, `params` are the parameters of the circuit. `StateFn()` is by itself another function of qiskit, defined as follows.

_"State functions are defined to be complex functions over a single binary string (as
    compared to an operator, which is defined as a function over two binary strings, or a
    function taking a binary function to another binary function). This function may be
    called by the eval() method.
    Measurements are defined to be functionals over StateFns, taking them to real values.
    Generally, this real value is interpreted to represent the probability of some classical
    state (binary string) being observed from a probabilistic or quantum system represented
    by a StateFn. This leads to the equivalent definition, which is that a measurement m is
    a function over binary strings producing StateFns, such that the probability of measuring
    a given binary string b from a system with StateFn f is equal to the inner
    product between f and m(b).
    NOTE: State functions here are not restricted to wave functions, as there is
    no requirement of normalization."_

As it stands it doesnt change the circuit at all, it just _wraps_ it with in a class to allow for further functionality. `gradient.convert` is completely different tho. It's call leads to this:

```py
expec_op = PauliExpectation(group_paulis=False).convert(operator).reduce()
```

First, it takes the `PauliExpectation` of the circuit, which is changed the Paulis measurment to a diagonal ({Z, I}^n) basis. It also groups Paulis with the same post-rotation to reduce circuit execution overhead.

After this, the following line in `Gradient` is applied:

```py
cleaned_op = self._factor_coeffs_out_of_composed_op(expec_op)
```

`_factor_coeffs_out_of_composed_op` is a function that comes from the superclass `derivative_base`:

_"Factor all coefficients of ComposedOp out into a single global coefficient.
        Part of the automatic differentiation logic inside of Gradient and Hessian
        counts on the fact that no product or chain rules need to be computed between
        operators or coefficients within a ComposedOp. To ensure this condition is met,
        this function traverses an operator and replaces each ComposedOp with an equivalent
        ComposedOp, but where all coefficients have been factored out and placed onto the
        ComposedOp. Note that this cannot be done properly if an OperatorMeasurement contains
        a SummedOp as it's primitive."_

It checks to see if the type is `ListOp`, and if it is, it traverses `expec_op` using `_factor_coeffs_out_of_composed_op`.

```py
if isinstance(operator, ListOp) and not isinstance(operator, ComposedOp):
    return operator.traverse(cls._factor_coeffs_out_of_composed_op)
if isinstance(operator, ComposedOp):
    total_coeff = operator.coeff
    take_norm_of_coeffs = False
    for k, op in enumerate(operator.oplist):
        if take_norm_of_coeffs:
            #multiply the total coefficient with the current coefficient and its conjugated form
            total_coeff *= op.coeff * np.conj(op.coeff)  # type: ignore
        else:
            total_coeff *= op.coeff  # type: ignore
        if hasattr(op, "primitive"):
            prim = op.primitive  # type: ignore
            if isinstance(op, StateFn) and isinstance(prim, TensoredOp):
                # Check if any of the coefficients in the TensoredOp is a
                # ParameterExpression
                for prim_op in prim.oplist:
                    # If a coefficient is a ParameterExpression make sure that the
                    # coefficients are pulled together correctly
                    if isinstance(prim_op.coeff, ParameterExpression):
                        prim_tensored = StateFn(
                            prim.reduce(), is_measurement=op.is_measurement, coeff=op.coeff
                        )
                        operator.oplist[k] = prim_tensored
                        return operator.traverse(cls._factor_coeffs_out_of_composed_op)
            elif isinstance(prim, ListOp):
                raise ValueError(
                    "This operator was not properly decomposed. "
                    "By this point, all operator measurements should "
                    "contain single operators, otherwise the coefficient "
                    "gradients will not be handled properly."
                )
            if hasattr(prim, "coeff"):
                if take_norm_of_coeffs:
                    total_coeff *= prim._coeff * np.conj(prim._coeff)
                else:
                    total_coeff *= prim._coeff
        if isinstance(op, OperatorStateFn) and op.is_measurement:
            take_norm_of_coeffs = True
    return cls._erase_operator_coeffs(operator).mul(total_coeff)

else:
    return operator
```





### `_probability_gradient`

The class contains a function called `_probability_gradients`. Per se, it doesn't do much at all. It prepares the features and the weigths so that they can be bound to the circuit and evaluated. Before that happens, the following line is executed:

```py
grad = (
    self._sampler.convert(self._gradient_circuit, param_values)
    .bind_parameters(param_values)
    .eval()
)
```

The `_sampler` is an instance of `CircuitSampler`, and it's documentation goes as follows:

```
The CircuitSampler traverses an Operator and converts any CircuitStateFns into
approximations of the state function by a DictStateFn or VectorStateFn using a quantum
backend. Note that in order to approximate the value of the CircuitStateFn, it must 1) send
state function through a depolarizing channel, which will destroy all phase information and
2) replace the sampled frequencies with **square roots** of the frequency, rather than the raw
probability of sampling (which would be the equivalent of sampling the **square** of the
state function, per the Born rule.
The CircuitSampler aggressively caches transpiled circuits to handle re-parameterization of
the same circuit efficiently. If you are converting multiple different Operators,
you are better off using a different CircuitSampler for each Operator to avoid cache thrashing.
```