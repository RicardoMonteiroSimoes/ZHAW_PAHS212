# Heart Failure

## MLPClassifier

| Classic MLPClassifier Accuracy \| Solver: `Adam` | Plot                                                           |
| :----------------------------------------------- | :------------------------------------------------------------- |
| [size: 25, hidden layers: 1]:  `88.33%`          | <img src="assets/MLPClassifier_heart-failure_01.png" alt="" /> |
| [size: 25, hidden layers: 13]:  `91.67%`         | <img src="assets/MLPClassifier_heart-failure_02.png" alt="" /> |
| [size: 25, hidden layers: 25]:  `93.33%`         | <img src="assets/MLPClassifier_heart-failure_03.png" alt="" /> |
| [size: 50, hidden layers: 1]:  `90.00%`          | <img src="assets/MLPClassifier_heart-failure_04.png" alt="" /> |
| [size: 50, hidden layers: 13]:  `93.33%`         | <img src="assets/MLPClassifier_heart-failure_05.png" alt="" /> |
| [size: 50, hidden layers: 25]:  `91.67%`         | <img src="assets/MLPClassifier_heart-failure_06.png" alt="" /> |
| [size: 75, hidden layers: 1]:  `86.67%`          | <img src="assets/MLPClassifier_heart-failure_07.png" alt="" /> |
| [size: 75, hidden layers: 13]:  `90.00%`         | <img src="assets/MLPClassifier_heart-failure_08.png" alt="" /> |
| [size: 75, hidden layers: 25]:  `93.33%`         | <img src="assets/MLPClassifier_heart-failure_09.png" alt="" /> |


## PA Quatum circuit

### Original data
| Quantum Circuit                                                               | Loss function plot                                                                 | Mean Accuracy                                                                                        |
| :---------------------------------------------------------------------------- | :--------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_01.png"  alt="" />          | <img src="assets/PA-QuantumCicrcuit_heart-failure_plot_01.png"  alt="" />          | training: `0.5899581589958159`<br />testing: `0.6333333333333333`<br />overall: `0.6020066889632107` |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_5layers_01.png"  alt="" />  | <img src="assets/PA-QuantumCicrcuit_heart-failure_5layers_plot_01.png"  alt="" />  | training: `0.7112970711297071`<br />testing: `0.75`<br />overall: `0.7224080267558528`               |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_10layers_01.png"  alt="" /> | <img src="assets/PA-QuantumCicrcuit_heart-failure_10layers_plot_01.png"  alt="" /> | traingin: `0.702928870292887`<br />testing: `0.75`<br />overall: `0.7123745819397993`                                                                           |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_15layers_01.png"  alt="" /> | <img src="assets/PA-QuantumCicrcuit_heart-failure_15layers_plot_01.png"  alt="" /> | traingin: `0.6861924686192469`<br />testing: `0.75`<br />overall: `0.6956521739130435`                                                                |
### Normalized data
| Quantum Circuit                                                               | Loss function plot                                                                 | Mean Accuracy                                                                         |
| :---------------------------------------------------------------------------- | :--------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------ |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_01.png"  alt="" />          | <img src="assets/PA-QuantumCicrcuit_heart-failure_plot_02.png"  alt="" />          | training: `0.6778242677824268`<br />testing: `0.7`<br />overall: `0.6822742474916388` |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_5layers_01.png"  alt="" />  | <img src="assets/PA-QuantumCicrcuit_heart-failure_5layers_plot_02.png"  alt="" />  | training: `0.698744769874477`<br />testing: `0.75`<br />overall: `0.7090301003344481` |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_10layers_01.png"  alt="" /> | <img src="assets/PA-QuantumCicrcuit_heart-failure_10layers_plot_02.png"  alt="" /> | traingin: `0.702928870292887`<br />testing: `0.75`<br />overall: `0.7123745819397993`                        |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_15layers_01.png"  alt="" /> | <img src="assets/PA-QuantumCicrcuit_heart-failure_15layers_plot_02.png"  alt="" /> | processing...                                                                         |

#### Other Circuits

**Cascade mirrored**

| Quantum Circuit                                                              | Loss function plot                                                                | Mean Accuracy                                                                                        |
| :--------------------------------------------------------------------------- | :-------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_02.png"  alt="" />         | <img src="assets/PA-QuantumCicrcuit_heart-failure_plot_03.png"  alt="" />         | training: `0.694560669456067`<br />testing: `0.7166666666666667`<br />overall: `0.7023411371237458`  |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_5layers_02.png"  alt="" /> | <img src="assets/PA-QuantumCicrcuit_heart-failure_5layers_plot_03.png"  alt="" /> | training: `0.6820083682008368`<br />testing: `0.7333333333333333`<br />overall: `0.6923076923076923` |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_10layers_02.png"  alt="" /><br />todo change image here | <img src="assets/PA-QuantumCicrcuit_heart-failure_10layers_plot_03.png"  alt="" /> | processing...                      |

**Entangle all qubits with cx**

| Quantum Circuit                                                              | Loss function plot                                                                | Mean Accuracy                                                                                        |
| :--------------------------------------------------------------------------- | :-------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_03.png"  alt="" />         | <img src="assets/PA-QuantumCicrcuit_heart-failure_plot_04.png"  alt="" />         | training: `0.6694560669456067`<br />testing: `0.7166666666666667`<br />overall: `0.6789297658862876` |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_5layers_03.png"  alt="" /> | <img src="assets/PA-QuantumCicrcuit_heart-failure_5layers_plot_04.png"  alt="" /> | training: `0.6903765690376569`<br />testing: `0.7333333333333333`<br />overall: `0.6989966555183946` |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_10layers_03.png"  alt="" /> | <img src="assets/PA-QuantumCicrcuit_heart-failure_10layers_plot_04.png"  alt="" /> | processing...                      |


**Entangle all qubits with ccx**

| Quantum Circuit                                                              | Loss function plot                                                                | Mean Accuracy                                                                                        |
| :--------------------------------------------------------------------------- | :-------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_04.png"  alt="" />         | <img src="assets/PA-QuantumCicrcuit_heart-failure_plot_05.png"  alt="" />         | training: `0.6861924686192469`<br />testing: `0.7166666666666667`<br />overall: `0.6923076923076923` |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_5layers_04.png"  alt="" /> | <img src="assets/PA-QuantumCicrcuit_heart-failure_5layers_plot_05.png"  alt="" /> | training: `0.6903765690376569`<br />testing: `0.7333333333333333`<br />overall: `0.7023411371237458` |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_10layers_04.png"  alt="" /> | <img src="assets/PA-QuantumCicrcuit_heart-failure_10layers_plot_05.png"  alt="" /> | processing...                      |

**Hadamard start and end**

| Quantum Circuit                                                              | Loss function plot                                                                | Mean Accuracy                                                                                       |
| :--------------------------------------------------------------------------- | :-------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------- |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_05.png"  alt="" />         | <img src="assets/PA-QuantumCicrcuit_heart-failure_plot_06.png"  alt="" />         | training: `0.698744769874477`<br />testing: `0.7166666666666667`<br />overall: `0.6989966555183946` |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_5layers_05.png"  alt="" /> | <img src="assets/PA-QuantumCicrcuit_heart-failure_5layers_plot_06.png"  alt="" /> | training: `0.6778242677824268`<br />testing: `0.7333333333333333`<br />overall: `0.68561872909699`  |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_10layers_05.png"  alt="" /> | <img src="assets/PA-QuantumCicrcuit_heart-failure_10layers_plot_06.png"  alt="" /> | processing...                      |

**More weights (UGate)**

| Quantum Circuit                                                              | Loss function plot                                                                | Mean Accuracy                                                                                        |
| :--------------------------------------------------------------------------- | :-------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_06.png"  alt="" />         | <img src="assets/PA-QuantumCicrcuit_heart-failure_plot_07.png"  alt="" />         | training: `0.702928870292887`<br />testing: `0.75`<br />overall: `0.7023411371237458`                |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_5layers_06.png"  alt="" /> | <img src="assets/PA-QuantumCicrcuit_heart-failure_5layers_plot_07.png"  alt="" /> | training: `0.6861924686192469`<br />testing: `0.7166666666666667`<br />overall: `0.6989966555183946` |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_10layers_06.png"  alt="" /> | <img src="assets/PA-QuantumCicrcuit_heart-failure_10layers_plot_07.png"  alt="" /> | processing...                      |

**More weights (UGate), no entanglement**

| Quantum Circuit                                                              | Loss function plot                                                                | Mean Accuracy                                                                         |
| :--------------------------------------------------------------------------- | :-------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------ |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_07.png"  alt="" />         | <img src="assets/PA-QuantumCicrcuit_heart-failure_plot_08.png"  alt="" />         | training: `0.6778242677824268`<br />testing: `0.7`<br />overall: `0.6822742474916388` |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_5layers_07.png"  alt="" /> | <img src="assets/PA-QuantumCicrcuit_heart-failure_5layers_plot_08.png"  alt="" /> | training: `0.6736401673640168`<br />testing: `0.7`<br />overall: `0.68561872909699`   |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_10layers_07.png"  alt="" /> | <img src="assets/PA-QuantumCicrcuit_heart-failure_10layers_plot_08.png"  alt="" /> | processing...                      |

**Early cx gates, ugate**

| Quantum Circuit                                                              | Loss function plot                                                                | Mean Accuracy                                                                                        |
| :--------------------------------------------------------------------------- | :-------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_08.png"  alt="" />         | <img src="assets/PA-QuantumCicrcuit_heart-failure_plot_09.png"  alt="" />         | training: `0.6694560669456067`<br />testing: `0.7166666666666667`<br />overall: `0.6789297658862876` |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_5layers_08.png"  alt="" /> | <img src="assets/PA-QuantumCicrcuit_heart-failure_5layers_plot_09.png"  alt="" /> | training: `0.6694560669456067`<br />testing: `0.7166666666666667`<br />overall: `0.6789297658862876` |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_10layers_08.png"  alt="" /> | <img src="assets/PA-QuantumCicrcuit_heart-failure_10layers_plot_09.png"  alt="" /> | processing...                      |

**Ugate and cx instead of ry and cry**

| Quantum Circuit                                                              | Loss function plot                                                                | Mean Accuracy                                                                                        |
| :--------------------------------------------------------------------------- | :-------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_09.png"  alt="" />         | <img src="assets/PA-QuantumCicrcuit_heart-failure_plot_10.png"  alt="" />         | training: `0.6694560669456067`<br />testing: `0.7166666666666667`<br />overall: `0.6789297658862876` |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_5layers_09.png"  alt="" /> | <img src="assets/PA-QuantumCicrcuit_heart-failure_5layers_plot_10.png"  alt="" /> | training: `0.6820083682008368`<br />testing: `0.7166666666666667`<br />overall: `0.6956521739130435` |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_10layers_09.png"  alt="" /> | <img src="assets/PA-QuantumCicrcuit_heart-failure_10layers_plot_10.png"  alt="" /> | processing...                      |

**Only features and two weights**

| Quantum Circuit                                                              | Loss function plot                                                                | Mean Accuracy                                                                         |
| :--------------------------------------------------------------------------- | :-------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------ |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_10.png"  alt="" />         | <img src="assets/PA-QuantumCicrcuit_heart-failure_plot_11.png"  alt="" />         | training: `0.6736401673640168`<br />testing: `0.7`<br />overall: `0.6822742474916388` |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_5layers_10.png"  alt="" /> | <img src="assets/PA-QuantumCicrcuit_heart-failure_5layers_plot_11.png"  alt="" /> | training: `0.6778242677824268`<br />testing: `0.7`<br />overall: `0.6822742474916388` |
| <img src="assets/PA-QuantumCicrcuit_heart-failure_10layers_10.png"  alt="" /> | <img src="assets/PA-QuantumCicrcuit_heart-failure_10layers_plot_11.png"  alt="" /> | processing...                      |



## Pennylane Costs


| Classic = MLPClassifier                                                                                                               | Quantum Hybrid with Pennylane                                                                                                                                                                                        |
| :------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <img src="images/heartFailure_classic_01.png" alt="" /><br />Hidden layer size: 2, Layers 15<br />Max. 300 iterations - solver 'Adam' | <img src="images/heartFailure_quantum_pa_circuit_single.png" alt="" /><br />heartFailure_quantum_pa_circuit_single<br />25 iterations - Overall cost: 0.7494543 \| Acc train: 0.6875000 \| Acc validation: 0.6666667 |
| <img src="images/heartFailure_classic_02.png" alt="" /><br />Hidden layer size: 2, Layers 15<br />First 20 iterations - solver 'Adam' | <img src="images/heartFailure_quantum_pa_circuit.png" alt="" /><br />heartFailure_quantum_pa_circuit<br />75 iterations - Overall cost: 0.6581579 \| Acc train: 0.8392857 \| Acc validation: 0.8133333               |
| "                                                                                                                                     | <img src="images/heartFailure_quantum_repeated.png" alt="" /><br />heartFailure_quantum_repeated<br />30 iterations - Overall cost: 0.8352485 \| Acc train: 0.7053571 \| Acc validation: 0.6533333                   |
| "                                                                                                                                     | <img src="images/heartFailure_quantum.png" alt="" /><br />heartFailure_quantum<br />50 iterations - Overall cost: 0.9220091 \| Acc train: 0.6428571 \| Acc validation: 0.6133333                                     |
