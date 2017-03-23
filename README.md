# Sport-Analytic-NN
Neural Network Realization of Project Sport Analytic

## Dataset
* issue 1 **The coordinates x and y axis data is strange **


## Experiments

### Simple Neural Network TD-prediction

### Recurrent Neural Network TD-prediction
* issue 1 **problems: NN Network doesn't converge**
* Solving issue 1 
    1. Add a dense layer over the output of network.
    2. **[fail]** Coordinate some parameters like "keep-Probability" or "Number of LSTM Layers".
    3. Try to balance the data, for example number of 0 and 1 of target y
    4. Coordinate the x and y axis of network
    5. Try to focus on the difference of V between two team. (set reward to be goal difference?)
    6. Try batch Normalization
    7. reduce the length of trace data
*

### Evaluation Methods
