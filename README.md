# Neural Network
## A library to create a neural network in JS

+ **Initiate:**
```
var nn = new NeuralNetwork(inputs, hiddens, outputs)  
```

inputs: Number of input values  
hiddens: Number of neural cells in hidden layer  
output: Number of output values  

+ **Usage**:
```
nn.train(i, o)
```

i: Input value (an array)  
o: Expect output value (an array)  

```
nn.predict(i)
```

i: Input value (an array)  
result: Output value (an array)

+ **RESULT**:  
This is a result when using this library to resolve XOR problem with NeuralNetwork(2, 2, 1):  
![alt text](https://github.com/trandinhquy97/neuralnetwork/blob/master/images/result.PNG?raw=true)
