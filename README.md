# Neural Network
## A library to create a neural network in JS

+ **Init:**
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

i: input value (an array)  
o: expect output value (an array)  

```
nn.predict(i)
```

i: input value (an array)  
result: output value (an array)