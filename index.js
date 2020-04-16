var NeuralNetwork = require('./neuralnetwork')

let training_data = [{
    inputs: [0, 0],
    targets:[0]
}, {
    inputs: [0, 1],
    targets:[1]
},{
    inputs: [1, 0],
    targets:[1]
},{
    inputs: [1, 1],
    targets:[0]
},]
let nn = new NeuralNetwork(2, 2, 1)

for (let i = 0; i < 50000; i++) {
    let item = training_data[Math.floor(Math.random()*training_data.length)]
    nn.train(item.inputs, item.targets)
}

console.log("0 XOR 0 = " + nn.predict([0, 0]))
console.log("0 XOR 1 = " + nn.predict([0, 1]))
console.log("1 XOR 0 = " + nn.predict([1, 0]))
console.log("1 XOR 1 = " + nn.predict([1, 1]))



