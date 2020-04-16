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

console.log(nn.feedForward([0, 0]))
console.log(nn.feedForward([0, 1]))
console.log(nn.feedForward([1, 0]))
console.log(nn.feedForward([1, 1]))



