var Matrix = require("./matrix")
class NeuralNetwork {
    constructor(input_nodes, hidden_nodes, output_nodes){
        this.input_nodes = input_nodes
        this.hidden_nodes = hidden_nodes
        this.output_nodes = output_nodes
        this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes)
        this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes)
        this.weights_ih.randomize()
        this.weights_ho.randomize()
        this.bias_h = new Matrix(this.hidden_nodes, 1)
        this.bias_o = new Matrix(this.output_nodes, 1)
        this.bias_h.randomize()
        this.bias_o.randomize()
        this.learning_rate = 0.1
    }

    predict(input_array) {

        //Generating the hidden outputs
        let inputs = Matrix.fromArray(input_array)
        let hiddens = Matrix.multiply(this.weights_ih, inputs)
        hiddens.add(this.bias_h)
        //Activation function
        hiddens.map(this.sigmoid)  

        //Generating output's output
        let outputs = Matrix.multiply(this.weights_ho, hiddens)
        outputs.add(this.bias_o)
        //Activation function
        outputs.map(this.sigmoid)

        return outputs.toArray()
    }

    train(inputs_array, targets_array) {
        //Generating the hidden outputs
        let inputs = Matrix.fromArray(inputs_array)
        // inputs.print()
        
        let hiddens = Matrix.multiply(this.weights_ih, inputs)
        hiddens.add(this.bias_h)
        //Activation function
        hiddens.map(this.sigmoid)      
        // hiddens.print()

        //Generating output's output
        let outputs = Matrix.multiply(this.weights_ho, hiddens)
        outputs.add(this.bias_o)
        //Activation function
        outputs.map(this.sigmoid)
        // outputs.print()

        //Convert array to matrix
        let targets = Matrix.fromArray(targets_array)
        // targets.print()

        //==Calculate the error==
        //ERROR = TARGET-OUTPUT
        let output_errors = Matrix.subtract(targets, outputs)
        // output_errors.print()

        //Calculate gradient
        let gradients = Matrix.map(outputs, this.dsigmoid)
        // gradients.print()
        gradients.multiply(output_errors)
        // gradients.print()
        gradients.multiply(this.learning_rate)

        //Calculate deltas
        let hiddens_T = Matrix.transpose(hiddens)
        let weight_ho_deltas = Matrix.multiply(gradients, hiddens_T)

        this.weights_ho.add(weight_ho_deltas)
        this.bias_o.add(gradients)
        
        //==Calculate the hidden layout error==
        let who_t = Matrix.transpose(this.weights_ho)
        let hidden_errors = Matrix.multiply(who_t, output_errors)

        //Calculate hidden gradient
        let hidden_gradients = Matrix.map(hiddens, this.dsigmoid)
        hidden_gradients.multiply(hidden_errors)
        hidden_gradients.multiply(this.learning_rate)

        //Calculate input -> hidden deltas
        let input_T = Matrix.transpose(inputs)
        let weight_ih_deltas = Matrix.multiply(hidden_gradients, input_T)

        this.weights_ih.add(weight_ih_deltas)
        this.bias_h.add(hidden_gradients)
    }

    sigmoid(x) {
        return 1/(1 + Math.exp(-x))
    }

    dsigmoid(x) {
        return x*(1-x)
    }
}

module.exports = NeuralNetwork