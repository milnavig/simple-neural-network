const Neuron = require('./Neuron');

class NeuronLayer {
    constructor(num_neurons, bias) {
        // Every neuron in a layer shares the same bias
        this.bias = bias ?? Math.random();
        this.neurons = [];

        for (let i = 0; i < num_neurons; i++) {
            this.neurons.push(new Neuron(this.bias));
        }
    }

    inspect() {
        console.log(`Neurons: ${this.neurons.length}`);

        this.neurons.forEach((neuron, i) => {
            console.log(` Neuron ${i}`);

            neuron.weights.forEach(weight => {
                console.log(`  Weight: ${weight}`);
            });

            console.log(`  Bias: ${this.bias}`);
        });
    }

    feed_forward(inputs) {
        let outputs = [];

        for (let neuron of this.neurons) {
            outputs.push(neuron.calculate_output(inputs));
        }

        return outputs;
    }

    get_outputs() {
        let outputs = [];
        for (let neuron of this.neurons) {
            outputs.push(neuron.output)
        }
        return outputs;
    }
}

module.exports = NeuronLayer;