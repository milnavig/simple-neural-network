const NeuronLayer = require('./NeuronLayer');

class NeuralNetwork {
    constructor(num_inputs, num_hidden, num_outputs, hidden_layer_weights = null, hidden_layer_bias = null, output_layer_weights = null, output_layer_bias = null) {
        this.LEARNING_RATE = 0.5;

        this.num_inputs = num_inputs;

        this.hidden_layer = new NeuronLayer(num_hidden, hidden_layer_bias);
        this.output_layer = new NeuronLayer(num_outputs, output_layer_bias);

        this.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights);
        this.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights);
    }

    init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights) {
        let weight_num = 0;

        for (let neuron of this.hidden_layer.neurons) {
            for (let i = 0; i < this.num_inputs; i++) {
                if (!hidden_layer_weights) {
                    neuron.weights.push(Math.random());
                } else {
                    neuron.weights.push(hidden_layer_weights[weight_num]);
                }
                weight_num++;
            }
        }
    }

    init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights) {
        let weight_num = 0;

        for (let neuron of this.output_layer.neurons) {
            for (let i = 0; i < this.hidden_layer.neurons.length; i++) {
                if (!output_layer_weights) {
                    neuron.weights.push(Math.random());
                } else {
                    neuron.weights.push(output_layer_weights[weight_num]);
                }
                weight_num++;
            }
        }
    }

    inspect() {
        console.log('------');
        console.log(`* Inputs: ${this.num_inputs}`);
        console.log('------');
        console.log('Hidden Layer');
        this.hidden_layer.inspect();
        console.log('------');
        console.log('* Output Layer');
        this.output_layer.inspect();
        console.log('------');
    }


    feed_forward(inputs) {
        let hidden_layer_outputs = this.hidden_layer.feed_forward(inputs);
        return this.output_layer.feed_forward(hidden_layer_outputs);
    }

    train(training_inputs, training_outputs) {
        this.feed_forward(training_inputs);

        // 1. Output neuron deltas
        let pd_errors_wrt_output_neuron_total_net_input = new Array(this.output_layer.neurons.length).fill(0);

        for (let o = 0; o < this.output_layer.neurons.length; o++) {
            // ∂E/∂zⱼ
            pd_errors_wrt_output_neuron_total_net_input[o] = this.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o]);
        }

        // 2. Hidden neuron deltas
        let pd_errors_wrt_hidden_neuron_total_net_input = new Array(this.hidden_layer.neurons.length).fill(0);
        for (let h = 0; h < this.hidden_layer.neurons.length; h++) {
            // We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
            // dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
            let d_error_wrt_hidden_neuron_output = 0;
            for (let o = 0; o < this.output_layer.neurons.length; o++) {
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * this.output_layer.neurons[o].weights[h];
            }

            // ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * this.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input();
        }

        // 3. Update output neuron weights
        for (let o = 0; o < this.output_layer.neurons.length; o++) {
            for (let w_ho = 0; w_ho < this.output_layer.neurons[o].weights; w_ho++) {
                // ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                let pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * this.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho);
                // Δw = α * ∂Eⱼ/∂wᵢ
                this.output_layer.neurons[o].weights[w_ho] -= this.LEARNING_RATE * pd_error_wrt_weight;
            }
        }

        // 4. Update hidden neuron weights
        for (let h = 0; h < this.hidden_layer.neurons.length; h++) {
            for (let w_ih = 0; w_ih < this.hidden_layer.neurons[h].weights; w_ih++) {
                // ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                let pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * this.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih);
                // Δw = α * ∂Eⱼ/∂wᵢ
                this.hidden_layer.neurons[h].weights[w_ih] -= this.LEARNING_RATE * pd_error_wrt_weight;
            }
        }
    }

    calculate_total_error(training_sets) {
        let total_error = 0;
        for (let t = 0; t < training_sets.length; t++) {
            let [training_inputs, training_outputs] = training_sets[t];
            this.feed_forward(training_inputs);

            for (let o = 0; o < training_outputs.length; o++) {
                total_error += this.output_layer.neurons[o].calculate_error(training_outputs[o]);
            }
        }
        return total_error;
    }
}

module.exports = NeuralNetwork;