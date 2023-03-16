class Neuron {
    constructor(bias) {
        this.bias = bias;
        this.weights = [];
    }

    calculate_output(inputs) {
        this.inputs = inputs;
        this.output = this.squash(this.calculate_total_net_input());

        return this.output;
    }

    calculate_total_net_input() {
        let total = 0;
        for (let i = 0; i < this.inputs.length; i++){
            total += this.inputs[i] * this.weights[i];
        }
        return total + this.bias;
    }

    // Apply the logistic function to squash the output of the neuron
    // The result is sometimes referred to as 'net' [2] or 'net' [1]
    squash(total_net_input) {
        return 1 / (1 + Math.exp(-total_net_input));
    }

    // Determine how much the neuron's total input has to change to move closer to the expected output
    // Now that we have the partial derivative of the error with respect to the output (∂E/∂yⱼ) and
    // the derivative of the output with respect to the total net input (dyⱼ/dzⱼ) we can calculate
    // the partial derivative of the error with respect to the total net input.
    // This value is also known as the delta (δ) [1]
    // δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
    calculate_pd_error_wrt_total_net_input(target_output) {
        return this.calculate_pd_error_wrt_output(target_output) * this.calculate_pd_total_net_input_wrt_input();
    }

    // The error for each neuron is calculated by the Mean Square Error method:
    calculate_error(target_output) {
        return 0.5 * Math.pow(target_output - this.output, 2);
    }

    // The partial derivate of the error with respect to actual output then is calculated by:
    // = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1
    // = -(target output - actual output)
    // The Wikipedia article on backpropagation [1] simplifies to the following, but most other learning material does not [2]
    // = actual output - target output
    // Alternative, you can use (target - output), but then need to add it during backpropagation [3]
    // Note that the actual output of the output neuron is often written as yⱼ and target output as tⱼ so:
    // = ∂E/∂yⱼ = -(tⱼ - yⱼ)
    calculate_pd_error_wrt_output(target_output) {
        return -(target_output - this.output);
    }

    // The total net input into the neuron is squashed using logistic function to calculate the neuron's output:
    // yⱼ = φ = 1 / (1 + e^(-zⱼ))
    // Note that where ⱼ represents the output of the neurons in whatever layer we're looking at and ᵢ represents the layer below it
    // The derivative (not partial derivative since there is only one variable) of the output then is:
    // dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
    calculate_pd_total_net_input_wrt_input() {
        return this.output * (1 - this.output);
    }

    // The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
    // = zⱼ = netⱼ = x₁w₁ + x₂w₂ ...
    // The partial derivative of the total net input with respective to a given weight (with everything else held constant) then is:
    // = ∂zⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ
    calculate_pd_total_net_input_wrt_weight(index) {
        return this.inputs[index];
    }
}

module.exports = Neuron;