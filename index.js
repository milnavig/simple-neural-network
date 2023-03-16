const NeuralNetwork = require('./NeuralNetwork');

let nn = new NeuralNetwork(2, 2, 2, [0.15, 0.2, 0.25, 0.3], 0.35, [0.4, 0.45, 0.5, 0.55], 0.6);

for (let i = 0; i < 10000; i++) {
    nn.train([0.05, 0.1], [0.01, 0.99]);
    console.log(`${i} ${nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]])}`);
}