#include "network.h"

// training data for XOR
double training_inputs[4][NUMBER_OF_INPUTS] = {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}
};

double training_outputs[4] = {0, 1, 1, 0};

double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

double sigmoid_derivative(double z) {
    return z * (1.0 - z);
}

void initialize_network(neural_network_t* n) {
    srand(time(NULL));

    // input to hidden
    for (int i = 0; i < NUMBER_OF_HIDDEN; i++) {
        for (int j = 0; j < NUMBER_OF_INPUTS; j++) {
            n->weights_input_hidden[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
        n->bias_hidden[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }

    // hidden to output
    for (int i = 0; i < NUMBER_OF_OUTPUTS; i++) {
        for (int j = 0; j < NUMBER_OF_HIDDEN; j++) {
            n->weights_hidden_output[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
        n->bias_output[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }
}

double forward(neural_network_t* n, double inputs[NUMBER_OF_INPUTS], double hidden_out[NUMBER_OF_HIDDEN]) {
    // hidden layer
    for (int i = 0; i < NUMBER_OF_HIDDEN; i++) {
        double sum = 0;
        for (int j = 0; j < NUMBER_OF_INPUTS; j++) {
            sum += inputs[j] * n->weights_input_hidden[i][j];
        }
        sum += n->bias_hidden[i];
        hidden_out[i] = sigmoid(sum);
    }

    // output layer
    double output_sum = 0;
    for (int j = 0; j < NUMBER_OF_HIDDEN; j++) {
        output_sum += hidden_out[j] * n->weights_hidden_output[0][j];
    }
    output_sum += n->bias_output[0];
    return sigmoid(output_sum);
}


void train(neural_network_t* n) {
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = 0; i < 4; i++) {
            double hidden_out[NUMBER_OF_HIDDEN];
            double output = forward(n, training_inputs[i], hidden_out);

            double error = training_outputs[i] - output;

            double delta_output = error * sigmoid_derivative(output);

            // hidden deltas
            double delta_hidden[NUMBER_OF_HIDDEN];
            for (int h = 0; h < NUMBER_OF_HIDDEN; h++) {
                delta_hidden[h] = delta_output * n->weights_hidden_output[0][h] * sigmoid_derivative(hidden_out[h]);
            }

            // update weights from hidden to output
            for (int h = 0; h < NUMBER_OF_HIDDEN; h++) {
                n->weights_hidden_output[0][h] += LEARNING_RATE * delta_output * hidden_out[h];
            }
            n->bias_output[0] += LEARNING_RATE * delta_output;

            // update weigths from input to hidden
            for (int h = 0; h < NUMBER_OF_HIDDEN; h++) {
                for (int j = 0; j < NUMBER_OF_INPUTS; j++) {
                    n->weights_input_hidden[h][j] += LEARNING_RATE * delta_hidden[h] * training_inputs[i][j];
                }
                n->bias_hidden[h] += LEARNING_RATE * delta_hidden[h];
            }
        }
    }
}


void test(neural_network_t n) {
    printf("*Testing the neural network:*\n");
    for (int i = 0; i < 4; i++) {
        double hidden_out[NUMBER_OF_HIDDEN];
        double prediction = forward(&n, training_inputs[i], hidden_out);
        printf("Input: [%.0f, %.0f], Expected: %.0f, Predicted: %.4f\n",
               training_inputs[i][0], training_inputs[i][1], training_outputs[i], prediction);
    }
}

