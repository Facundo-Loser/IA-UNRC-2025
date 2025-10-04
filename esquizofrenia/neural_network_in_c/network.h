#ifndef NETWORK_H
#define NETWORK_H

// this is a simple neural network to that can learn to solve XOR

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUMBER_OF_INPUTS 2
#define NUMBER_OF_HIDDEN 2
#define NUMBER_OF_OUTPUTS 1
#define LEARNING_RATE 0.1
#define EPOCHS 100000

// sigmoid activation function
double sigmoid(double z);

// derivative of the sigmoid function (for backpropagation)
double sigmoid_derivative(double z);

typedef struct neural_network_t {
    // weights from input layer to hidden layer
    double weights_input_hidden[NUMBER_OF_HIDDEN][NUMBER_OF_INPUTS];
    double bias_hidden[NUMBER_OF_HIDDEN];

    // weights from hidden layer to output layer
    double weights_hidden_output[NUMBER_OF_OUTPUTS][NUMBER_OF_HIDDEN];
    double bias_output[NUMBER_OF_OUTPUTS];
} neural_network_t;

void initialize_network(neural_network_t* n);

void train(neural_network_t* n);

void test(neural_network_t n);

#endif
