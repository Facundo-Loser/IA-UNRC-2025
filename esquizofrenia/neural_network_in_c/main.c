#include <stdio.h>
#include "network.h"

neural_network_t n;

int main(void) {
    initialize_network(&n);
    train(&n);
    test(n);
    return 0;
}
