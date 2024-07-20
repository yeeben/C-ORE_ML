#include <stdint.h>
#include <stdio.h> 
#include <stdlib.h> 

#define MAX_INT_VALUE ((1 << 32) - 1)

void init_weights(float** weights, uint32_t rows, uint32_t columns) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            // bound each weight between -0.5 and 0.5
            weights[i][j] = ((float)rand() / (float)RAND_MAX ) - 0.5;
        }
    }
}

uint8_t relu(uint8_t x) {
    if (x > 0) {
        return x;
    } else {
        return 0;
    }
}
