#include <stdint.h>
#include <stdio.h> 
#include <stdlib.h> 
#include "nueralBuildingBlocks.h"


// void init_params(NueralLayer *layer1, uint16_t layer1Size, NueralLayer *layer2, uint16_t layer2Size) {
//     for(int i = 0; i < layer1Size; i++) {
//         layer1[i].weights = ((float)rand() / (float)RAND_MAX ) - 0.5;
//         layer1[i].bias = ((float)rand() / (float)RAND_MAX ) - 0.5;
//     }
    
//     for(int i = 0; i < layer2Size; i++) {
//         layer2[i].weights = ((float)rand() / (float)RAND_MAX ) - 0.5;
//         layer2[i].bias = ((float)rand() / (float)RAND_MAX ) - 0.5;
//     }
// }


uint32_t relu(uint32_t x) {
    if (x > 0) {
        return x;
    } else {
        return 0;
    }
}
