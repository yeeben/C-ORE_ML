#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nueralBuildingBlocks.h"
#include "kaggleDatasetReader.h"

float get_rand_float() {
    return (((float) rand()) / ((float) RAND_MAX)) - 0.5;
}

float relu(float x) {
    if (x > 0) {
        return x;
    } else {
        return 0;
    }
}

void init_params(NetworkLayer_t *layer) {
    for (int i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
        layer->bias[i] = get_rand_float();
        for (int j = 0; j < KAGGLE_IMAGE_SIZE; j++) {
            layer->weights[i][j] = get_rand_float();
        }
    }
}

