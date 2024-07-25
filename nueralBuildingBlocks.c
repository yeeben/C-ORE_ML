#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "nueralBuildingBlocks.h"
#include "kaggleDatasetReader.h"

static float get_rand_float() {
    return (((float) rand()) / ((float) RAND_MAX)) - 0.5;
}

static float relu(float x) {
    if (x > 0) {
        return x;
    } else {
        return 0;
    }
}

static void softmax(float *A) {
    float exp_sum = 0;
    for(int i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
        exp_sum += exp(A[i]);
    }
    for(int i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
        A[i] = exp(A[i]) / exp(exp_sum);
    }
}

void init_params(NetworkLayer_t *layer, HiddenLayer_t *hiddenLayer) {
    for (int i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
        layer->bias[i] = get_rand_float();
        for (int j = 0; j < KAGGLE_IMAGE_SIZE; j++) {
            layer->weights[i][j] = get_rand_float();
        }
    }

    for (int i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
        hiddenLayer->bias[i] = get_rand_float();
        for (int j = 0; j < KAGGLE_OUTPUT_LABELS; j++) {
            hiddenLayer->weights[i][j] = get_rand_float();
        }
    }
}

float *forward_prop(NetworkLayer_t *layer, HiddenLayer_t *hiddenLayer, KaggleImage_t *singleImage) {
    float A1[KAGGLE_OUTPUT_LABELS] = {0}; // Activation after Input Layer
    float *A2 = (float *)calloc(KAGGLE_OUTPUT_LABELS, sizeof(float)); // Activation after Output Layer

    int i, j;
    for(j = 0; j < KAGGLE_OUTPUT_LABELS; j++) {
        for (i = 0; i < KAGGLE_IMAGE_SIZE; i++) {
            A1[j] += layer->weights[j][i] * (singleImage->pixels[i]/ 255.0);
        }
        A1[j] += layer->bias[j];
    }

    // Activation Function
    for (i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
        A1[i] = relu(A1[i]);
    }
    
    // Hidden Layer
    for(j = 0; j < KAGGLE_OUTPUT_LABELS; j++) {
        for (i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
            A2[j] += hiddenLayer->weights[j][i] * A1[i];
        }
        A2[j] += hiddenLayer->bias[j];
    }
    // Activation Function
    softmax(A2);
    return A2;
}