#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "nueralBuildingBlocks.h"
#include "kaggleDatasetReader.h"

#define BATCH_SIZE 5


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
    float expSum = 0;
    for(int i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {

        expSum += exp(A[i]);
    }
    for(int i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
        A[i] = exp(A[i]) / expSum;
    }
}

static void init_params(NetworkLayer_t *layer, HiddenLayer_t *hiddenLayer) {
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

uint8_t get_predictions(float *A2) {
    uint8_t idx = 0;
    for(uint8_t i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
        if(A2[i] > A2[idx]) {
            idx = i;
        }
    }
    return idx;
}

static void forward_prop(NetworkLayer_t *layer, 
                    HiddenLayer_t *hiddenLayer, 
                    float *A1,
                    float *A2,
                    KaggleImage_t image) {
    int i, j;

    for(j = 0; j < KAGGLE_OUTPUT_LABELS; j++) {
        for (i = 0; i < KAGGLE_IMAGE_SIZE; i++) {
            A1[j] += layer->weights[j][i] * (image.pixels[i]/ 255.0);
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

}


static float backward_prop(float *A2, KaggleImage_t singleImage, uint8_t singleLabel, uint32_t dataset_size) {



    return 0.0f - log(A2[singleLabel]);
}


void gradient_descent(KaggleImage_t *pImages, uint8_t *pLabels, uint32_t dataset_size, float alpha, uint32_t epochs) {
    NetworkLayer_t *inputLayer = (NetworkLayer_t *)calloc(1, sizeof(NetworkLayer_t));
    HiddenLayer_t *hiddenLayer = (HiddenLayer_t *)calloc(1, sizeof(HiddenLayer_t));
    NetworkLayerGradient_t *inputGrad = (NetworkLayerGradient_t *)calloc(1, sizeof(NetworkLayerGradient_t));
    HiddenLayerGradient_t *hiddenGrad = (HiddenLayerGradient_t *)calloc(1, sizeof(HiddenLayerGradient_t));


    float A1[KAGGLE_OUTPUT_LABELS] = {0};
    float A2[KAGGLE_OUTPUT_LABELS] = {0};

    float totalLoss = 0;

    memset(inputGrad, 0, sizeof(NetworkLayerGradient_t));
    memset(hiddenGrad, 0, sizeof(HiddenLayerGradient_t));
    init_params(inputLayer, hiddenLayer);


    for(int i =0; i < BATCH_SIZE; i++) {
        forward_prop(inputLayer, hiddenLayer, A1, A2, pImages[i]);
        totalLoss += backward_prop(A2, pImages[i], pLabels[i], BATCH_SIZE);

    }
    printf("Total Loss: %f\n", totalLoss);

    free(inputLayer);
    free(hiddenLayer);
    free(inputGrad);
    free(hiddenGrad);


}