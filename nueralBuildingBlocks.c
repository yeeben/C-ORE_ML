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

static float deriv_relu(float x) {
    return x > 0 ? x : 0;
}

static float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

static float sigmoid_prime(float x) {
    return x * (1 - x);
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

void visualize_forward_prop(float *Z1, float *Z2, float *A1, float *A2) {
    for(int i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
        printf("%f -> %f -> %f -> %f ", Z1[i], A1[i], Z2[i], A2[i]);
        printf("\n");
    }
    printf("\n");
}


static void forward_prop(NetworkLayer_t *layer, 
                    HiddenLayer_t *hiddenLayer, 
                    float *A1,
                    float *A2,
                    float *Z1,
                    float *Z2,
                    KaggleImage_t image) {
    int i, j;

    for(j = 0; j < KAGGLE_OUTPUT_LABELS; j++) {
        for (i = 0; i < KAGGLE_IMAGE_SIZE; i++) {
            Z1[j] += layer->weights[j][i] * (image.pixels[i]/ 255.0);
        }
        Z1[j] += layer->bias[j];
    }

    // Activation Function
    for (i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
        A1[i] = relu(Z1[i]);
    }

    // Hidden Layer
    for(j = 0; j < KAGGLE_OUTPUT_LABELS; j++) {
        for (i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
            Z2[j] += hiddenLayer->weights[j][i] * Z1[i];
        }
        Z2[j] += hiddenLayer->bias[j];
    }
    for (i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
            A2[i] = sigmoid(Z2[i]);
    }
    visualize_forward_prop(Z1, Z2, A1, A2);

}

static float backward_prop_hidden(float *A1,
                                  float *A2,
                                uint8_t singleLabel,
                                HiddenLayerGradient_t *hiddenGrad) {
    int i,j = 0;
    float cost = 0;
    float b_grad, w_grad = 0;

    for(i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
        if(i == singleLabel) {
            cost = A2[i] - 1;
        } else {
            cost = A2[i];
        }

        for(j = 0; j < KAGGLE_OUTPUT_LABELS; j++) {
            w_grad = A1[j] * b_grad;
            hiddenGrad->weights_grad[i][j] += w_grad;
        }
        hiddenGrad->bias_grad[i] += b_grad;
    }

    return 0.0f - log(A2[singleLabel]);
}

static void backward_prop_input(float *A1,
                                  KaggleImage_t image,
                                  NetworkLayerGradient_t *inputGrad) {
    int i,j = 0;
    float b_grad, w_grad = 0;

    for(i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
        b_grad = deriv_relu(A1[i]);

        for(j = 0; j < KAGGLE_IMAGE_SIZE; j++) {
            w_grad = image.pixels[j] * b_grad;
            inputGrad->weights_grad[i][j] += w_grad;
        }
        inputGrad->bias_grad[i] += b_grad;
    }
}

static void update_params(  NetworkLayer_t *layer, 
                            HiddenLayer_t *hiddenLayer, 
                            NetworkLayerGradient_t *inputGrad, 
                            HiddenLayerGradient_t *hiddenGrad,
                            float learning_rate) {
    int i, j = 0;
    for(i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
        hiddenLayer->bias[i] -= learning_rate * hiddenGrad->bias_grad[i] / (float)BATCH_SIZE;

        for(j = 0; j < KAGGLE_OUTPUT_LABELS; j++) {
            hiddenLayer->weights[i][j] -= learning_rate * hiddenGrad->weights_grad[i][j] / (float)BATCH_SIZE;
        }
    }

    for(i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
        layer->bias[i] -= learning_rate * inputGrad->bias_grad[i] / (float)BATCH_SIZE;

        for(j = 0; j < KAGGLE_IMAGE_SIZE; j++) {
            layer->weights[i][j] -= learning_rate * inputGrad->weights_grad[i][j] / (float)BATCH_SIZE;
        }
    }
}

void gradient_descent(KaggleImage_t *pImages, uint8_t *pLabels, uint32_t dataset_size, float learning_rate, uint32_t epochs) {
    NetworkLayer_t *inputLayer = (NetworkLayer_t *)calloc(1, sizeof(NetworkLayer_t));
    HiddenLayer_t *hiddenLayer = (HiddenLayer_t *)calloc(1, sizeof(HiddenLayer_t));
    NetworkLayerGradient_t *inputGrad = (NetworkLayerGradient_t *)calloc(1, sizeof(NetworkLayerGradient_t));
    HiddenLayerGradient_t *hiddenGrad = (HiddenLayerGradient_t *)calloc(1, sizeof(HiddenLayerGradient_t));


    float A1[KAGGLE_OUTPUT_LABELS] = {0};
    float A2[KAGGLE_OUTPUT_LABELS] = {0};
    float Z1[KAGGLE_OUTPUT_LABELS] = {0};
    float Z2[KAGGLE_OUTPUT_LABELS] = {0};

    float totalLoss = 0;

    memset(inputGrad, 0, sizeof(NetworkLayerGradient_t));
    memset(hiddenGrad, 0, sizeof(HiddenLayerGradient_t));
    init_params(inputLayer, hiddenLayer);


    for(int i =0; i < BATCH_SIZE; i++) {
        forward_prop(inputLayer, hiddenLayer, A1, A2, Z1, Z2, pImages[i]);
        // totalLoss += backward_prop_hidden(A1, A2, pLabels[i], hiddenGrad);
        // backward_prop_input(A1, pImages[i], inputGrad);

    }

    // update_params(inputLayer, hiddenLayer, inputGrad, hiddenGrad, learning_rate);

    printf("Total Loss: %f\n", totalLoss);

    free(inputLayer);
    free(hiddenLayer);
    free(inputGrad);
    free(hiddenGrad);


}