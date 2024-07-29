#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "nueralBuildingBlocks.h"
#include "kaggleDatasetReader.h"

NetworkLayerGradient_t inputGrad =  {};
HiddenLayerGradient_t hiddenGrad = {};


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

static float relu_prime(float x) {
    return x > 0 ? 1.0 : 0.0;
}

static float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

static float sigmoid_prime(float x) {
    return x * (1 - x);
}

static float softmax(float z, float sum) {
    return exp(z) / sum;
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

uint8_t get_predictions(float *A2) {
    uint8_t idx = 0;
    for(uint8_t i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
        if(A2[i] > A2[idx]) {
            idx = i;
        }
    }
    return idx;
}

void dump_gradients(NetworkLayerGradient_t *inputGrad, HiddenLayerGradient_t *hiddenGrad) {
    for(int i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
        printf("Bias Grad %d: %f ", i, inputGrad->bias_grad[i]);
        printf("Weights: \n");
        for(int j = 0; j < KAGGLE_IMAGE_SIZE; j++) {
            printf("%f ", inputGrad->weights_grad[i][j]);
        }
        printf("\n");
    }

    for(int i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
        printf("Bias Grad %d: %f ", i, hiddenGrad->bias_grad[i]);
        for(int j = 0; j < KAGGLE_OUTPUT_LABELS; j++) {
            printf("%f ", hiddenGrad->weights_grad[i][j]);
        }
        printf("\n");
    }
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
    memset(Z1, 0, sizeof(float) * KAGGLE_OUTPUT_LABELS);
    memset(Z2, 0, sizeof(float) * KAGGLE_OUTPUT_LABELS);
    memset(A1, 0, sizeof(float) * KAGGLE_OUTPUT_LABELS);
    memset(A2, 0, sizeof(float) * KAGGLE_OUTPUT_LABELS);

    for(i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
        for (j = 0; j < KAGGLE_IMAGE_SIZE; j++) {
            Z1[i] += layer->weights[i][j] * (image.pixels[j]/ 255.0);
        }
        Z1[i] += layer->bias[i];
    }

    // Activation Function
    for (i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
        A1[i] = relu(Z1[i]);
    }

    // Hidden Layer
    for(i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
        for (j = 0; j < KAGGLE_OUTPUT_LABELS; j++) {
            Z2[i] += hiddenLayer->weights[i][j] * Z1[j];
        }
        Z2[i] += hiddenLayer->bias[i];
    }
    // Sigmoid Activation Function
    // for (i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
    //         A2[i] = sigmoid(Z2[i]);
    // }

    float sum = 0;

    for(i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
        A2[i] = exp(Z2[i]);
        sum += A2[i];
    }

    for (i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
        A2[i] /=  sum;
    }

    // visualize_forward_prop(Z1, Z2, A1, A2);

}

static float back_propogation(KaggleImage_t *image, 
                    uint8_t label,
                    HiddenLayer_t *hiddenLayer,
                    NetworkLayerGradient_t *inputGrad, 
                    HiddenLayerGradient_t *hiddenGrad,
                    float *A1,
                    float *A2,
                    float *Z1,
                    float *Z2
                    ) {
    int i,j = 0;
    float Y[KAGGLE_OUTPUT_LABELS] = {0}; // Oneshot of the label
    float dZ1[KAGGLE_OUTPUT_LABELS] = {0};
    float dZ2[KAGGLE_OUTPUT_LABELS] = {0};

    Y[label] = 1;
    // Partial Derivatives wrt to Z2
    // Partial Derivatives wrt to W2 and B2
    for(i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
        dZ2[i] = A2[i] - Y[i];

        for(j = 0; j < KAGGLE_OUTPUT_LABELS; j++) {
            hiddenGrad->weights_grad[i][j] += dZ2[i] * A1[j];
        }
        hiddenGrad->bias_grad[i] += dZ2[i];
    }

    // Partial Derivatives wrt to Z1
    for(i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
        for(j = 0; j < KAGGLE_OUTPUT_LABELS; j++) {
            dZ1[i] += dZ2[i] * hiddenLayer->weights[i][j] * relu_prime(Z1[j]);
        }
    }
    // Partial Derivatives wrt to W1 and B1
    for(i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
        for(j = 0; j < KAGGLE_IMAGE_SIZE; j++) {
            inputGrad->weights_grad[i][j] += dZ1[i] * (image->pixels[j] / 255.0);
        }
        inputGrad->bias_grad[i] += dZ1[i];
    }
    return 0.0f - log(A2[label]);
}


static void update_params(  NetworkLayer_t *layer, 
                            HiddenLayer_t *hiddenLayer, 
                            NetworkLayerGradient_t *inputGrad, 
                            HiddenLayerGradient_t *hiddenGrad,
                            float learning_rate,
                            uint32_t batch_size) {
    int i, j = 0;
    for(i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
        hiddenLayer->bias[i] -= learning_rate * hiddenGrad->bias_grad[i] / ((float)batch_size);

        for(j = 0; j < KAGGLE_OUTPUT_LABELS; j++) {
            hiddenLayer->weights[i][j] -= learning_rate * hiddenGrad->weights_grad[i][j] / ((float)batch_size);
        }
    }

    for(i = 0; i < KAGGLE_OUTPUT_LABELS; i++) {
        layer->bias[i] -= learning_rate * inputGrad->bias_grad[i] / ((float)batch_size);

        for(j = 0; j < KAGGLE_IMAGE_SIZE; j++) {
            layer->weights[i][j] -= learning_rate * inputGrad->weights_grad[i][j] / ((float)batch_size);
        }
    }
}

void gradient_descent(KaggleImageSubset_t *dataset, 
                        NetworkLayer_t *inputLayer,
                        HiddenLayer_t *hiddenLayer,
                        float learning_rate) {


    float A1[KAGGLE_OUTPUT_LABELS] = {0};
    float A2[KAGGLE_OUTPUT_LABELS] = {0};
    float Z1[KAGGLE_OUTPUT_LABELS] = {0};
    float Z2[KAGGLE_OUTPUT_LABELS] = {0};

    int i,j = 0;
    float total_loss = 0.0f;

    memset(&inputGrad, 0, sizeof(NetworkLayerGradient_t));
    memset(&hiddenGrad, 0, sizeof(HiddenLayerGradient_t));

    total_loss = 0;
    for(j =0; j < dataset->imageDatasetCount; j++) {
        forward_prop(inputLayer, hiddenLayer, A1, A2, Z1, Z2, dataset->images[j]);
        total_loss += back_propogation(&dataset->images[j], 
                                        dataset->labels[j], 
                                        hiddenLayer, 
                                        &inputGrad, 
                                        &hiddenGrad, 
                                        A1, A2, Z1, Z2);
    }
    printf("%f \t", total_loss/dataset->imageDatasetCount);

    update_params(inputLayer, hiddenLayer, &inputGrad, &hiddenGrad, learning_rate, dataset->imageDatasetCount);

}

float calculate_accuracy(KaggleImageSubset_t *dataset, 
                        NetworkLayer_t *inputLayer,
                        HiddenLayer_t *hiddenLayer) {
    float A1[KAGGLE_OUTPUT_LABELS] = {0};
    float A2[KAGGLE_OUTPUT_LABELS] = {0};
    float Z1[KAGGLE_OUTPUT_LABELS] = {0};
    float Z2[KAGGLE_OUTPUT_LABELS] = {0};
    uint32_t correct = 0;
    uint32_t total = 0;
    uint8_t prediction = 0;

    for(int i = 0; i < dataset->imageDatasetCount; i++) {
        forward_prop(inputLayer, hiddenLayer, A1, A2, Z1, Z2, dataset->images[i]);
        prediction = get_predictions(A2);
        if(prediction == dataset->labels[i]) {
            correct++;
        }
    }

    return (float)correct / (float)dataset->imageDatasetCount;
}