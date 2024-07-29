#ifndef NUERAL_BUIDLING_BLOCKS_H_
#define NUERAL_BUIDLING_BLOCKS_H_

#include <stdint.h>
#include <kaggleDatasetReader.h>

#define KAGGLE_OUTPUT_LABELS 10

typedef struct {
    float bias[KAGGLE_OUTPUT_LABELS];
    float weights[KAGGLE_OUTPUT_LABELS][KAGGLE_IMAGE_SIZE];
} NetworkLayer_t;

typedef struct {
    float bias[KAGGLE_OUTPUT_LABELS];
    float weights[KAGGLE_OUTPUT_LABELS][KAGGLE_OUTPUT_LABELS];
} HiddenLayer_t;

typedef struct {
    float bias_grad[KAGGLE_OUTPUT_LABELS];
    float weights_grad[KAGGLE_OUTPUT_LABELS][KAGGLE_IMAGE_SIZE];
} NetworkLayerGradient_t;

typedef struct {
    float bias_grad[KAGGLE_OUTPUT_LABELS];
    float weights_grad[KAGGLE_OUTPUT_LABELS][KAGGLE_OUTPUT_LABELS];
} HiddenLayerGradient_t;

void init_params(NetworkLayer_t *layer, HiddenLayer_t *hiddenLayer);

void gradient_descent(KaggleImageSubset_t *dataset, 
                        NetworkLayer_t *inputLayer,
                        HiddenLayer_t *hiddenLayer,
                        float learning_rate);

float calculate_accuracy(KaggleImageSubset_t *dataset, 
                        NetworkLayer_t *inputLayer,
                        HiddenLayer_t *hiddenLayer);

#endif