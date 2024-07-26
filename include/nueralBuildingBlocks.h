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

void gradient_descent(KaggleImage_t *pImages, uint8_t *pLabels, uint32_t dataset_size, float learning_rate, uint32_t epochs);

#endif