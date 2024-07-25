#ifndef NUERAL_BUIDLING_BLOCKS_H_
#define NUERAL_BUIDLING_BLOCKS_H_

#include <stdint.h>
#include <kaggleDatasetReader.h>

#define KAGGLE_OUTPUT_LABELS 10

typedef struct {
    float bias[KAGGLE_OUTPUT_LABELS];
    float weights[KAGGLE_OUTPUT_LABELS][KAGGLE_IMAGE_SIZE];
} NetworkLayer_t;


float relu(float x);
void init_params(NetworkLayer_t *layer);

#endif