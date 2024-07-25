#ifndef KAGGLE_DATASET_READER_H_
#define KAGGLE_DATASET_READER_H_

#include <stdint.h>

#define IMAGES_MAGIC_NUMBER 2051
#define LABEL_MAGIC_NUMBER 2049
#define KAGGLE_IMAGE_WIDTH 28
#define KAGGLE_IMAGE_HEIGHT 28
#define KAGGLE_IMAGE_SIZE KAGGLE_IMAGE_WIDTH * KAGGLE_IMAGE_HEIGHT

typedef struct {
    uint32_t magic;
    uint32_t size;
    uint32_t rows;
    uint32_t columns;
} ImageHeader_t;

typedef struct {
    uint32_t magic;
    uint32_t size;
} LabelHeader_t;

typedef struct {
    uint8_t pixels[KAGGLE_IMAGE_SIZE];
} KaggleImage_t;

KaggleImage_t *loadImagesData(const char *filename, uint32_t* imageDatasetCount);   
uint8_t *loadLabelsData(const char *filename, uint32_t* imageDatasetCount);
void outFileSampleImage(KaggleImage_t *images, uint8_t *labels, uint32_t imageDatasetCount);

#endif