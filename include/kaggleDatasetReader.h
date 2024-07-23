#include <stdint.h>

#define IMAGES_MAGIC_NUMBER 2051
#define LABEL_MAGIC_NUMBER 2049

#define BITS_IN_BYTE 8
#define IMAGE_HEADER_BUFFER_SIZE 16
#define LABEL_HEADER_BUFFER_SIZE 8

#define KAGGLE_IMAGE_HEIGHT 28
#define KAGGLE_IMAGE_WIDTH 28
#define KAGGLE_DATASET_MAX_SIZE 60000
#define KAGGLE_DATASET_OUTPUT_LAYER 10

typedef struct {
    uint32_t magic;
    uint32_t size;
    uint32_t rows;
    uint32_t columns;
} ImageHeader;

typedef struct {
    uint32_t magic;
    uint32_t size;
} LabelHeader;

typedef struct {
    uint8_t pixels[KAGGLE_IMAGE_HEIGHT][KAGGLE_IMAGE_WIDTH];
} GreyScaleImage;

void loadImagesData(GreyScaleImage *imageDataset[KAGGLE_DATASET_MAX_SIZE], const char *filename, uint32_t* imageDatasetCount);    
void loadLabelsData(uint8_t *labelDataset[KAGGLE_DATASET_MAX_SIZE], const char *filename, uint32_t* imageDatasetCount);

void test_func();