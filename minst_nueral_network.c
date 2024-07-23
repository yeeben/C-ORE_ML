#include <stdio.h> 
#include <stdlib.h> 
#include "kaggleDatasetReader.h"

static GreyScaleImage imageDataset[KAGGLE_DATASET_MAX_SIZE + 1] = {};
static uint8_t labelDataset[KAGGLE_DATASET_MAX_SIZE + 1] = {};


int main(int argc, char *argv[]) {


    // uint32_t imageDatasetCount = 0;
    // uint32_t labelDatasetCount = 0;
    // printf("\n");
    // printf("Starting Reading Process\n");


    // if (argc < 3) {
    //     fprintf(stderr, "Usage: %s <filename>  <filename>\n", argv[0]);
    //     return 1;
    // }


    // loadImagesData(&imageDataset, argv[1], &imageDatasetCount);
    // loadLabelsData(&labelDataset, argv[2], &labelDatasetCount);

    // if (imageDatasetCount != labelDatasetCount) {
    //     fprintf(stderr, "Image and label dataset sizes do not match: %d != %d\n", imageDatasetCount, labelDatasetCount);
    //     return 1;
    // }

    test_func();

    return 0;
}


