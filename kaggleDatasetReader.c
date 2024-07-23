#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "kaggleDatasetReader.h"

void readImagesHeader(FILE* fp, uint32_t* imageDatasetCount) {
    ImageHeader header = {};
    uint8_t headerReadBuffer[IMAGE_HEADER_BUFFER_SIZE] = {};

    if (fp == NULL) {
        perror("Error opening file");
        return;
    }

    if (fread(headerReadBuffer, 1, IMAGE_HEADER_BUFFER_SIZE, fp) != IMAGE_HEADER_BUFFER_SIZE) {
        perror("Error reading file");
        fclose(fp);
        return;
    }
    header.magic = (headerReadBuffer[0] << 24) | (headerReadBuffer[1] << 16) | (headerReadBuffer[2] << 8) | headerReadBuffer[3];
    header.size = (headerReadBuffer[4] << 24) | (headerReadBuffer[5] << 16) | (headerReadBuffer[6] << 8) | headerReadBuffer[7];
    header.rows = (headerReadBuffer[8] << 24) | (headerReadBuffer[9] << 16) | (headerReadBuffer[10] << 8) | headerReadBuffer[11];
    header.columns = (headerReadBuffer[12] << 24) | (headerReadBuffer[13] << 16) | (headerReadBuffer[14] << 8) | headerReadBuffer[15];

    if (header.magic != IMAGES_MAGIC_NUMBER) {
        fprintf(stderr, "Invalid images magic number: %d\n", header.magic);
        fclose(fp);
        return;
    }

    printf("Parsed Image File Header: magic=%d, size=%d, rows=%d, columns=%d\n", header.magic, header.size, header.rows, header.columns);
    *imageDatasetCount = header.size;
}

void readLabelsHeader(FILE* fp, uint32_t *labelDatasetCount) {
    LabelHeader header = {};
    
    if (fp == NULL) {
        perror("Error opening file");
        return;
    }

    uint8_t headerReadBuffer[LABEL_HEADER_BUFFER_SIZE];
    if (fread(headerReadBuffer, 1, LABEL_HEADER_BUFFER_SIZE, fp) != LABEL_HEADER_BUFFER_SIZE) {
        perror("Error reading file");
        fclose(fp);
        return;
    }

    header.magic = (headerReadBuffer[0] << 24) | (headerReadBuffer[1] << 16) | (headerReadBuffer[2] << 8) | headerReadBuffer[3];
    header.size = (headerReadBuffer[4] << 24) | (headerReadBuffer[5] << 16) | (headerReadBuffer[6] << 8) | headerReadBuffer[7];
    if (header.magic != LABEL_MAGIC_NUMBER) {
        fprintf(stderr, "Invalid labels magic number: %d\n", header.magic);
        fclose(fp);
        return;
    }

    printf("Parsed Label File Header: magic=%d, size=%d\n", header.magic, header.size);
    *labelDatasetCount = header.size;
}

void loadImagesData(GreyScaleImage *imageDataset[KAGGLE_DATASET_MAX_SIZE], const char *filename, uint32_t* imageDatasetCount) {    
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    readImagesHeader(file, imageDatasetCount);

    for(int i = 0; i < *imageDatasetCount; i++) {
        fread(&imageDataset[i], 1, KAGGLE_IMAGE_HEIGHT * KAGGLE_IMAGE_WIDTH, file);
    }
    fclose(file);
    // I like to sample the images produced here to sanity check that my arrays have been loaded correctly
    // Using a python plotter to view the grey scaled value
}
void loadLabelsData(uint8_t *labelDataset[KAGGLE_DATASET_MAX_SIZE], const char *filename, uint32_t* imageDatasetCount) {
    
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    readLabelsHeader(file, imageDatasetCount);

    for(int i = 0; i < *imageDatasetCount; i++) {
        fread(&labelDataset[i], 1, sizeof(uint8_t), file);
    }
    fclose(file);
}

void test_func() {
    printf("Hello World\n");
}
