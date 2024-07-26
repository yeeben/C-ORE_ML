#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "kaggleDatasetReader.h"

uint32_t swapEndian(uint32_t value) {
    return ((value >> 24) & 0x000000FF) |
           ((value >> 8) & 0x0000FF00) |
           ((value << 8) & 0x00FF0000) |
           ((value << 24) & 0xFF000000);
}

KaggleImage_t *loadImagesData(const char *filename, uint32_t *imageDatasetSize) {    
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    ImageHeader_t header = {};

    if (fread(&header, 1, sizeof(ImageHeader_t), file) != sizeof(ImageHeader_t)) {
        perror("Error reading file");
        fclose(file);
        return NULL;
    }

    // Endianess
    header.magic = swapEndian(header.magic);
    header.size = swapEndian(header.size);
    header.rows = swapEndian(header.rows);
    header.columns = swapEndian(header.columns);

    printf("Magic: %d \t Label Size: %d\n", header.magic, header.size);
    *imageDatasetSize = header.size;
    if (header.magic != IMAGES_MAGIC_NUMBER) {
        fprintf(stderr, "Invalid magic number\n");
        fclose(file);
        return NULL;
    }

    KaggleImage_t *images = (KaggleImage_t *) calloc(header.size, sizeof(KaggleImage_t));
    if (images == NULL) {
        perror("Error allocating memory");
        fclose(file);
        return NULL;
    }

    if (fread(images, sizeof(KaggleImage_t), header.size, file) != header.size) {
        perror("Error reading file");
        fclose(file);
        free(images);
        return NULL;
    }

    return images;

}

uint8_t *loadLabelsData(const char *filename, uint32_t *labelDatasetSize) {
    
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    LabelHeader_t header = {};
    
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    if (fread(&header, 1, sizeof(LabelHeader_t), file) != sizeof(LabelHeader_t)) {
        perror("Error reading file");
        fclose(file);
        return NULL;
    }
    // Endianess
    header.magic = swapEndian(header.magic);
    header.size = swapEndian(header.size);

    printf("Magic: %d \t Label Size: %d\n", header.magic, header.size);
    *labelDatasetSize = header.size;

    if (header.magic != LABEL_MAGIC_NUMBER) {
        fprintf(stderr, "Invalid magic number\n");
        fclose(file);
        return NULL;
    }

    uint8_t *labels = (uint8_t *) calloc(header.size, sizeof(uint8_t));

    if (labels == NULL) {
        perror("Error allocating memory");
        fclose(file);
        return NULL;
    }

    if (fread(labels, sizeof(uint8_t), header.size, file) != header.size) {
        perror("Error reading file");
        fclose(file);
        free(labels);
        return NULL;
    }

    return labels;
}

void outFileSampleImage(KaggleImageSubset_t *pImageSubset, uint32_t currentIdx) {

    char outputFilename[50];
    uint8_t sampledLabel = 0;
    for(int i =0; i < 3; i++) {
        sampledLabel = pImageSubset->labels[i];
        snprintf(outputFilename, sizeof(outputFilename), "outFile/sample%d_idx%d.png", sampledLabel, currentIdx);

        FILE *outFile = fopen(outputFilename, "wb");
        if (outFile == NULL) {
            perror("Error opening output file");
            return;
        }
        fwrite(&pImageSubset->images[i], 1, KAGGLE_IMAGE_HEIGHT * KAGGLE_IMAGE_WIDTH, outFile);
        fclose(outFile);
    }
}
