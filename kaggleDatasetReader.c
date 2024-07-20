#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#define IMAGES_MAGIC_NUMBER 2051
#define LABEL_MAGIC_NUMBER 2049

#define BITS_IN_BYTE 8
#define IMAGE_HEADER_BUFFER_SIZE 16
#define LABEL_HEADER_BUFFER_SIZE 8

#define KAGGLE_IMAGE_HEIGHT 28
#define KAGGLE_IMAGE_WIDTH 28
#define KAGGLE_DATASET_MAX_SIZE 60000


struct ImageHeader {
    uint32_t magic;
    uint32_t size;
    uint32_t rows;
    uint32_t columns;
};

struct LabelHeader {
    uint32_t magic;
    uint32_t size;
};

struct GreyScaleImage {
    uint8_t pixels[KAGGLE_IMAGE_HEIGHT][KAGGLE_IMAGE_WIDTH];
};

static struct GreyScaleImage imageDataset[KAGGLE_DATASET_MAX_SIZE + 1] = {};
static uint8_t labelDataset[KAGGLE_DATASET_MAX_SIZE + 1] = {};


void readImagesHeader(FILE* fp, uint32_t* imageDatasetCount) {
    struct ImageHeader header = {};
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
    struct LabelHeader header = {};
    
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

void loadImagesData(const char *filename, uint32_t* imageDatasetCount) {    
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
void loadLabelsData(const char *filename, uint32_t* imageDatasetCount) {
    
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

void output_sample_images(uint32_t imageDatasetCount) {
    for(int i = 0; i < 6; i++) {

        char outputFilename[50];
        uint32_t sampledIdx = rand() % imageDatasetCount;
        uint8_t sampledLabel = labelDataset[sampledIdx];
        snprintf(outputFilename, sizeof(outputFilename), "outFile/sample%d_idx%d.png", sampledLabel, sampledIdx);

        FILE *outFile = fopen(outputFilename, "wb");
        if (outFile == NULL) {
            perror("Error opening output file");
            continue;
        }
        fwrite(&imageDataset[sampledIdx], 1, KAGGLE_IMAGE_HEIGHT * KAGGLE_IMAGE_WIDTH, outFile);
        fclose(outFile);
    }
}

int main(int argc, char *argv[]) {
    uint32_t imageDatasetCount = 0;
    uint32_t labelDatasetCount = 0;
    printf("\n");
    printf("Starting Reading Process\n");


    if (argc < 3) {
        fprintf(stderr, "Usage: %s <filename>  <filename>\n", argv[0]);
        return 1;
    }


    loadImagesData(argv[1], &imageDatasetCount);
    loadLabelsData(argv[2], &labelDatasetCount);

    if (imageDatasetCount != labelDatasetCount) {
        fprintf(stderr, "Image and label dataset sizes do not match: %d != %d\n", imageDatasetCount, labelDatasetCount);
        return 1;
    }

    // I like to sample the images produced here to sanity check that my arrays have been loaded correctly
    // Using a python plotter to view the grey scaled value
    output_sample_images(imageDatasetCount);



    return 0;
}