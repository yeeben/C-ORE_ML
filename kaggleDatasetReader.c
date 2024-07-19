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


void readImagesHeader(const char *filename, uint32_t *imageDatasetCount) {
    struct ImageHeader header = {};
    uint8_t headerReadBuffer[IMAGE_HEADER_BUFFER_SIZE] = {};

    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    if (fread(headerReadBuffer, 1, IMAGE_HEADER_BUFFER_SIZE, file) != IMAGE_HEADER_BUFFER_SIZE) {
        perror("Error reading file");
        fclose(file);
        return;
    }
    header.magic = (headerReadBuffer[0] << 24) | (headerReadBuffer[1] << 16) | (headerReadBuffer[2] << 8) | headerReadBuffer[3];
    header.size = (headerReadBuffer[4] << 24) | (headerReadBuffer[5] << 16) | (headerReadBuffer[6] << 8) | headerReadBuffer[7];
    header.rows = (headerReadBuffer[8] << 24) | (headerReadBuffer[9] << 16) | (headerReadBuffer[10] << 8) | headerReadBuffer[11];
    header.columns = (headerReadBuffer[12] << 24) | (headerReadBuffer[13] << 16) | (headerReadBuffer[14] << 8) | headerReadBuffer[15];

    if (header.magic != IMAGES_MAGIC_NUMBER) {
        fprintf(stderr, "Invalid images magic number: %d\n", header.magic);
        fclose(file);
        return;
    }

    printf("Parsed %s Header: magic=%d, size=%d, rows=%d, columns=%d\n", filename, header.magic, header.size, header.rows, header.columns);
    *imageDatasetCount = header.size;
    fclose(file);
}

void readLabelsHeader(const char *filename, uint32_t *labelDatasetCount) {
    struct LabelHeader header = {};
    
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    uint8_t headerReadBuffer[LABEL_HEADER_BUFFER_SIZE];
    if (fread(headerReadBuffer, 1, LABEL_HEADER_BUFFER_SIZE, file) != LABEL_HEADER_BUFFER_SIZE) {
        perror("Error reading file");
        fclose(file);
        return;
    }

    header.magic = (headerReadBuffer[0] << 24) | (headerReadBuffer[1] << 16) | (headerReadBuffer[2] << 8) | headerReadBuffer[3];
    header.size = (headerReadBuffer[4] << 24) | (headerReadBuffer[5] << 16) | (headerReadBuffer[6] << 8) | headerReadBuffer[7];
    if (header.magic != LABEL_MAGIC_NUMBER) {
        fprintf(stderr, "Invalid labels magic number: %d\n", header.magic);
        fclose(file);
        return;
    }

    printf("Parsed %s Header: magic=%d, size=%d\n", filename, header.magic, header.size);
    *labelDatasetCount = header.size;
    fclose(file);
}

void loadImagesData(const char *filename, uint32_t imageDatasetCount, bool outputImage) {
    struct GreyScaleImage singleImageBuffer = {};
    
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    fread(&singleImageBuffer, 1, IMAGE_HEADER_BUFFER_SIZE, file); // Get Past the Header in the stream
    fread(&singleImageBuffer, 1, KAGGLE_IMAGE_HEIGHT * KAGGLE_IMAGE_WIDTH, file);

    for(int i = 0; i < imageDatasetCount; i++) {
        fread(&imageDataset[i], 1, KAGGLE_IMAGE_HEIGHT * KAGGLE_IMAGE_WIDTH, file);
    }
    fclose(file);
    // I like to sample the images produced here to sanity check that my arrays have been loaded correctly
    // Using a python plotter to view the grey scaled value
    if(outputImage) {
        for(int i = 0; i < 6; i++) {

            char outputFilename[20];
            uint32_t sampledData = rand() % imageDatasetCount;
            snprintf(outputFilename, sizeof(outputFilename), "sampleImage%d.png", i);

            FILE *outFile = fopen(outputFilename, "wb");
            if (outFile == NULL) {
                perror("Error opening output file");
                continue;
            }
            fwrite(&imageDataset[sampledData], 1, KAGGLE_IMAGE_HEIGHT * KAGGLE_IMAGE_WIDTH, outFile);
            fclose(outFile);
        }
    }
}

int main(int argc, char *argv[]) {
    uint32_t imageDatasetCount = 0;
    uint32_t labelDatasetCount = 0;
    printf("\n");
    printf("Starting Reading Process\n");


    if (argc < 2) {
        fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        return 1;
    }
    readImagesHeader(argv[1], &imageDatasetCount);
    readLabelsHeader(argv[2], &labelDatasetCount);

    if (imageDatasetCount != labelDatasetCount) {
        fprintf(stderr, "Image and label dataset sizes do not match: %d != %d\n", imageDatasetCount, labelDatasetCount);
        return 1;
    }

    loadImagesData(argv[1], imageDatasetCount, false);

    return 0;
}