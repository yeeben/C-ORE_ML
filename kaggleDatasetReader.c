#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "kaggleDatasetReader.h"

void loadImagesData(const char *filename, uint32_t* imageDatasetCount) {    
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    ImageHeader_t header = {};

    if (fread(&header, 1, sizeof(ImageHeader_t), file) != sizeof(ImageHeader_t)) {
        perror("Error reading file");
        fclose(file);
        return;
    }

}

void loadLabelsData(const char *filename, uint32_t* imageDatasetCount) {
    
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    LabelHeader_t header = {};
    
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    if (fread(&header, 1, sizeof(LabelHeader_t), file) != sizeof(LabelHeader_t)) {
        perror("Error reading file");
        fclose(file);
        return;
    }

}
