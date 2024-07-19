#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#define IMAGES_MAGIC_NUMBER 2051
#define LABEL_MAGIC_NUMBER 2049
#define BITS_IN_BYTE 8
#define HEADER_BUFFER_SIZE 16

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

static bool verifyMagic(const char *filename, unsigned int magic);


void readImagesFile(const char *filename) {
    struct ImageHeader header = {};

    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    uint8_t readBuffer[HEADER_BUFFER_SIZE];
    if (fread(readBuffer, 1, HEADER_BUFFER_SIZE, file) != HEADER_BUFFER_SIZE) {
        perror("Error reading file");
        fclose(file);
        return;
    }
    header.magic = (readBuffer[0] << 24) | (readBuffer[1] << 16) | (readBuffer[2] << 8) | readBuffer[3];
    header.size = (readBuffer[4] << 24) | (readBuffer[5] << 16) | (readBuffer[6] << 8) | readBuffer[7];
    header.rows = (readBuffer[8] << 24) | (readBuffer[9] << 16) | (readBuffer[10] << 8) | readBuffer[11];
    header.columns = (readBuffer[12] << 24) | (readBuffer[13] << 16) | (readBuffer[14] << 8) | readBuffer[15];

    if (header.magic != IMAGES_MAGIC_NUMBER) {
        fprintf(stderr, "Invalid images magic number: %d\n", header.magic);
        fclose(file);
        return;
    }

    printf("Parsed Header: magic=%d, size=%d, rows=%d, columns=%d\n", header.magic, header.size, header.rows, header.columns);
    fclose(file);
}

void readLablesFile(const char *filename) {
    struct LabelHeader header = {};
    
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    uint8_t buffer[4 * BITS_IN_BYTE];
    if (fread(buffer, 1, 8, file) != 8) {
        perror("Error reading file");
        fclose(file);
        return;
    }

    header.magic = (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];
    header.size = (buffer[4] << 24) | (buffer[5] << 16) | (buffer[6] << 8) | buffer[7];
    if (header.magic != LABEL_MAGIC_NUMBER) {
        fprintf(stderr, "Invalid labels magic number: %d\n", header.magic);
        fclose(file);
        return;
    }

    printf("Parsed Header: magic=%d, size=%d\n", header.magic, header.size);
    fclose(file);

}

int main(int argc, char *argv[]) {
    printf("\n");
    printf("Starting Reading Process\n");


    if (argc < 2) {
        fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        return 1;
    }
    printf("Reading file %s\n", argv[1]);
    readImagesFile(argv[1]);
    readLablesFile(argv[2]);


    return 0;
}