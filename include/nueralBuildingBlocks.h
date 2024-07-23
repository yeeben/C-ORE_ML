#include <stdint.h>


typedef struct {
    float weights;
    float bias;
} NueralLayer;

void init_params(NueralLayer *layer1, uint16_t layer1Size, NueralLayer *layer2, uint16_t layer2Size);
uint32_t relu(uint32_t x);
