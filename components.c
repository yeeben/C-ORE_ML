#include <stdint.h>

uint8_t relu(uint8_t x) {
    if (x > 0) {
        return x;
    } else {
        return 0;
    }
}
=