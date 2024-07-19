#include "stdio.h"
#include <stdint.h>


int main() {
    uint32_t testArray[10][10] = {};
    printf("Size of array: %ld\n", sizeof(testArray));

    uint32_t otherArray[20] = {};
    printf("Size of other array: %ld\n", sizeof(otherArray));

    return 0;
}


