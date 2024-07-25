#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nueralBuildingBlocks.h"


void test_relu_1() {
    int input[5] = {-2, -1, 0, 1, 2};
    int expected_output[5] = {0, 0, 0, 1, 2};
    for (int i = 0; i < 5; i++) {
        if (relu(input[i]) != expected_output[i]) {
            printf("Test failed for input: %d\n", input[i]);
        }
    }
}

// int main() {
//     test_relu_1();

//     // test_10x1_dot_product();
//     // test_dot_product_1();
//     // test_dot_product_2();
//     // test_dot_product_3();
//     return 0;
// }