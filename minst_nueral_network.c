#include <stdio.h> 
#include <stdlib.h> 
#include "kaggleDatasetReader.h"
/*
    Step 1: Load the dataset
        A[0] = X  
        
        Sizing:
            A[0] = 784 pixels x len(dataset)
            X = 784 pixels x len(dataset)
        

    Step 2: Forward Propagation
        Z[1] = W[1] * A[0] + B[1] 

        Sizing:
            A[0] = 784 pixels x len(dataset)
            W[1] = Weights1 // input layer (784 pixels x 10 neurons)
            B[1] = Bias1 // 10 neurons x len(dataset)
            Z[1] = 10 x len(dataset)

    Step 3: Activation Function
        A[1] = relu(Z[1])

        Sizing:
            A[1] = 10 x len(dataset)
    
    Step 4: Forward Propagation
        Z[2] = W[2] * A[1] + B[2]

        Sizing:
            A[1] = 10 x len(dataset)
            W[2] = Weights2 // input layer (10 neurons x 10 neurons)
            B[2] = Bias2 // 10 neurons x len(dataset)
            Z[2] = 10 x len(dataset)

    Step 5: Activation Function
        A[2] = softmax(Z[2])

        Sizing:
            A[2] = 10 x len(dataset)
    
*/


int main(int argc, char *argv[]) {

    uint32_t image_dataset_count = 0;
    uint32_t label_dataset_count = 0;

    printf("\n");
    printf("Starting Reading Process\n");


    if (argc < 3) {
        fprintf(stderr, "Usage: %s <filename>  <filename>\n", argv[0]);
        return 1;
    }


    loadImagesData(argv[1], &image_dataset_count);
    loadLabelsData(argv[2], &label_dataset_count);

    if (image_dataset_count != label_dataset_count && image_dataset_count < 1) {
        fprintf(stderr, "Image and label dataset sizes do not match: %d != %d\n", image_dataset_count, label_dataset_count);
        return 1;
    }

    return 0;
}



