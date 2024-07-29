#include <stdio.h> 
#include <stdlib.h> 
#include "kaggleDatasetReader.h"
#include "nueralBuildingBlocks.h"

#define TRAIN_STEPS 50
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
KaggleImageSubset_t pTrainImageSubset = {};
KaggleImageSubset_t pTestImageSubset = {};

NetworkLayer_t inputLayer = {};
HiddenLayer_t hiddenLayer = {};


int main(int argc, char *argv[]) {
    uint32_t imageTrainDatasetCount = 0;
    uint32_t labelTrainDatasetCount = 0;

    uint32_t imageTestDatasetCount = 0;
    uint32_t labelTestDatasetCount = 0;

    float learning_rate = 0.05;
    uint32_t epochs = 0;
    int i,j = 0;
    float accuracy = 0.0;

    printf("\n");
    printf("Starting Reading Process\n");


    if (argc < 3) {
        fprintf(stderr, "Usage: %s <filename>  <filename>\n", argv[0]);
        return 1;
    }


    KaggleImage_t *pTrainImages = loadImagesData("minst_dataset/t10k-images.idx3-ubyte", &imageTrainDatasetCount);
    KaggleImage_t *pTestImages = loadImagesData("minst_dataset/train-images.idx3-ubyte", &imageTestDatasetCount);

    if (pTrainImages == NULL || pTestImages == NULL) {
        fprintf(stderr, "Error loading images\n");
        return 1;
    }

    uint8_t *pTrainLabels = loadLabelsData("minst_dataset/t10k-labels.idx1-ubyte", &labelTrainDatasetCount);
    uint8_t *pTestLabels = loadLabelsData("minst_dataset/train-labels.idx1-ubyte", &labelTestDatasetCount);

    if (pTrainLabels == NULL || pTestLabels == NULL) {
        fprintf(stderr, "Error loading labels\n");
        return 1;
    }

    if (imageTrainDatasetCount != labelTrainDatasetCount && imageTrainDatasetCount < 1) {
        fprintf(stderr, "Image and label dataset sizes do not match: %d != %d\n", imageTrainDatasetCount, labelTrainDatasetCount);
        return 1;
    }

    if (imageTestDatasetCount != labelTestDatasetCount && imageTestDatasetCount < 1) {
        fprintf(stderr, "Image and label dataset sizes do not match: %d != %d\n", imageTestDatasetCount, labelTestDatasetCount);
        return 1;
    }


    init_params(&inputLayer, &hiddenLayer);
    pTestImageSubset.images = pTestImages;
    pTestImageSubset.labels = pTestLabels;
    pTestImageSubset.imageDatasetCount = imageTestDatasetCount;

    for(epochs = 0; epochs < TRAIN_STEPS; epochs ++) {
        for(i = 0; i < imageTrainDatasetCount; i += BATCH_SIZE) {
            pTrainImageSubset.images = &pTrainImages[i];
            pTrainImageSubset.labels = &pTrainLabels[i];
            if ((i + BATCH_SIZE) > imageTestDatasetCount) {
                pTrainImageSubset.imageDatasetCount = imageTestDatasetCount - i;
            } else {
                pTrainImageSubset.imageDatasetCount = BATCH_SIZE;
            }

            gradient_descent(&pTrainImageSubset, &inputLayer, &hiddenLayer, learning_rate);
        }
        accuracy = calculate_accuracy(&pTestImageSubset, &inputLayer, &hiddenLayer);
        printf("\n--------------------\n");
        printf("Steps: %d Accuracy: %f\n", epochs, accuracy);
    }

    return 0;
}



