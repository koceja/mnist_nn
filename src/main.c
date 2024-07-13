#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include <Images.h>
#include <Files.h>
#include <Utils.h>
#include <Layer.h>
#include <Network.h>


void train(Images images)
{
    
}

void testMatrixMult()
{
    Matrix* first = createMatrix(3,2);
    Matrix* second = createMatrix(2,3);

    for (int i = 0; i < first->rows; ++i)
    {
        for (int j = 0; j < first->cols; ++j)
        {
            setValue(first, i, j, i + j);
        }
    }

    setValue(second, 0, 0, 2.0);
    setValue(second, 0, 1, 3.0);
    setValue(second, 0, 2, 6.0);
    setValue(second, 1, 0, 3.0);
    setValue(second, 1, 1, 7.0);
    setValue(second, 1, 2, 1.0);

    Matrix* result = multiply(first, second);

    for (int i = 0; i < first->rows; ++i)
    {
        for (int j = 0; j < first->cols; ++j)
        {
            printf("%5.2f ", getValue(first, i, j));
        }
        printf("\n");
    }

    printf("\n");

    for (int i = 0; i < second->rows; ++i)
    {
        for (int j = 0; j < second->cols; ++j)
        {
            printf("%5.2f ", getValue(second, i, j));
        }
        printf("\n");
    }

    printf("\n");

    for (int i = 0; i < result->rows; ++i)
    {
        for (int j = 0; j < result->cols; ++j)
        {
            printf("%5.2f ", getValue(result, i, j));
        }
        printf("\n");
    }

    freeMatrix(first);
    freeMatrix(second);
    freeMatrix(result);
}


int main()
{
    srand(time(NULL));
    //testMatrixMult();
    printf("Loading in dataset...\n");
    Images* trainingImages = getTrainingImages();
    Images* testImages = getTestImages();

    if (trainingImages->isNull || testImages->isNull)
    {
        printf("Failed to get image.\n");
        return 0;
    }

    printf("Completed loading in dataset.\n");

    size_t numInputs = getImageSize(trainingImages);

    //size_t imageNum = 3062;

    // printImage(trainingImages, imageNum);

    printf("Initializing neural network...\n");

    size_t layerSizes[2] = { 5, 10 };
    
    Network *network = createNetwork(numInputs, layerSizes, 2);

    printf("Done initializing neural network.\n");

    printf("Training...\n");

    trainNetwork(network, trainingImages, 0.0001, 100);

    printf("Done training.\n");

    size_t index = 1230;

    uint8_t digit = (uint8_t) predictNetwork(network, getImage(trainingImages, index), getLabel(trainingImages, index));

    Matrix *digitProbs = feedForward(network, getImage(trainingImages, index));
    printMatrix(digitProbs);
    freeMatrix(digitProbs);

    printf("Digit: %u, ActualDigit=%u\n", digit, getLabel(trainingImages, index));

    // Clean up
    freeNetwork(network);
    freeImages(trainingImages);
    freeImages(testImages);

    printf("Finished :)\n");

    return 0;
}