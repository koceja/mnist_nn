#pragma once

#include <Layer.h>
#include <Images.h>

typedef struct
{
    size_t numLayers;
    Layer** layers;

} Network;

Network* createNetwork(size_t numInputs, size_t* layers, size_t numLayers)
{
    Network* network = (Network*) malloc(sizeof(Network));

    network->numLayers = numLayers;

    network->layers = (Layer**) malloc(numLayers * sizeof(Layer*));

    size_t lastOutputSize = numInputs;

    for (size_t i = 0; i < numLayers; ++i)
    {
        size_t currNumNodes = layers[i];

        network->layers[i] = createInitializedLayer(lastOutputSize, currNumNodes);
        lastOutputSize = currNumNodes;
    }

    return network;
}

void freeNetwork(Network *network)
{
    // Free individual layers
    for (size_t i = 0; i < network->numLayers; ++i)
    {
        freeLayer(network->layers[i]);
    }

    // Free list of layers
    free(network->layers);

    free(network);
}

Matrix* feedForward(Network *network, Matrix *image)
{
    // will be freed after first iteration of calc
    Matrix *flattenedInput = flattenMatrix(image);

    Matrix *lastOutput = flattenedInput;

    for (size_t i = 0; i < network->numLayers; ++i)
    {
        Layer *currLayer = network->layers[i];

        // Keep track of last output here to free
        Matrix *tempOutput = lastOutput;

        lastOutput = calculateOutput(currLayer, lastOutput);

        // Gotta clean up prev output
        freeMatrix(tempOutput);
    }

    return lastOutput;
}

// Many ways to do this, stick to simple right now
double computeLoss(Matrix *probabilities, uint8_t label)
{
    double totalLoss = 0.0;
    for (size_t i = 0; i < probabilities->rows; ++i)
    {
        double actual = getValue(probabilities, i, 0);
        double expected = (double) (label == i);

        // this actually should be 1/2 because it makes the derivative easier
        // I just dont want to include that in the computation because its useless
        totalLoss += (actual - expected) * (actual - expected);
    }

    return totalLoss;
}

void trainNetwork(Network *network, Images *trainingImages, double stepSize, size_t iterations)
{
    double lossSinceLast = 0;
    size_t period = 1000;

    for (size_t i = 0; i < iterations; ++i)
    {
        // Use sgd
        size_t randIndex = randNum(trainingImages->numImages - 1);

        Matrix *currImage = getImage(trainingImages, randIndex);
        uint8_t label = getLabel(trainingImages, randIndex);

        // Create expected list
        Matrix *expectedOutput = createMatrix(10, 1);
        for (size_t i = 0; i < 10; ++i)
        {
            if (label == i)
            {
                setValue(expectedOutput, i, 0, 1.0);
            }
            else
            {
                setValue(expectedOutput, i, 0, 0.0);
            }
        }

        // Run predictions
        Matrix *finalOutput = feedForward(network, currImage);

        double totalLoss = computeLoss(finalOutput, label);

        lossSinceLast += totalLoss;

        if (i % period == 0)
        {
            printf("current loss=%f\n", lossSinceLast/period);
            lossSinceLast = 0.0;
        }

        // START OF BACKPROP

        // General premise of backprop
        // Each layer needs partial derivative of loss with respect to weights
        // 1. work backwards from loss function
        // 2. For first step, update weights_last with direct computations of gradients
        // 3. For all subsequent layers, use the calculations from the i+1 layer and current gradients to update weights_i
        //
        // Updating gradients
        // W_new = W_old - stepSize * gradient
        // Note the minus, we are trying to move in the opposite direction to minimize

        // First step
        // dLoss/dW_L = dLoss/dA * dA/dZ * dZ/dW
        // dZ/dW = d(Z)/dW = d(wT * A_prev)/dW = A_prev
        // dLoss/dW_L = A_prev * (dLoss/dZ)^T

        // All next steps (for W_i)
        // dLoss/dW = dLoss/dW_L = A_prev * (dLoss/dZ)^T
        
        // dLoss/dZ = dLoss/dA * dA/dZ * dZ/dA_-1 * dA_-1/dZ_-1 * dZ_-1/dA_-2 * ... * dA_i/dZ_i
        // Rewrite: dLoss/dZ = dA_i/dZ_i * dZ_i+1/dA_i * dA_i+1/dZ_i+1 * dZ_i+2/dA_i+1 * ... * dZ_L/dA_L * dLoss/dA_L
        // Remember Z = wT * A_i-1, can treat as composition of last A
        // dLoss/dZ = dA_i/dZ_i * W_i+1 * ... * dLoss/dA_L
        //          = dA_i/dZ_i * W_i+1 * dLoss/dZ_i+1
        // So basically, propagate last derivative to each iteration for calculation of gradients

        size_t numLayers = network->numLayers;

        // Dimensions: n x 1
        Matrix *prev_dLdZ = NULL;

        for (int i = numLayers - 1; i >= 0; --i)
        {
            Layer *currentLayer = network->layers[i];

            //printf("Layer=%u, m=%zu, n=%zu\n", i, currentLayer->weights->rows, currentLayer->weights->rows);

            // Dimensions: n x 1
            Matrix *dLdA;

            // If last layer (first iteration), need to compute relation to loss
            if (i == (numLayers - 1))
            {
                // derivative of 1/2(yhat - y)^2
                dLdA = subtract(finalOutput, expectedOutput);
            }
            else
            {
                // Dimensions: m x n  n x 1
                Matrix *lastWeights = network->layers[i+1]->weights;

                //printf("lastWeights: %zuX%zu, prev: %zuX%zu\n", lastWeights->rows, lastWeights->cols, prev_dLdZ->rows, prev_dLdZ->cols);

                // Dimensions here is confusing since its working off of the m and n
                // from the last iteration
                // Basically, for last its m x n times n x 1
                // This works out to be m x 1
                // Last iterations m is current iterations n
                // So this ends up being n x 1
                dLdA = multiply(lastWeights, prev_dLdZ);
            }

            // derivative of sigmoid is sigmoid(X)(1 - sigmoid(x))
            // sigmoid(x) = A by definition
            // Can replace with A(1 - A);

            Matrix *dAdZ;
            {
                Matrix *A = currentLayer->lastOutput;

                // TODO: Something off here
                // A should be n x 1 so this is n x 1 with all 1s
                Matrix *oneMatrix = createMatrix(A->rows, A->cols);
                for (int i = 0; i < A->rows; ++i)
                {
                    for (int j = 0; j < oneMatrix->cols; ++j)
                    {
                        setValue(oneMatrix, i, j, 1.0);
                    }
                }

                // Dimensions: n x 1
                Matrix *diffMatrix = subtract(oneMatrix, A);
                // Dimensions: 1 x n
                Matrix *aT = transpose(A);

                // Dimensions: 1 x 1
                dAdZ = multiply(aT, diffMatrix);

                freeMatrix(oneMatrix);
                freeMatrix(diffMatrix);
                freeMatrix(aT);
            }

            // Dimensions: n x 1 times 1 x 1 is n x 1

            // printf("dLdA: %zuX%zu, dAdZ: %zuX%zu\n", dLdA->rows, dLdA->cols, dAdZ->rows, dAdZ->cols);
            Matrix *dLdZ = multiply(dLdA, dAdZ);

            // Dimensions: 1 x n
            Matrix *dLdZ_T = transpose(dLdZ);

            Matrix *currentInput = currentLayer->lastInput;

            // Dimensions:  m x 1 times 1 x n
            //  so m x n
            Matrix *dLdW = multiply(currentInput, dLdZ_T);

            // Dimensions: n x 1
            Matrix *dLdW0 = dLdZ;

            printMatrix(dLdW);
            Matrix *intermediate = scalarMultiply(dLdW, (1.0/(i + 1.0))); // Dimensions: m x n
            Matrix *currWeights = currentLayer->weights; // Dimensions: m x n
            Matrix *currW0 = currentLayer->w0; // Dimensions: n x 1

            // Update weights
            subtractInPlace(currWeights, intermediate);
            subtractInPlace(currW0, dLdW0);

            // Free previous iterations gradient
            if (prev_dLdZ != NULL)
                freeMatrix(prev_dLdZ);

            // Store for next iteration (back-prop part)
            prev_dLdZ = dLdZ;

            
            freeMatrix(dLdA);
            freeMatrix(dAdZ);
            freeMatrix(dLdW);
            freeMatrix(intermediate);
        }

        // Clean-up
        if (prev_dLdZ != NULL)
            freeMatrix(prev_dLdZ);

        freeMatrix(finalOutput);
        freeMatrix(expectedOutput);
    }

    
}

size_t predictNetwork(Network *network, Matrix *image, uint8_t label)
{
    Matrix *digitProbabilities = feedForward(network, image);

    double totalLoss = computeLoss(digitProbabilities, label);

    printf("Total loss: %f\n", totalLoss);

    double highestProbability = getValue(digitProbabilities, 0, 0);
    double highestProabilityIdx = 0;

    for (size_t i = 1; i < digitProbabilities->rows; ++i)
    {
        double currVal = getValue(digitProbabilities, i, 0);

        bool isHighest = currVal > highestProbability;

        highestProbability = isHighest ? currVal : highestProbability;
        highestProabilityIdx = isHighest ? i : highestProabilityIdx;
    }

    freeMatrix(digitProbabilities);

    return highestProabilityIdx;
}