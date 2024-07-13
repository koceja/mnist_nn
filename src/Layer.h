#pragma once

#include <Utils.h>

typedef struct
{
    Matrix *weights;
    Matrix *w0;

    // Just store copies of this because im lazy
    Matrix *lastInput;
    Matrix *lastOutput;

} Layer;

size_t getNumNodes(Layer *layer)
{
    return layer->weights->cols;
}

size_t getNumInputs(Layer *layer)
{
    return layer->weights->rows;
}

Layer *createLayer(size_t numInputs, size_t numNodes)
{
    Layer *result = (Layer *)malloc(sizeof(Layer));

    result->weights = createMatrix(numInputs, numNodes);
    result->w0 = createMatrix(numNodes, 1);
    result->lastInput = NULL;
    result->lastOutput = NULL;

    return result;
}

void initializeWeights(Layer *layer)
{
    size_t m = getNumInputs(layer);

    for (size_t i = 0; i < layer->weights->rows; ++i)
    {
        for (size_t j = 0; j < layer->weights->cols; ++j)
        {
            double gaussianRand = generateGaussian(0.0, 1.0 / m);

            setValue(layer->weights, i, j, gaussianRand);
        }
    }
}

void initializeW0(Layer *layer)
{
    for (size_t i = 0; i < layer->w0->rows; ++i)
    {
        for (size_t j = 0; j < layer->w0->cols; ++j)
        {
            double gaussianRand = generateGaussian(0, 1);

            setValue(layer->w0, i, j, gaussianRand);
        }
    }
}

void initializeLayer(Layer *layer)
{
    initializeWeights(layer);
    initializeW0(layer);
}

Layer *createInitializedLayer(size_t numInputs, size_t numNodes)
{
    Layer *result = createLayer(numInputs, numNodes);

    initializeLayer(result);

    return result;
}

Matrix *calculateOutput(Layer *layer, Matrix *input)
{
    layer->lastInput = cloneMatrix(input);

    Matrix *weightsT = transpose(layer->weights);
    Matrix *wTx = multiply(weightsT, input);

    Matrix *Z = add(wTx, layer->w0);

    Matrix *A = sigmoidMatrix(Z);

    // Clean up intermediate matrices
    // TODO: Make these calculations in place to remove heap allocations
    freeMatrix(weightsT);
    freeMatrix(wTx);
    freeMatrix(Z);

    layer->lastOutput = cloneMatrix(A);

    return A;
}

void freeLayer(Layer *layer)
{
    freeMatrix(layer->weights);
    freeMatrix(layer->w0);

    if (layer->lastInput != NULL)
        freeMatrix(layer->lastInput);

    if (layer->lastOutput != NULL)
        freeMatrix(layer->lastOutput);

    free(layer);
}