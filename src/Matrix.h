#pragma once

#include <stdlib.h>
#include <Utils.h>

typedef struct
{
    size_t rows;
    size_t cols;

    double *data;

} Matrix;

Matrix *createMatrix(size_t rows, size_t cols)
{
    Matrix *result = (Matrix *)malloc(sizeof(Matrix));
    result->rows = rows;
    result->cols = cols;
    result->data = malloc(rows * cols * sizeof(double));

    return result;
}

void freeMatrix(Matrix *matrix)
{
    free(matrix->data);
    free(matrix);
}

double getValue(Matrix *matrix, size_t row, size_t col)
{
    return matrix->data[(row * matrix->cols) + col];
}

void setValue(Matrix *matrix, size_t row, size_t col, double value)
{
    matrix->data[(row * matrix->cols) + col] = value;
}

Matrix *cloneMatrix(Matrix *matrix)
{
    Matrix *result = createMatrix(matrix->rows, matrix->cols);

    for (size_t i = 0; i < matrix->rows; ++i)
    {
        for (size_t j = 0; j < matrix->cols; ++j)
        {
            setValue(result, i, j, getValue(matrix, i, j));
        }
    }

    return result;
}

void printMatrix(Matrix *matrix)
{
    for (size_t i = 0; i < matrix->rows; ++i)
    {
        for (size_t j = 0; j < matrix->cols; ++j)
        {
            printf("%f ", getValue(matrix, i, j));
        }
        printf("\n");
    }
}

Matrix *flattenMatrix(Matrix *matrix)
{
    size_t rows = matrix->rows, cols = matrix->cols; 
    size_t size = rows * cols;

    Matrix *newMatrix = createMatrix(size, 1);

    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            size_t newIdx = i * cols + j;
            setValue(newMatrix, newIdx, 0, getValue(matrix, i, j));
        }
    }

    return newMatrix;
}

Matrix *scalarMultiply(Matrix *lhs, double scalar)
{
    Matrix *result = createMatrix(lhs->rows, lhs->cols);

    for (size_t i = 0; i < lhs->rows; ++i)
    {
        for (size_t j = 0; j < lhs->cols; ++j)
        {
            double product = getValue(lhs, i, j) * scalar;
            setValue(result, i, j, product);
        }
    }

    return result;
}

Matrix *add(Matrix *lhs, Matrix *rhs)
{
    if (lhs->cols != rhs->cols || lhs->rows != rhs->rows)
    {
        printf("Matrix dimensions aren't the same for addition.\n");
        return NULL;
    }

    Matrix *result = createMatrix(lhs->rows, lhs->cols);

    for (size_t i = 0; i < lhs->rows; ++i)
    {
        for (size_t j = 0; j < lhs->cols; ++j)
        {
            double sum = getValue(lhs, i, j) + getValue(rhs, i, j);
            setValue(result, i, j, sum);
        }
    }

    return result;
}

Matrix *subtract(Matrix *lhs, Matrix *rhs)
{
    if (lhs->cols != rhs->cols || lhs->rows != rhs->rows)
    {
        printf("Matrix dimensions aren't the same for addition.\n");
        return NULL;
    }

    Matrix *result = createMatrix(lhs->rows, lhs->cols);

    for (size_t i = 0; i < lhs->rows; ++i)
    {
        for (size_t j = 0; j < lhs->cols; ++j)
        {
            double diff = getValue(lhs, i, j) - getValue(rhs, i, j);
            setValue(result, i, j, diff);
        }
    }

    return result;
}

Matrix *multiply(Matrix *lhs, Matrix *rhs)
{
    if (lhs->cols != rhs->rows)
    {
        printf("Matrix dimensions aren't compatible for multiplication.\n");
        return NULL;
    }

    Matrix *result = createMatrix(lhs->rows, rhs->cols);

    // for each row of the lhs
    for (int i = 0; i < lhs->rows; ++i)
    {
        // for each col of the rhs
        for (int j = 0; j < rhs->cols; ++j)
        {
            double total = 0;
            // for each row of the rhs
            for (int k = 0; k < rhs->rows; ++k)
            {
                total += getValue(lhs, i, k) * getValue(rhs, k, j);
            }

            setValue(result, i, j, total);
        }
    }

    return result;
}

Matrix *transpose(Matrix *matrix)
{
    Matrix *newMatrix = createMatrix(matrix->cols, matrix->rows);

    for (int i = 0; i < matrix->rows; ++i)
    {
        for (int j = 0; j < matrix->cols; ++j)
        {
            setValue(newMatrix, j, i, getValue(matrix, i, j));
        }
    }

    return newMatrix;
}

Matrix *sigmoidMatrix(Matrix *matrix)
{
    Matrix *newMatrix = createMatrix(matrix->rows, matrix->cols);

    for (size_t i = 0; i < matrix->rows; ++i)
    {
        for (size_t j = 0; j < matrix->cols; ++j)
        {
            double newVal = sigmoid(getValue(matrix, i, j));
            setValue(newMatrix, i, j, newVal);
        }
    }

    return newMatrix;
}

void subtractInPlace(Matrix *lhs, Matrix *rhs)
{
    for (size_t i = 0; i < lhs->rows; ++i)
    {
        for (size_t j = 0; j < lhs->cols; ++j)
        {
            double newVal = getValue(lhs, i, j) - getValue(rhs, i, j);
            setValue(lhs, i, j, newVal);
        }
    }
}




void scalarMultiplyInPlace(Matrix *matrix, double scalar)
{
    for (size_t i = 0; i < matrix->rows; ++i)
    {
        for (size_t j = 0; j < matrix->cols; ++j)
        {
            double newVal = scalar * getValue(matrix, i, j);
            setValue(matrix, i, j, newVal);
        }
    }
}
