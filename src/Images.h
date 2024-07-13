#pragma once

#include <Files.h>
#include <Matrix.h>
#include <Utils.h>

typedef struct
{
    bool isNull;

    Matrix **images;
    uint8_t *labels;

    size_t numImages;
} Images;

Matrix *getImage(Images *images, size_t index)
{
    return images->images[index];
}

void setImage(Images *images, size_t index, Matrix *val)
{
    images->images[index] = val;
}

size_t getImageSize(Images* currImages)
{
    Matrix *firstImage = currImages->images[0];
    return firstImage->rows * firstImage->cols;
}

double getPixel(Images *currImages, size_t image, size_t row, size_t col)
{
    Matrix *currImage = getImage(currImages, image);
    return getValue(currImage, row, col);
}

void writePixel(Images *currImages, size_t image, size_t row, size_t col, double val)
{
    Matrix *currImage = getImage(currImages, image);
    setValue(currImage, row, col, val);
}

uint8_t getLabel(Images *currImages, size_t image)
{
    return currImages->labels[image];
}

void writeLabel(Images *currImages, size_t image, uint8_t val)
{
    currImages->labels[image] = val;
}

Images *createImages(size_t numImages, size_t numRows, size_t numCols)
{
    Images *myImages = (Images *)malloc(sizeof(Images));
    myImages->isNull = false;
    myImages->images = malloc(numImages * sizeof(Matrix *));

    for (int i = 0; i < numImages; ++i)
    {
        setImage(myImages, i, createMatrix(numRows, numCols));
    }

    myImages->labels = malloc(numImages * sizeof(uint8_t));
    myImages->numImages = numImages;

    return myImages;
}

Images *getNullImage()
{
    Images *myImages = (Images *)malloc(sizeof(Images));
    myImages->isNull = true;

    return myImages;
}

void freeImages(Images *currImages)
{
    // free individual images
    for (int i = 0; i < currImages->numImages; ++i)
    {
        freeMatrix(getImage(currImages, i));
    }

    // free lists
    free(currImages->images);
    free(currImages->labels);

    free(currImages);
}

Images *getImages(FILE *imagesFile, FILE *labelsFile)
{
    uint32_t numImages = readUnsignedInt(imagesFile);
    uint32_t numRows = readUnsignedInt(imagesFile);
    uint32_t numCols = readUnsignedInt(imagesFile);

    uint32_t numLabels = readUnsignedInt(labelsFile);

    if (numLabels != numImages)
    {
        printf("Num images and labels mismatch: #images=%u, #labels=%u\n", numImages, numLabels);
        return getNullImage();
    }

    Images *images = createImages(numImages, numRows, numCols);

    for (int i = 0; i < numImages; ++i)
    {
        writeLabel(images, i, readUnsigned(labelsFile));

        for (int x = 0; x < numRows; ++x)
        {
            for (int y = 0; y < numCols; ++y)
            {
                writePixel(images, i, x, y, readUnsigned(imagesFile));
            }
        }
    }

    return images;
}

Images *getImagesByPath(char *imagesPath, char *labelsPath)
{
    FILE *imagesFile = getFile(imagesPath, 2051u);
    FILE *labelsFile = getFile(labelsPath, 2049u);

    if (imagesFile == NULL || labelsFile == NULL)
    {
        return getNullImage();
    }

    Images *images = getImages(imagesFile, labelsFile);

    // close files
    fclose(imagesFile);
    fclose(labelsFile);

    return images;
}

Images *getTrainingImages()
{
    return getImagesByPath("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte");
}

Images *getTestImages()
{
    return getImagesByPath("../data/t10k-images.idx3-ubyte", "../data/t10k-labels.idx1-ubyte");
}

Matrix *getFlattenedImage(Images *images, size_t index)
{
    return flattenMatrix(getImage(images, index));
}

void printImage(Images *images, size_t index)
{
    Matrix *image = getImage(images, index);

    for (size_t i = 0; i < image->rows; ++i)
    {
        for (size_t j = 0; j < image->cols; ++j)
        {
            printf("%c", intensityToChar(getValue(image, i, j)));
        }
        printf("\n");
    }

    printf("Label: %u\n", getLabel(images, index));
}