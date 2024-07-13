#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

uint32_t readUnsignedInt(FILE *file)
{
    uint32_t val;
    fread(&val, sizeof(uint32_t), 1, file);

    return __builtin_bswap32(val);
}

uint8_t readUnsigned(FILE *file)
{
    uint8_t val;
    fread(&val, sizeof(uint8_t), 1, file);

    return val;
}

FILE *getFile(char *path, uint32_t magicNum)
{
    FILE *file;
    file = fopen(path, "rb");

    if (file == NULL)
    {
        return NULL;
    }

    uint32_t readMagNum = readUnsignedInt(file);
    if (readMagNum != magicNum)
    {
        printf("Cannot read file, unexpected magic number: %u\n", magicNum);
        return NULL;
    }

    return file;
}
