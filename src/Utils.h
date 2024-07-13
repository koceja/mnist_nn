#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

char intensityToChar(double intensity) {
    if (intensity < 0) intensity = 0;
    if (intensity > 255) intensity = 255;

    if (intensity >= 0 && intensity < 51) {
        return ' '; // Space character for intensity [0, 50]
    } else if (intensity >= 51 && intensity < 102) {
        return '.'; // Light shade character for intensity [51, 101]
    } else if (intensity >= 102 && intensity < 153) {
        return '*'; // Medium shade character for intensity [102, 152]
    } else if (intensity >= 153 && intensity < 204) {
        return 'o'; // Dark shade character for intensity [153, 203]
    } else if (intensity >= 204) {
        return '#'; // Full block character for intensity [204, 255]
    }

    return ' '; // Default to space character
}

// double generateGaussian(double mean, double std_dev) {
//     static std::random_device rd;
//     static std::mt19937 gen(rd());
//     static std::uniform_real_distribution<> dis(0.0, 1.0);
    
//     double u1 = dis(gen);
//     double u2 = dis(gen);
//     double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

//     printf("Random num=%f\n", mean + std_dev * z0);
    
//     return mean + std_dev * z0;
// }

double generateGaussian(double mean, double std_dev)
{
    double u1 = ((double) rand() / RAND_MAX);
    double u2 = ((double) rand() / RAND_MAX);
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

    //printf("mean=%f, stddev=%f, Random num=%f\n", mean, std_dev, mean + std_dev * z0);
    return mean + std_dev * z0;
}

double sigmoid(double val)
{
    return 1 / (1 + exp(-val));
}

size_t randNum(int upperBound)
{
    return rand() % (upperBound + 1);
}