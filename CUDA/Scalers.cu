#include "Scalers.cuh"
#include "Stopwatch.cuh"

#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <utility>
#include <iostream>

#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )

static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),file, line);
        exit(EXIT_FAILURE);
    }
}

namespace
{
    using namespace std;

    const uint32_t BLOCK_DIM = 100;

	const uint32_t ROWS_AMOUNT = 20000;
    const uint32_t ATTRIBUTES_AMOUNT = 16;
    
    namespace Normalization
    {
        __device__ void findLocalMinMax(double* devAttributes, double* mins, double* maxes)
        {
            int thisThreadStart = threadIdx.x * ROWS_AMOUNT / blockDim.x + blockIdx.x * ROWS_AMOUNT;
            const int nextThreadStart = (threadIdx.x + 1) * ROWS_AMOUNT / blockDim.x + blockIdx.x * ROWS_AMOUNT;
            double localMin = devAttributes[thisThreadStart];
            double localMax = localMin;
            __syncthreads();
            for (int row = thisThreadStart; row < nextThreadStart; ++row)
            {
                auto value = devAttributes[row];
                if (value < localMin)
                {
                    localMin = value;
                }
                
                if (value > localMax)
                {
                    localMax = value;
                }
            }

            mins[threadIdx.x] = localMin;
            maxes[threadIdx.x] = localMax;
        }

        __device__ void findMinMax(double* min, double* max, double* localMin, double* localMax)
        {
            if (threadIdx.x == 0)
            {
                *min = localMin[0];
                *max = localMax[0];
            }
            __syncthreads();

            for (int i = 0; i < blockDim.x; ++i)
            {
                auto localMinValue = localMin[i];
                if (*min > localMinValue)
                {
                    *min = localMinValue;
                }
                auto localMaxValue = localMax[i];
                if (*max < localMaxValue)
                {
                    *max = localMaxValue;
                }
            }
        }
        
        __device__ void transformValues(double* devAttributes, double* min, double* max)
        {
            int thisThreadStart = threadIdx.x * ROWS_AMOUNT / blockDim.x + blockIdx.x * ROWS_AMOUNT;
            const int nextThreadStart = (threadIdx.x + 1) * ROWS_AMOUNT / blockDim.x + blockIdx.x * ROWS_AMOUNT;
            double diff = *max - *min;
            for (int row = thisThreadStart; row < nextThreadStart; ++row)
            {
                devAttributes[row] = (devAttributes[row] - *min) / diff;
            }
        }

        __global__ void normalize(double* devAttributes)
        {
            __shared__ double max;
            __shared__ double min;
            {
            __shared__ double localMax[BLOCK_DIM];
            __shared__ double localMin[BLOCK_DIM];
            findLocalMinMax(devAttributes, localMin, localMax);
            __syncthreads();

            findMinMax(&min, &max, localMin, localMax);
            __syncthreads();
            } // scoped shared memory variable localMin and localMax to save memory

            transformValues(devAttributes, &min, &max);
        }
    }
    
}

void Scalers::normalize(vector<double>& attributesValues)
{
    double* attributes = attributesValues.data();
	double* devAttributes = nullptr;
	HANDLE_ERROR(cudaMalloc(&devAttributes, attributesValues.size() * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(devAttributes, attributes, attributesValues.size() * sizeof(double), cudaMemcpyHostToDevice));
	Normalization::normalize<<<ATTRIBUTES_AMOUNT, BLOCK_DIM>>>(devAttributes);
	HANDLE_ERROR(cudaMemcpy(attributes, devAttributes, attributesValues.size() * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(devAttributes);
}

void Scalers::standarize(vector<double>& attributesValues)
{
    double* attributes = attributesValues.data();
	double* devAttributes = nullptr;
	HANDLE_ERROR(cudaMalloc(&devAttributes, attributesValues.size() * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(devAttributes, attributes, attributesValues.size() * sizeof(double), cudaMemcpyHostToDevice));
	Normalization::normalize<<<ATTRIBUTES_AMOUNT, BLOCK_DIM>>>(devAttributes);
	HANDLE_ERROR(cudaMemcpy(attributes, devAttributes, attributesValues.size() * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(devAttributes);
}

// void Scalers::standarize(vector<double> &attributeSet)
// {
//     const auto averageVariation = findAverageAndVariation(attributeSet);

//     for (auto& value : attributeSet)
//     {
//         value = (value - averageVariation.first) / averageVariation.second;
//     }
// }

// pair<double, double> Scalers::findAverageAndVariation(vector<double> &attributeSet)
// {
//     double average{};
    
//     for (const auto& value : attributeSet)
//     {
//         average += value;
//     }
//     average /= attributeSet.size();

//     double variation{};
//     for (const auto& value : attributeSet)
//     {
//         auto tmp = value - average;
//         variation += tmp * tmp;
//     }
//     variation /= attributeSet.size(); // variance
//     variation = sqrt(variation);

//     return std::make_pair(average, variation);
// }
