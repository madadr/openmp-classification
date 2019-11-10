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

    const uint32_t BLOCK_DIM = 200;

	const uint32_t ROWS_AMOUNT = 20000;
	const uint32_t ATTRIBUTES_AMOUNT = 16;
    const uint32_t MATRIX_SIZE = ATTRIBUTES_AMOUNT + 1; // attributes + its class
    
    __device__ void findLocalMinMax(double* devAttributes, double* mins, double* maxes)
    {
        int thisThreadStart = threadIdx.x * ROWS_AMOUNT / blockDim.x + blockIdx.x * ROWS_AMOUNT;
        const int nextThreadStart = (threadIdx.x + 1) * ROWS_AMOUNT / blockDim.x + blockIdx.x * ROWS_AMOUNT;
        double localMin = devAttributes[thisThreadStart];
        double localMax = localMin;
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

        mins[blockIdx.x * blockDim.x + threadIdx.x] = localMin;
        maxes[blockIdx.x * blockDim.x + threadIdx.x] = localMax;
    }

    __device__ void findMinMax(volatile double* min, volatile double* max, double* localMin, double* localMax)
    {
        if (threadIdx.x == 0)
        {
            min[blockIdx.x] = localMin[blockIdx.x * blockDim.x];
            max[blockIdx.x] = localMax[blockIdx.x * blockDim.x];
        }
        __syncthreads();

        for (int i = threadIdx.x; i < blockDim.x; ++i)
        {
        auto localMinValue = localMin[blockIdx.x * blockDim.x + i];
        // auto localMinValue = localMin[blockIdx.x * blockDim.x + threadIdx.x];
        if (min[blockIdx.x] > localMinValue)
        {
            min[blockIdx.x] = localMinValue;
        }
        auto localMaxValue = localMax[blockIdx.x * blockDim.x + i];
        // auto localMaxValue = localMax[blockIdx.x * blockDim.x + threadIdx.x];
        if (max[blockIdx.x] < localMaxValue)
        {
            max[blockIdx.x] = localMaxValue;
        }
        }
    }
    
    __global__ void normalize(double* devAttributes, double* localMin, double* localMax)
    {
        // double localMin[ATTRIBUTES_AMOUNT * BLOCK_DIM];
        // double localMax[ATTRIBUTES_AMOUNT * BLOCK_DIM];

        findLocalMinMax(devAttributes, localMin, localMax);
        __syncthreads();

        __shared__ double max[ATTRIBUTES_AMOUNT];
        __shared__ double min[ATTRIBUTES_AMOUNT];

        // if (threadIdx.x == 0)
        // {
        findMinMax(min, max, localMin, localMax);
        // }
        __syncthreads();

        // printf ("BEFORE SAVE blockIdx.x=%d threadIdx.x=%d [%lf : %lf]\n", blockIdx.x, threadIdx.x, min[blockIdx.x], max[blockIdx.x]);
        if (threadIdx.x == 0)
        {
            devAttributes[blockIdx.x] = min[blockIdx.x];
            devAttributes[20000 + blockIdx.x] = max[blockIdx.x];
        }








        // if (threadIdx.x == 0)
        // {
        //     for (int i = blockIdx.x * blockDim.x; i < (blockIdx.x + 1) * blockDim.x; ++i)
        //     {
        //         if (localMin[i] < min[blockIdx.x])
        //         {
        //             min[blockIdx.x] = localMin[i];
        //         }
        //         if (localMax[i] > max[blockIdx.x])
        //         {
        //             max[blockIdx.x] = localMax[i];
        //         }
        //     }
        //     // gridGroup.sync();

        //     if (threadIdx.x == 0)
        //     {
        //         for (int i = 0; i < 3; ++i)
        //         {
        //             // printf("%d row: min=%d; max=%d\n", i, min[i], max[i]);
        //             devAttributes[i] = min[i];
        //             devAttributes[20000 + i] = max[i];
        //         }
        //     }
        // }
    

        //double diff = minMax.first - minMax.second;
    
        //for (auto& value : attributeSet)
        //{
            //value = (value - minMax.first) / diff;
        //}
    }
}

void Scalers::normalize(vector<double>& attributesValues)
{
    double* devLocalMin;
    HANDLE_ERROR(cudaMalloc(&devLocalMin, ATTRIBUTES_AMOUNT * BLOCK_DIM * sizeof(double)));
    double* devLocalMax;
    HANDLE_ERROR(cudaMalloc(&devLocalMax, ATTRIBUTES_AMOUNT * BLOCK_DIM * sizeof(double)));
    

    double* attributes = attributesValues.data();
	double* devAttributes = nullptr;
	HANDLE_ERROR(cudaMalloc(&devAttributes, attributesValues.size() * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(devAttributes, attributes, attributesValues.size() * sizeof(double), cudaMemcpyHostToDevice));
    // Stopwatch watch;
    // watch.start();
	::normalize<<<ATTRIBUTES_AMOUNT, BLOCK_DIM>>>(devAttributes, devLocalMin, devLocalMax);
    // watch.stop();
    // watch.displayTime();
	HANDLE_ERROR(cudaMemcpy(attributes, devAttributes, attributesValues.size() * sizeof(double), cudaMemcpyDeviceToHost));
    // for (int i = 0; i < ATTRIBUTES_AMOUNT; ++i)
    // {
    //     cout << "j = " << i << endl;
    //     HANDLE_ERROR(cudaMemcpy(&attributes[i], &devAttributes[i], ROWS_AMOUNT * sizeof(double), cudaMemcpyDeviceToHost));
    //     HANDLE_ERROR(cudaFree(&devAttributes[i]));
    // }
    cudaFree(devAttributes);
    

    
    double localMax[ATTRIBUTES_AMOUNT * BLOCK_DIM];
	HANDLE_ERROR(cudaMemcpy(localMax, devLocalMax, ATTRIBUTES_AMOUNT * BLOCK_DIM * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(devLocalMax);
    double localMin[ATTRIBUTES_AMOUNT * BLOCK_DIM];
	HANDLE_ERROR(cudaMemcpy(localMin, devLocalMin, ATTRIBUTES_AMOUNT * BLOCK_DIM * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(devLocalMin);

    // for (int i = 0; i < ATTRIBUTES_AMOUNT; ++i)
    // {
    //     cout << " localMin[" << i <<"] ";
    //     for (int j = 0; j < BLOCK_DIM; ++j)
    //     {
    //         cout << localMin[i*BLOCK_DIM + j] << " ";
    //     }
    //     cout << endl;
    //     cout << " localMax[" << i <<"] ";
    //     for (int j = 0; j < BLOCK_DIM; ++j)
    //     {
    //         cout << localMax[i*BLOCK_DIM + j] << " ";
    //     }
    //     cout << endl;
    // }
}

// vector<double*> Scalers::transformToRawPointer(vector<vector<double>>& matrix)
// {
//     // vector<double*> rawMatrix;
//     // int i = 0;
// 	// for (auto& column : matrix)
// 	// {
//     //     double* a = new double[ROWS_AMOUNT];
//     //     for (int j = 0; j < ROWS_AMOUNT; ++j)
//     //     {
//     //         a[j] = matrix[i][j];
//     //     }
//     //     rawMatrix.push_back(a);
//     //     ++i;
// 	// }
// 	// return rawMatrix.data();

//     // double** a = new double*[MATRIX_SIZE];
//     // for (int i = 0; i < MATRIX_SIZE; ++i)
//     // {
//     //     a[i] = new double[ROWS_AMOUNT];
//     //     for (int j = 0; j < ROWS_AMOUNT; ++j)
//     //     {
//     //         a[i][j] = matrix[i][j];
//     //     }
//     // }
//     // return a;

// 	vector<double*> rawMatrix;
// 	for (auto& column : matrix)
// 	{
// 		rawMatrix.push_back(column.data());
// 	}
// 	return rawMatrix;
// }

//pair<double, double> Scalers::findMinMax(vector<double> &attributeSet)
//{
//    double min = attributeSet.at(0);
//    double max = min;
//
//    for (const auto& value : attributeSet)
//    {
//        if (value < min)
//        {
//            min = value;
//        }
//        
//        if (value > max)
//        {
//            max = value;
//        }
//    }
//
//    return std::make_pair(min, max);
//}

void Scalers::standarize(vector<double> &attributeSet)
{
    const auto averageVariation = findAverageAndVariation(attributeSet);

    for (auto& value : attributeSet)
    {
        value = (value - averageVariation.first) / averageVariation.second;
    }
}

pair<double, double> Scalers::findAverageAndVariation(vector<double> &attributeSet)
{
    double average{};
    
    for (const auto& value : attributeSet)
    {
        average += value;
    }
    average /= attributeSet.size();

    double variation{};
    for (const auto& value : attributeSet)
    {
        auto tmp = value - average;
        variation += tmp * tmp;
    }
    variation /= attributeSet.size(); // variance
    variation = sqrt(variation);

    return std::make_pair(average, variation);
}
