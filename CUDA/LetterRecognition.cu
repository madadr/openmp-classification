#include "LetterRecognition.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdint>
#include <set>

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
    using namespace cooperative_groups;

    const uint32_t BLOCK_DIM = 2;

    const uint32_t ROWS_AMOUNT = 20000;
    const uint32_t ATTRIBUTES_AMOUNT = 16;
	
    const uint32_t TRAIN_SET_SIZE = ROWS_AMOUNT * 0.9;
    const uint32_t TEST_SET_SIZE = ROWS_AMOUNT - TRAIN_SET_SIZE;

    namespace GPU
    {
        __device__ void calculateSquares(double* dataset, int testRowIndex)
        {
            int thisThreadStart = threadIdx.x * TRAIN_SET_SIZE / blockDim.x + blockIdx.x * ROWS_AMOUNT;
            const int nextThreadStart = (threadIdx.x + 1) * TRAIN_SET_SIZE / blockDim.x + blockIdx.x * ROWS_AMOUNT;

            // Calculate squares for every attribute
            double testAttribute = dataset[blockIdx.x * ROWS_AMOUNT + testRowIndex];
            for (uint32_t k = thisThreadStart; k < nextThreadStart; ++k)
            {
                double tmp = testAttribute - dataset[k];
                // if (k % 20000 == 0)
                // {
                //     printf("calculateSquares tstattribute = %lf at index [%d] [%d]\n", testAttribute, testRowIndex, blockIdx.x * ROWS_AMOUNT + testRowIndex);
                //     printf("calculateSquares blockIdx %d threadIdx %d: changing dataset[0] from [%lf] to [%lf]\n", blockIdx.x, threadIdx.x, dataset[k], tmp*tmp);
                // }
                dataset[k] = tmp * tmp;
            }
        }
    
        __device__ void calculateSums(double* dataset)
        {
            if (blockIdx.x == 0)
            {
                return;
            }

            int firstAttributeIndex = threadIdx.x * TRAIN_SET_SIZE / blockDim.x;

            int thisThreadStart = threadIdx.x * TRAIN_SET_SIZE / blockDim.x + blockIdx.x * ROWS_AMOUNT;
            const int nextThreadStart = (threadIdx.x + 1) * TRAIN_SET_SIZE / blockDim.x + blockIdx.x * ROWS_AMOUNT;

            // // Sum each row & calculate square root
            for (uint32_t k = thisThreadStart; k < nextThreadStart; ++k)
            {
                // if (firstAttributeIndex == 0)
                // {
                    // printf("calculateSums blockIdx %d threadIdx %d: changing dataset[0] adding [%lf]\n", blockIdx.x, threadIdx.x, dataset[k]);
                // }
                atomicAdd(&(dataset[firstAttributeIndex++]), dataset[k]);
                // if (firstAttributeIndex == 1)
                // {
                    // printf("calculateSums blockIdx %d threadIdx %d: changed dataset[0] to [%lf]\n", blockIdx.x, threadIdx.x, dataset[firstAttributeIndex - 1]);
                // }
            }
        }
    
        __device__ void calculateSquaredRoots(double* dataset)
        {
            // Split only first row (train data) into blocks; then splitted rows into block split into threads
            int thisThreadStart = threadIdx.x * TRAIN_SET_SIZE / gridDim.x / blockDim.x + blockIdx.x * TRAIN_SET_SIZE / gridDim.x;
            int nextThreadStart = (threadIdx.x + 1) * TRAIN_SET_SIZE / gridDim.x / blockDim.x + blockIdx.x * TRAIN_SET_SIZE / gridDim.x;

            // Sum each row & calculate square root
            for (uint32_t k = thisThreadStart; k < nextThreadStart; ++k)
            {
                // if (k == 0)
                // {
                    // printf("calculateSquaredRoots blockIdx %d threadIdx %d: changing dataset[0] from [%lf] to [%lf]\n", blockIdx.x, threadIdx.x, dataset[k], sqrt(dataset[k]));
                // }
                dataset[k] = sqrt(dataset[k]);
            }
        }

        __device__ void findLocalMinSumWithIndex(double* dataset, double* mins, int* indices)
        {
            // Split only first row (train data) into blocks; then splitted rows into block split into threads
            
            int thisThreadStart = threadIdx.x * TRAIN_SET_SIZE / gridDim.x / blockDim.x + blockIdx.x * TRAIN_SET_SIZE / gridDim.x;
            const int nextThreadStart = (threadIdx.x + 1) * TRAIN_SET_SIZE / gridDim.x / blockDim.x + blockIdx.x * TRAIN_SET_SIZE / gridDim.x;
            double localMin = dataset[thisThreadStart];
            int localMinIndex = thisThreadStart;
            // printf("findLocalMinSumWithIndex blockIdx %d threadIdx %d | range [%d:%d] | init localMin %lf at index %d\n", blockIdx.x, threadIdx.x, thisThreadStart, nextThreadStart, localMin, localMinIndex);
            __syncthreads();
            for (int row = thisThreadStart; row < nextThreadStart; ++row)
            {
                // if (row < 10)
                // printf("row details blockIdx %d threadIdx %d = min %lf at index %d\n", blockIdx.x, threadIdx.x, dataset[row], row);
                auto value = dataset[row];
                if (value < localMin)
                {
                    localMin = value;
                    localMinIndex = row;
                }
            }

            // if (blockIdx.x == 0 && threadIdx.x < 10)
            // {
                // printf("Local min blockIdx %d threadIdx %d = %lf at index %d\n", blockIdx.x, threadIdx.x, localMin, localMinIndex);
            // }
            mins[threadIdx.x] = localMin;
            indices[threadIdx.x] = localMinIndex;
        }

        __device__ void findMinSumWithIndex(double* min, int* minIndex, double* localMin, int* localMinIndices)
        {
            for (int i = 0; i < blockDim.x; ++i)
            {
                auto localMinValue = localMin[i];
                if (*min > localMinValue)
                {
                    *min = localMinValue;
                    *minIndex = localMinIndices[i];
                }
            }
        }

        __global__ void knn(double* dataset, int testRowIndex, double* min, int* minIndex)
        {
            grid_group grid = this_grid();
            calculateSquares(dataset, testRowIndex);
            __syncthreads();
            calculateSums(dataset);
            __syncthreads();
            grid.sync();
            calculateSquaredRoots(dataset);
            __syncthreads();

            {
            __shared__ double localMin[BLOCK_DIM];
            __shared__ int localMinIndices[BLOCK_DIM];
            findLocalMinSumWithIndex(dataset, localMin, localMinIndices);
            __syncthreads();

            findMinSumWithIndex(min, minIndex, localMin, localMinIndices);
            __syncthreads();
            } // scoped shared memory variables
        }
    }
}

auto LetterRecognition::fetchData(const string& path) -> LetterData
{
    const uint32_t MATRIX_SIZE = ATTRIBUTES_AMOUNT + 1; // attributes + its class

    LetterData data;
    vector<vector<double>> matrix;
    matrix.resize(ATTRIBUTES_AMOUNT);
    for (auto& attributes : matrix)
    {
        attributes.reserve(ROWS_AMOUNT);
    }
    data.letters.reserve(ROWS_AMOUNT);
    data.attributesAmount = ATTRIBUTES_AMOUNT;

    ifstream file(path.c_str());
    string line;

    while (getline(file, line))
    {
        stringstream stream(line);
        string stringValue; // represents double value
        uint32_t position = 0;
        while (getline(stream, stringValue, ','))
        {
            if (position == 0)
            {
                data.letters.push_back(stringValue.at(0));
            } else {
                matrix.at(position - 1).push_back(stod(stringValue));
            }
            position = (position + 1) % MATRIX_SIZE;
        }
    }

    // Flatten 2D array
    data.attributes.reserve(ATTRIBUTES_AMOUNT * ROWS_AMOUNT);

    for (auto& column : matrix)
    {
        for (auto& value : column)
        {
            data.attributes.push_back(value);
        }
    }

    return data;
}

void LetterRecognition::Result::printOverallResult()
{
    double percentage = static_cast<double>(correct) / static_cast<double>(all) * 100.0;
    std::cout << "Accuracy: " << correct << "/" << all
        << ", Percentage: " << percentage << "%" << std::endl;
}

auto LetterRecognition::knn(LetterData& letterData) -> Result
{
	double* devAttributes;
	HANDLE_ERROR(cudaMalloc(&devAttributes, ATTRIBUTES_AMOUNT * ROWS_AMOUNT * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(devAttributes, letterData.attributes.data(), ATTRIBUTES_AMOUNT * ROWS_AMOUNT * sizeof(double), cudaMemcpyHostToDevice));
    // Copy dataset for each test row
    Result result{0, 0};
    for (int i = TRAIN_SET_SIZE; i < TRAIN_SET_SIZE + TEST_SET_SIZE; ++i)
    {
        double* dataset = nullptr;
        HANDLE_ERROR(cudaMalloc(&dataset, ATTRIBUTES_AMOUNT * ROWS_AMOUNT * sizeof(double)));
        cudaDeviceSynchronize();
        HANDLE_ERROR(cudaMemcpy(dataset, devAttributes, ATTRIBUTES_AMOUNT * ROWS_AMOUNT * sizeof(double), cudaMemcpyDeviceToDevice));
        cudaDeviceSynchronize();

        double* devMinValue;
        HANDLE_ERROR(cudaMalloc(&devMinValue, sizeof(double)));
        HANDLE_ERROR(cudaMemset(devMinValue, 1000000.0, sizeof(double))); // randomly high value
        int* devMinIndex;
        HANDLE_ERROR(cudaMalloc(&devMinIndex, sizeof(int)));

        dim3 dimGrid(ATTRIBUTES_AMOUNT, 1, 1);
        dim3 dimBlock(BLOCK_DIM, 1, 1);
        void *kernelArgs[] = {
            (void *)&dataset,  (void *)&i, (void *)&devMinValue, (void *)&devMinIndex,
        };
        int sharedMemorySize = BLOCK_DIM * (sizeof(double) + sizeof(int));
        HANDLE_ERROR(cudaLaunchCooperativeKernel((void*)GPU::knn, dimGrid, dimBlock, kernelArgs, sharedMemorySize, nullptr));
        HANDLE_ERROR(cudaFree(devMinValue));

        int minIndex{};
        HANDLE_ERROR(cudaMemcpy(&minIndex, devMinIndex, sizeof(int), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaFree(devMinIndex));

        char predictedGenre = letterData.letters[minIndex];
        // cout << "predictedGenre index " << minIndex << endl;
        auto actualGenre = letterData.letters[i];
        if (predictedGenre == actualGenre)
            result.correct++;

        cudaDeviceSynchronize();
        HANDLE_ERROR(cudaFree(dataset));
    }

    result.all = TEST_SET_SIZE;

    HANDLE_ERROR(cudaFree(devAttributes));

    return result;
}

// auto LetterRecognition::knn(LetterData& letterData, uint32_t neighbours) -> Result
// {
//     const uint32_t TRAIN_SET_SIZE = ROWS_AMOUNT * 0.9;
//     const uint32_t TEST_SET_SIZE = ROWS_AMOUNT - TRAIN_SET_SIZE;

//     uint32_t i;
//     vector<double> dataset;
//     Result result{0, 0, {}};
//     for (i = TRAIN_SET_SIZE; i < TRAIN_SET_SIZE + TEST_SET_SIZE; ++i)
//     {
//         // Copy dataset for each test row
//         dataset = letterData.attributes;

//         // Calculate squares for every attribute
//         uint32_t j;
//         for (j = 0; j < letterData.attributesAmount; ++j)
//         {
//             double testAttribute = dataset.at(j * ROWS_AMOUNT + i);
//             uint32_t k;
//             for (k = 0; k < TRAIN_SET_SIZE; ++k)
//             {
//                 double tmp = testAttribute - dataset.at(j * ROWS_AMOUNT + k);
//                 dataset.at(j * ROWS_AMOUNT + k) = tmp * tmp;
//             }
//         }
        
//         set<pair<double, char>> nearestNeighbours;
//         uint32_t k;
//         // Sum each row & calculate square root
//         for (k = 0; k < TRAIN_SET_SIZE; ++k)
//         {
//             double sum = 0.0;
//             char genre = '0';
//             uint32_t a;
//             for (a = 0; a < ATTRIBUTES_AMOUNT; ++a)
//             {
//                 sum += dataset.at(a * ROWS_AMOUNT + k);
//             }

//             sum = sqrt(sum);
//             genre = letterData.letters.at(k);

//             if (k < neighbours)
//             {
//                 nearestNeighbours.emplace(make_pair(sum, genre));
//             }
//             else if ((*--nearestNeighbours.end()).first > sum)
//             {
//                 nearestNeighbours.erase(--nearestNeighbours.end());
//                 nearestNeighbours.emplace(make_pair(sum, genre));
//             }
//         }

//         // Vote/decide which neighbour
//         auto predictedGenre = voteOnGenre(nearestNeighbours);
//         auto actualGenre = letterData.letters.at(i);       

//         // std::cout << "Actual:\t" << actualGenre << ", predicted\t" << predictedGenre << std::endl;
//         if (predictedGenre == actualGenre)
//         {
//             ++result.correct;
//         }
//     }

//     result.all = TEST_SET_SIZE;

//     return result;
// }

// char LetterRecognition::voteOnGenre(const set<pair<double, char>>& nearestNeighbours)
// {
//     map<char, uint32_t> occurencesMap;

//     for (const auto& entry : nearestNeighbours)
//     {
//         char neighbour = entry.second;
//         if (occurencesMap.find(neighbour) == occurencesMap.end())
//         {
//             occurencesMap[neighbour] = 1;
//         } else {
//             ++occurencesMap[neighbour];
//         }
//     }

//     char chosenChar;
//     uint32_t chosenCharOccurences = 0;
//     for (const auto& entry : occurencesMap)
//     {
//         // cout << entry.first << ":" << entry.second << endl;
//         if (entry.second > chosenCharOccurences)
//         {
//             chosenChar = entry.first;
//             chosenCharOccurences = entry.second;
//         }
//     }

//     return chosenChar;
// }
