#include "LetterRecognition.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

    const uint32_t BLOCK_DIM = 100;

    const uint32_t ROWS_AMOUNT = 20000;
    const uint32_t ATTRIBUTES_AMOUNT = 16;
	
    const uint32_t TRAIN_SET_SIZE = ROWS_AMOUNT * 0.9;
    const uint32_t TEST_SET_SIZE = 100;
    // const uint32_t TEST_SET_SIZE = ROWS_AMOUNT - TRAIN_SET_SIZE;

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
                atomicAdd(&(dataset[firstAttributeIndex++]), dataset[k]);
            }
        }
    
        __device__ void calculateSquaredRoots(double* dataset)
        {
            // Split only first row (train data) into blocks; then rows splitted into block split into threads
            int thisThreadStart = threadIdx.x * TRAIN_SET_SIZE / gridDim.x / blockDim.x + blockIdx.x * TRAIN_SET_SIZE / gridDim.x;
            int nextThreadStart = (threadIdx.x + 1) * TRAIN_SET_SIZE / gridDim.x / blockDim.x + blockIdx.x * TRAIN_SET_SIZE / gridDim.x;

            // // TODO: fix (done above?)
            // int thisThreadStart = blockIdx.x * TRAIN_SET_SIZE / gridDim.x;
            // const int nextThreadStart = (blockIdx.x + 1) * TRAIN_SET_SIZE / gridDim.x;

            // Sum each row & calculate square root
            for (uint32_t k = thisThreadStart; k < nextThreadStart; ++k)
            {
                dataset[k] = sqrt(dataset[k]);
            }
        }

        // __device__ void findLocalMinSumWithIndex(double* dataset, double* mins, int* indexes)
        // {
        //     int thisThreadStart = threadIdx.x * ROWS_AMOUNT / blockDim.x + blockIdx.x * ROWS_AMOUNT;
        //     const int nextThreadStart = (threadIdx.x + 1) * ROWS_AMOUNT / blockDim.x + blockIdx.x * ROWS_AMOUNT;
        //     double localMin = devAttributes[thisThreadStart];
        //     double localMax = localMin;
        //     __syncthreads();
        //     for (int row = thisThreadStart; row < nextThreadStart; ++row)
        //     {
        //         auto value = devAttributes[row];
        //         if (value < localMin)
        //         {
        //             localMin = value;
        //         }
                
        //         if (value > localMax)
        //         {
        //             localMax = value;
        //         }
        //     }

        //     mins[threadIdx.x] = localMin;
        //     maxes[threadIdx.x] = localMax;
        // }

        // __device__ void findMinSumWithIndex(double* min, double* minIndex, double* localMin, int* localMinIndices)
        // {
        //     if (threadIdx.x == 0)
        //     {
        //         *min = localMin[0];
        //         *max = localMax[0];
        //     }
        //     __syncthreads();

        //     for (int i = 0; i < blockDim.x; ++i)
        //     {
        //         auto localMinValue = localMin[i];
        //         if (*min > localMinValue)
        //         {
        //             *min = localMinValue;
        //         }
        //         auto localMaxValue = localMax[i];
        //         if (*max < localMaxValue)
        //         {
        //             *max = localMaxValue;
        //         }
        //     }
        // }

        __global__ void knn(double* dataset, int testRowIndex, char* predictedGenre)
        {
            calculateSquares(dataset, testRowIndex);
            __syncthreads();
            calculateSums(dataset);
            __syncthreads();
            calculateSquaredRoots(dataset);
            __syncthreads();

            // __shared__ double min;
            // __shared__ int minIndex;
            // {
            // __shared__ double localMin[BLOCK_DIM];
            // __shared__ double localMinIndixes[BLOCK_DIM];
            // findLocalMinMax(devAttributes, localMin, localMax);
            // __syncthreads();

            // findMinMax(&min, &max, localMin, localMax);
            // __syncthreads();
            // } // scoped shared memory variable localMin and localMax to save memory


            // uint32_t k;
            // double minimalSum;
            // char predictedGenre = '0';
            // // Sum each row & calculate square root
            // for (k = 0; k < TRAIN_SET_SIZE; ++k)
            // {
            //     double sum = 0.0;
            //     uint32_t a;
            //     for (a = 0; a < ATTRIBUTES_AMOUNT; ++a)
            //     {
            //         sum += dataset.at(a * ROWS_AMOUNT + k);
            //     }
    
            //     sum = sqrt(sum);
    
            //     if (k == 0 || sum < minimalSum)
            //     {
            //         minimalSum = sum;
            //         predictedGenre = letterData.letters.at(k);
            //     }
            // }
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
        while (column.size() > 0)
        {
            // Always save memory. Anywhere. Always.
            data.attributes.push_back(column[0]);
            column.front() = std::move(column.back());
            column.pop_back();
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
    // char* devLetters;
	// HANDLE_ERROR(cudaMalloc(&devLetters, letterData.letters.size() * sizeof(char)));
    // HANDLE_ERROR(cudaMemcpy(devLetters, letterData.letters.data(), letterData.letters.size() * sizeof(char), cudaMemcpyHostToDevice));
	double* devAttributes;
	HANDLE_ERROR(cudaMalloc(&devAttributes, ATTRIBUTES_AMOUNT * ROWS_AMOUNT * sizeof(double)));
    HANDLE_ERROR(cudaMemcpy(devAttributes, letterData.attributes.data(), ATTRIBUTES_AMOUNT * ROWS_AMOUNT * sizeof(double), cudaMemcpyHostToDevice));
    // Copy dataset for each test row
    Result result{0, 0};
    for (int i = TRAIN_SET_SIZE; i < TRAIN_SET_SIZE + TEST_SET_SIZE; ++i)
    {
        // cout << "i = " << i << endl;
        // double* dataset = nullptr;
        // HANDLE_ERROR(cudaMalloc(&dataset, ATTRIBUTES_AMOUNT * ROWS_AMOUNT * sizeof(double)));
        // cudaDeviceSynchronize();
        // HANDLE_ERROR(cudaMemcpy(dataset, devAttributes, ATTRIBUTES_AMOUNT * ROWS_AMOUNT * sizeof(double), cudaMemcpyDeviceToDevice));
        // cudaDeviceSynchronize();

        char predictedGenre = '-';
        GPU::knn<<<ATTRIBUTES_AMOUNT, BLOCK_DIM>>>(devAttributes, i, &predictedGenre);

        auto actualGenre = letterData.letters[i];
        if (predictedGenre == actualGenre)
            result.correct++;

        cudaDeviceSynchronize();
        // HANDLE_ERROR(cudaFree(dataset));
    }

    result.all = TEST_SET_SIZE;

    HANDLE_ERROR(cudaFree(devAttributes));
    // HANDLE_ERROR(cudaFree(devLetters));

    // return result;
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
