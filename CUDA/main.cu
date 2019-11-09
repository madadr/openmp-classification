#include <cstdint>
#include <iostream>
#include <cmath>
#include "Stopwatch.cuh"
#include "LetterRecognition.hpp"
#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )

namespace
{
    using namespace std;
}
static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),file, line);
        exit(EXIT_FAILURE);
    }
}

__global__ void fulfillMinMaxArray(double *minTab, double *maxTab) {
    for(int i=0; i<16; i++) {
        minTab[i] = 7;
        maxTab[i] = 7;
    }
}

__global__ void fulfillAverageVariationArray(double *average, double *variation) {
    for(int i=0; i<16; i++) {
        variation[i] = 0;
    }
}

__global__ void minMax(double *tab, int rows, int columns, double *minTab, double *maxTab) {

    int tid = threadIdx.x;
    int attr = blockIdx.x;
    int blockSize = blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("Block dim: %d   Block number: %d    Thread number: %d    i: %d   att num %d\n", blockSize, attr, tid, i, i%16);

    for (int i = tid*rows/blockSize + attr*rows; i < (tid+1)*rows/blockSize + attr*rows; i++) {
        if (tab[i] < minTab[attr]) {
            minTab[attr] = tab[i];
        }
        if (tab[i] > maxTab[attr]) {
            maxTab[attr] = tab[i];
        }
    }
}

__global__ void normalizeCUDA(double *tab, int rows, int columns,  double *minTab, double *maxTab, double *normalizedTab)
{
    int tid = threadIdx.x;
    int attr = blockIdx.x;
    int blockSize = blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Block dim: %d   Block number: %d    Thread number: %d    i: %d   att num %d\n", blockSize, attr, tid, i, i%16);

    double diff = maxTab[attr] - minTab[attr];

    for (int i = tid*rows/blockSize + attr*rows; i < (tid+1)*rows/blockSize + attr*rows; i++) {
        tab[i] = (tab[i] - minTab[attr]) / diff;
    }
}

__global__ void findAverageVariation(double *tab, int rows, int columns, double *average, double *variation) {

    int attr = blockIdx.x;
    int blockSize = blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("Block dim: %d   Block number: %d    Thread number: %d    i: %d   att num %d\n", blockSize, attr, tid, i, i%16);
    
    for (int i = attr*rows; i < attr*rows + rows; i++) {
        average[attr] += tab[i];
    }
    average[attr] /= rows;

    for (int i = attr*rows; i < attr*rows + rows; i++) {
        variation[attr] += (tab[i] - average[attr]) * (tab[i] - average[attr]);
    }
    variation[attr] /= rows;
    variation[attr] = sqrt(variation[attr]);
}

__global__ void standarizeCUDA(double *tab, int rows, int columns,  double *average, double *variation)
{
    int tid = threadIdx.x;
    int attr = blockIdx.x;
    int blockSize = blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Block dim: %d   Block number: %d    Thread number: %d    i: %d   att num %d\n", blockSize, attr, tid, i, i%16);

    for (int i = tid*rows/blockSize + attr*rows; i < (tid+1)*rows/blockSize + attr*rows; i++) {
        tab[i] = (tab[i] - average[attr]) / variation[attr];
    }
}

int main()
{
    StopWatch stopWatch = StopWatch();
    LetterRecognition letterRecognition = LetterRecognition();

    LetterRecognition::LetterData letterData;
    const string DATASET_PATH{"csv/letter-recognition.csv"};
    letterData = letterRecognition.fetchData(DATASET_PATH);

    // Przerzucenie danych z tab dwuwymiarowej do jednowymiarowej
    vector<double> oneDimensionLetterData;
    oneDimensionLetterData.reserve(16*20000);

    for (int col = 0; col < 16; col++)
    {
        for (int row = 0; row < 20000; row++)
        {
            oneDimensionLetterData.push_back(letterData.attributes[col][row]);
        }
    }
    cout<<"Rozm: "<<oneDimensionLetterData.size()<<endl;


    double *min, *max, *average, *variation, *data, *normalizedData;
    double mins[16], maxes[16], normalizedAttributes[16*20000], standarizedAttributes[16*20000], averageOfAttributes[16], variationOfAttributes[16];

    // Alokacja pamięci na GPU
    HANDLE_ERROR( cudaMalloc( (void**)&data, 16 * 20000 * sizeof(double) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&normalizedData, 16 * 20000 * sizeof(double) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&min, 16 * sizeof(double) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&max, 16 * sizeof(double) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&average, 16 * sizeof(double) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&variation, 16 * sizeof(double) ) );

    // Skopiowanie tablicy oneDimensionLetterData na GPU
    HANDLE_ERROR( cudaMemcpy( data, oneDimensionLetterData.data(), 16 * 20000 * sizeof(double), cudaMemcpyHostToDevice ) );

    fulfillMinMaxArray<<<1,1>>>(min, max);
    fulfillAverageVariationArray<<<1,1>>>(average, variation);

    stopWatch.start();
    // minMax<<<16,32>>>( data, 20000, 16, min, max );
    // normalizeCUDA<<<16,16>>>( data, 20000, 16, min, max, normalizedData);
    findAverageVariation<<<16,1>>>( data, 20000, 16, average, variation );
    standarizeCUDA<<<16,16>>>( data, 20000, 16, average, variation);

    // Skopiowanie tablicy c z GPU na CPU
    HANDLE_ERROR( cudaMemcpy( mins, min, 16 * sizeof(double), cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR( cudaMemcpy( maxes, max, 16 * sizeof(double), cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR( cudaMemcpy( averageOfAttributes, average, 16 * sizeof(double), cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR( cudaMemcpy( variationOfAttributes, variation, 16 * sizeof(double), cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR( cudaMemcpy( normalizedAttributes, data, 16*20000 * sizeof(double), cudaMemcpyDeviceToHost ) );
    HANDLE_ERROR( cudaMemcpy( standarizedAttributes, data, 16*20000 * sizeof(double), cudaMemcpyDeviceToHost ) );


    stopWatch.stop();

    for (int i=0; i<16; i++) {
        cout<<averageOfAttributes[i]<<"    "<<variationOfAttributes[i]<<endl;
    }

    // for (int i=0; i<16; i++) {
    //     cout<<mins[i]<<"  "<<maxes[i]<<endl;
    // }

    for (int i=0; i<16; i++) {
        cout<<"Standarized: "<<standarizedAttributes[i]<<endl;
    }
    
    stopWatch.displayTime();
    return 0;
}
