#include "Scalers.cuh"

#include <cmath>
#include <iostream>
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

__global__ void normalizeOnGPU( int *attributes, int *normalizedAttributes, double min, double diff ) {
    int tid = blockIdx.x; // Ten wątek przetwarza dane pod określonym indeksem
    normalizedAttributes[tid] = (attributes[tid] - min) / diff;
  }

void Scalers::normalizeCUDA(vector<double> &attributeSet)
{
    const auto minMax = findMinMax(attributeSet);
    int N = attributeSet.size();
    double diff = minMax[1] - minMax[0];
    cout<<N<< "  "<<diff<<endl;

    double *dev_att, *dev_normAtt;
    double dev_min, dev_diff;
    double *normalizedAttributes = new double [N];
  
    // Alokacja pamięci na GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_att, N * sizeof(double) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_normAtt, N * sizeof(double) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_min, sizeof(double) ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_diff, sizeof(double) ) );

    // Skopiowanie tablicy attributeSet na GPU
    HANDLE_ERROR( cudaMemcpy( dev_att, attributeSet.data(), N * sizeof(double), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_min, min, sizeof(double), cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_diff, min, sizeof(double), cudaMemcpyHostToDevice ) );

    // Normalizacja
    normalizeOnGPU<<<N,1>>>(dev_att, dev_normAtt, dev_min, dev_diff);

    // Skopiowanie wyniku z GPU na CPU
    HANDLE_ERROR( cudaMemcpy( normalizedAttributes, dev_normAtt, N * sizeof(int), cudaMemcpyDeviceToHost ) );

    // Zwolnienie pamięci alokowanej na GPU
    HANDLE_ERROR( cudaFree( dev_att ) );
    HANDLE_ERROR( cudaFree( dev_normAtt ) );
    HANDLE_ERROR( cudaFree( dev_min ) );
    HANDLE_ERROR( cudaFree( dev_diff ) );
    for(int i=0; i< sizeof(normalizedAttributes); i++) {
        cout<<i<<") "<<attributeSet[i]<<" -> "<<normalizedAttributes[i]<<endl;
    }
}


void Scalers::normalize(vector<double> &attributeSet)
{
    const auto minMax = findMinMax(attributeSet);

    double diff = minMax[1] - minMax[0];

    for (auto& value : attributeSet)
    {
        value = (value - minMax[0]) / diff;
    }
}

vector<double> Scalers::findMinMax(vector<double> &attributeSet)
{
    double min = attributeSet.at(0);
    double max = min;

    for (const auto& value : attributeSet)
    {
        if (value < min)
        {
            min = value;
        }
        
        if (value > max)
        {
            max = value;
        }
    }

    vector<double> minMax;
    minMax.push_back(min);
    minMax.push_back(max);
    return minMax;
}
