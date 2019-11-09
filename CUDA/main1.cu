#include <stdio.h>
#include "Scalers.hpp"
#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )
#define N 10

static void HandleError(cudaError_t err, const char *file, int line)
{
  if (err != cudaSuccess)
{
  printf("%s in %s at line %d\n", cudaGetErrorString(err),file, line);
  exit(EXIT_FAILURE);
}
}

__global__ void add( int *a, int *b, int *c ) {
  int tid = blockIdx.x; // Ten wątek przetwarza dane pod określonym indeksem
  if (tid < N)
    c[tid] = a[tid] + b[tid];
}

int main( void ) {
  int a[N], b[N], c[N];
  int *dev_a, *dev_b, *dev_c;

  // Alokacja pamięci na GPU
  HANDLE_ERROR( cudaMalloc( (void**)&dev_a, N * sizeof(int) ) );
  HANDLE_ERROR( cudaMalloc( (void**)&dev_b, N * sizeof(int) ) );
  HANDLE_ERROR( cudaMalloc( (void**)&dev_c, N * sizeof(int) ) );


  // Zapełnienie tablic a i b na CPU
  for (int i=0; i<N; i++) {
    a[i] = -i;
    b[i] = i * i;
  }

  // Skopiowanie tablic a i b na GPU
  HANDLE_ERROR( cudaMemcpy( dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy( dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice ) );

  add<<<N,1>>>( dev_a, dev_b, dev_c );

  // Skopiowanie tablicy c z GPU na CPU
  HANDLE_ERROR( cudaMemcpy( c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost ) );

  // Wyświetlenie wyników
  for (int i=0; i<N; i++) {
    printf( "%d + %d = %d\n", a[i], b[i], c[i] );
  }

  // Zwolnienie pamięci alokowanej na GPU
  HANDLE_ERROR( cudaFree( dev_a ) );
  HANDLE_ERROR( cudaFree( dev_b ) );
  HANDLE_ERROR( cudaFree( dev_c ) );

  return 0;
}
