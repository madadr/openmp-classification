#include "MpiWrapper.hpp"

#include <mpi.h>

MpiWrapper::MpiWrapper()
{
  MPI_Init(nullptr, nullptr);
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
}

MpiWrapper::~MpiWrapper()
{
  MPI_Finalize();
}

int MpiWrapper::getWorldRank()
{
  return worldRank;
}

int MpiWrapper::getWorldSize()
{
  return worldSize;
}