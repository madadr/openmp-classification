// # Install g++ version 9 (c++20 experimental support)
// sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
// sudo apt-get update
// sudo apt-get upgrade
// sudo apt-get install g++-9

// # Install OpenMP
// sudo apt-get install libomp-dev

// # Compile & execute
// g++ -std=gnu++17 -fopenmp main.cpp -o main && ./main
// # or
// g++-9 -std=gnu++2a -fopenmp main.cpp -o main && ./main

// sudo apt install python3-pip
// pip3 install -U scikit-learn[alldeps]

#include <iostream>
#include "omp.h"

int main()
{
    #pragma omp parallel for
    for (int i = 1; i <= 10; ++i)
    {
        std::cout << i << std::endl;
    }

    return 0;
}