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
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include "omp.h"

namespace
{
    using namespace std;

    static constexpr uint32_t ATTRIBUTES = 4;
    static constexpr uint32_t MATRIX_SIZE = ATTRIBUTES + 1; // attributes + its class
}
 
vector<vector<double>> fetchDatasetFromFile()
{
    // CSV format:
    // Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species
    // Species:
    // 1 = Iris-setosa
    // 2 = Iris-versicolor
    // 3 = Iris-virginica
    vector<vector<double>> values{MATRIX_SIZE};

    ifstream file("Iris_trainset.csv");
    string line;
   
    while (getline(file, line))
    {
        stringstream stream(line);
        string stringValue; // represents double value
        int position = 0; 
        while(getline(stream, stringValue, ','))
        {
            if (position != 0) // id is on first position - skip it
            {
                values.at(position - 1).push_back(stod(stringValue));
            }

            position = (position + 1) % 6;
        }
    }

    return values;
}

void normalize(vector<double>& dataset)
{

}

void standarize(vector<double>& dataset)
{

}

void knn(vector<vector<double>>& dataset)
{
    // print type
    // 
}

int main()
{
    auto dataset = fetchDatasetFromFile();

    for(const auto& a : dataset)
        for(const auto& x : a)
            cout << x << " ";
    // #pragma omp parallel for
    // for (int i = 1; i <= 10; ++i)
    // {
    //     cout << i << endl;
    // }

    return 0;
}
