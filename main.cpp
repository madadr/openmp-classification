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

    ifstream file("csv/Iris_train_and_test.csv");
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

    for (const auto& a : dataset)
        for (const auto& x : a)
            cout << x << " ";
    // #pragma omp parallel for
    // for (int i = 1; i <= 10; ++i)
    // {
    //     cout << i << endl;
    // }

    return 0;
}
