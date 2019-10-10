#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <utility>
#include "omp.h"

namespace
{
using namespace std;

static constexpr uint32_t ATTRIBUTES = 4;
static constexpr uint32_t MATRIX_SIZE = ATTRIBUTES + 1; // attributes + its class
} // namespace

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
        while (getline(stream, stringValue, ','))
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

std::pair<double, double> findMinMax(vector<double> &dataset)
{
    double min = dataset.at(0);
    double max = min;

    for (const auto& value : dataset)
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

    return std::make_pair(min, max);
}

void normalize(vector<double> &dataset)
{
    const auto [min, max] = findMinMax(dataset);

    double diff = max - min;

    for (auto& value : dataset)
    {
        value = (value - min) / diff;
    }
}

std::pair<double, double> findAverageAndVariation(vector<double> &dataset)
{
    double average{};
    
    for (const auto& value : dataset)
    {
        average += value;
    }
    average /= dataset.size();

    double variation{};
    for (const auto& value : dataset)
    {
        auto tmp = value - average;
        variation += tmp * tmp;
    }
    variation /= dataset.size(); // variance
    variation = sqrt(variation);

    return std::make_pair(average, variation);
}

void standarize(vector<double> &dataset)
{
    const auto [average, variation] = findAverageAndVariation(dataset);

    for (auto& value : dataset)
    {
        value = (value - average) / variation;
    }
}

void knn(vector<vector<double>> &dataset)
{
}

int main()
{
    auto dataset = fetchDatasetFromFile();

    // #pragma omp parallel for
    for (int i = 0; i < ATTRIBUTES; ++i)
    {
        // normalize(dataset.at(i));
        standarize(dataset.at(i));
    }

    knn(dataset);

    for (const auto &a : dataset)
        for (const auto &x : a)
            cout << x << " ";
    // #pragma omp parallel for
    // for (int i = 1; i <= 10; ++i)
    // {
    //     cout << i << endl;
    // }

    return 0;
}
