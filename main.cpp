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

static constexpr uint32_t ATTRIBUTES = 11;
static constexpr uint32_t MATRIX_SIZE = ATTRIBUTES + 1; // attributes + its class

// static constexpr uint32_t TRAIN_SET_SIZE = 5;
static constexpr uint32_t TRAIN_SET_SIZE = 1000;
// static constexpr uint32_t TEST_SET_SIZE = 5;
static constexpr uint32_t TEST_SET_SIZE = 499;
}

vector<vector<double>> fetchDatasetFromFile()
{
    // Wine quality dataset
    // https://archive.ics.uci.edu/ml/datasets/Wine+Quality
    // CSV format:
    // "fixed acidity";"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"
    vector<vector<double>> values{MATRIX_SIZE};

    ifstream file("csv/winequality-red.csv");
    string line;

    while (getline(file, line))
    {
        stringstream stream(line);
        string stringValue; // represents double value
        int position = 0;
        while (getline(stream, stringValue, ';'))
        {
            values.at(position).push_back(stod(stringValue));
            position = (position + 1) % MATRIX_SIZE;
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

void knn(vector<vector<double>> dataset)
{
    int correct{};
    int minimalDistance{};
    int minimalDistanceIndex{};
    for (int i = TRAIN_SET_SIZE; i < TRAIN_SET_SIZE + TEST_SET_SIZE; ++i)
    {
        // Calculate squares for every attribute
        for (int j = 0; j < ATTRIBUTES; ++j)
        {
            double testAttribute = dataset.at(j).at(i);
            for (int k = 0; k < TRAIN_SET_SIZE; ++k)
            {
                double tmp = testAttribute - dataset.at(j).at(k);
                dataset.at(j).at(k) = tmp * tmp;                
            }
        }

        double minimalSum;
        double genre;
        // Sum each row & calculate square root
        for (int k = 0; k < TRAIN_SET_SIZE; ++k)
        {
            double sum = 0.0;
            for (int j = 0; j < ATTRIBUTES; ++j)
            {
                sum += dataset.at(j).at(k);
            }

            sum = sqrt(sum);

            // std::cout << "sum = " << sum << std::endl;
            // std::cout << "minimalSum = " << minimalSum << std::endl;
            // std::cout << "genre bfr = " << genre << std::endl;
            if (k == 0 || sum < minimalSum)
            {
                minimalSum = sum;
                // std::cout << "minimalSum aft = " << minimalSum << std::endl;
                genre = dataset.at(MATRIX_SIZE - 1).at(k);
                // std::cout << "genre aft = " << genre << std::endl;
            }
        }

        // std::cout << "Row i = " << i << " detected as \n\t" << genre << "\nactual\n\t" << dataset.at(MATRIX_SIZE - 1).at(i) << std::endl;
        if (genre == dataset.at(MATRIX_SIZE - 1).at(i))
        {
            ++correct;
        }
    }

    std::cout << "Accuracy: " << correct << "/" << TEST_SET_SIZE
        << "\nPercentage: " << static_cast<double>(correct) / static_cast<double>(TEST_SET_SIZE) * 100.0 << "%" << std::endl;
}

int main()
{
    auto dataset = fetchDatasetFromFile();

    // #pragma omp parallel for
    for (int i = 0; i < ATTRIBUTES; ++i)
    {
        normalize(dataset.at(i));
        // standarize(dataset.at(i));
    }

    knn(dataset);

    // #pragma omp parallel for
    // for (const auto &a : dataset)
    //     for (const auto &x : a)
    //         cout << x << " ";

    return 0;
}
