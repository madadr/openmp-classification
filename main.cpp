#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <utility>
#include "omp.h"
#include "stopwatch.hpp"

namespace
{
using namespace std;

static constexpr uint ATTRIBUTES = 16;
static constexpr uint MATRIX_SIZE = ATTRIBUTES + 1; // attributes + its class

static constexpr uint SET_SIZE = 20000;

static constexpr uint TRAIN_SET_SIZE = SET_SIZE * 0.9;
static constexpr uint TEST_SET_SIZE = SET_SIZE * 0.1;

vector<char> letters;
}

vector<vector<double>> fetchDatasetFromFile();
void normalize(vector<double> &attributeSet);
std::pair<double, double> findMinMax(vector<double> &attributeSet);
void standarize(vector<double> &attributeSet);
std::pair<double, double> findAverageAndVariation(vector<double> &attributeSet);
void knn(vector<vector<double>>& inputDataset);

int main()
{
    StopWatch timer;

    letters.reserve(SET_SIZE);
    auto dataset = fetchDatasetFromFile();

    timer.start();
    // #pragma omp parallel for
    for (uint i = 0; i < ATTRIBUTES; ++i)
    {
        normalize(dataset.at(i));
        standarize(dataset.at(i));
    }

    knn(dataset);

    timer.stop();
    timer.displayTime();

    return 0;
}


vector<vector<double>> fetchDatasetFromFile()
{
    vector<vector<double>> values(ATTRIBUTES);

    ifstream file("csv/letter-recognition.csv");
    string line;

    while (getline(file, line))
    {
        stringstream stream(line);
        string stringValue; // represents double value
        uint position = 0;
        while (getline(stream, stringValue, ','))
        {
            if (position == 0)
            {
                letters.push_back(stringValue.at(0));
            } else {
                values.at(position - 1).push_back(stod(stringValue));
            }
            position = (position + 1) % MATRIX_SIZE;
        }
    }

    return values;
}

std::pair<double, double> findMinMax(vector<double> &attributeSet)
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

    return std::make_pair(min, max);
}

void normalize(vector<double> &attributeSet)
{
    const auto [min, max] = findMinMax(attributeSet);

    double diff = max - min;

    for (auto& value : attributeSet)
    {
        value = (value - min) / diff;
    }
}

std::pair<double, double> findAverageAndVariation(vector<double> &attributeSet)
{
    double average{};
    
    for (const auto& value : attributeSet)
    {
        average += value;
    }
    average /= attributeSet.size();

    double variation{};
    for (const auto& value : attributeSet)
    {
        auto tmp = value - average;
        variation += tmp * tmp;
    }
    variation /= attributeSet.size(); // variance
    variation = sqrt(variation);

    return std::make_pair(average, variation);
}

void standarize(vector<double> &attributeSet)
{
    const auto [average, variation] = findAverageAndVariation(attributeSet);

    for (auto& value : attributeSet)
    {
        value = (value - average) / variation;
    }
}

void knn(vector<vector<double>>& inputDataset)
{
    uint correct{};
    #pragma omp parallel for
    for (uint i = TRAIN_SET_SIZE; i < TRAIN_SET_SIZE + TEST_SET_SIZE; ++i)
    {
        // Copy dataset for each test row
        auto dataset = inputDataset;

        // Calculate squares for every attribute
        #pragma omp parallel for
        for (uint j = 0; j < ATTRIBUTES; ++j)
        {
            double testAttribute = dataset.at(j).at(i);
            for (uint k = 0; k < TRAIN_SET_SIZE; ++k)
            {
                double tmp = testAttribute - dataset.at(j).at(k);
                dataset.at(j).at(k) = tmp * tmp;                
            }
        }

        double minimalSum;
        char genre;
        // Sum each row & calculate square root
        for (uint k = 0; k < TRAIN_SET_SIZE; ++k)
        {
            double sum = 0.0;
            for (uint j = 0; j < ATTRIBUTES; ++j)
            {
                sum += dataset.at(j).at(k);
            }

            sum = sqrt(sum);

            if (k == 0 || sum < minimalSum)
            {
                minimalSum = sum;
                genre = letters.at(k);
            }
        }

        // std::cout << "Row i = " << i << " detected as \n\t" << genre << "\nactual\n\t" << letters.at(i) << std::endl;
        if (genre == letters.at(i))
        {
            ++correct;
        }
    }

    std::cout << "Accuracy: " << correct << "/" << TEST_SET_SIZE
        << "\nPercentage: " << static_cast<double>(correct) / static_cast<double>(TEST_SET_SIZE) * 100.0 << "%" << std::endl;
}