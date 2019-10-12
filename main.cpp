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

static constexpr uint32_t ATTRIBUTES = 16;
static constexpr uint32_t MATRIX_SIZE = ATTRIBUTES + 1; // attributes + its class

static constexpr uint32_t SET_SIZE = 20000;

static constexpr uint32_t TRAIN_SET_SIZE = SET_SIZE * 0.9;
static constexpr uint32_t TEST_SET_SIZE = SET_SIZE * 0.1;

vector<char> letters;
}

vector<vector<double>> fetchDatasetFromFile()
{
    // Letter recognition dataset
    // https://archive.ics.uci.edu/ml/datasets/Letter+Recognition
    // CSV format:
    // 1. lettr capital letter (26 values from A to Z)
    // 2. x-box horizontal position of box (integer)
    // 3. y-box vertical position of box (integer)
    // 4. width width of box (integer)
    // 5. high height of box (integer)
    // 6. onpix total # on pixels (integer)
    // 7. x-bar mean x of on pixels in box (integer)
    // 8. y-bar mean y of on pixels in box (integer)
    // 9. x2bar mean x variance (integer)
    // 10. y2bar mean y variance (integer)
    // 11. xybar mean x y correlation (integer)
    // 12. x2ybr mean of x * x * y (integer)
    // 13. xy2br mean of x * y * y (integer)
    // 14. x-ege mean edge count left to right (integer)
    // 15. xegvy correlation of x-ege with y (integer)
    // 16. y-ege mean edge count bottom to top (integer)
    // 17. yegvx correlation of y-ege with x (integer)
    vector<vector<double>> values(ATTRIBUTES);

    ifstream file("csv/letter-recognition.data");
    string line;

    while (getline(file, line))
    {
        stringstream stream(line);
        string stringValue; // represents double value
        int position = 0;
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
    int correct{};
    int minimalDistance{};
    int minimalDistanceIndex{};
    for (int i = TRAIN_SET_SIZE; i < TRAIN_SET_SIZE + TEST_SET_SIZE; ++i)
    {
        // Copy dataset for each test row
        auto dataset = inputDataset;

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
        char genre;
        // Sum each row & calculate square root
        for (int k = 0; k < TRAIN_SET_SIZE; ++k)
        {
            double sum = 0.0;
            for (int j = 0; j < ATTRIBUTES; ++j)
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

int main()
{
    letters.reserve(SET_SIZE);
    auto dataset = fetchDatasetFromFile();

    // #pragma omp parallel for
    for (int i = 0; i < ATTRIBUTES; ++i)
    {
        // normalize(dataset.at(i));
        // standarize(dataset.at(i));
    }

    knn(dataset);

    // #pragma omp parallel for
    // for (const auto &a : dataset)
    //     for (const auto &x : a)
    //         cout << x << " ";

    return 0;
}
