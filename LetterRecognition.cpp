#include "LetterRecognition.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdint>
#include <omp.h>

namespace
{
    using namespace std;
}

auto LetterRecognition::fetchData(const string& path) -> LetterData
{
    LetterData data;
    data.attributes.resize(ATTRIBUTES);
    for (auto& attributes : data.attributes)
    {
        attributes.reserve(SET_SIZE);
    }
    data.letters.reserve(SET_SIZE);
    data.attributesAmount = ATTRIBUTES;

    ifstream file(path.c_str());
    string line;

    while (getline(file, line))
    {
        stringstream stream(line);
        string stringValue; // represents double value
        uint32_t position = 0;
        while (getline(stream, stringValue, ','))
        {
            if (position == 0)
            {
                data.letters.push_back(stringValue.at(0));
            } else {
                data.attributes.at(position - 1).push_back(stod(stringValue));
            }
            position = (position + 1) % MATRIX_SIZE;
        }
    }

    return data;
}

void LetterRecognition::Result::print()
{
    double percentage = static_cast<double>(correct) / static_cast<double>(all) * 100.0;
    std::cout << "Accuracy: " << correct << "/" << all
        << "\nPercentage: " << percentage << "%" << std::endl;
}

auto LetterRecognition::knn(LetterData& letterData) -> Result
{
    const uint32_t TRAIN_SET_SIZE = SET_SIZE * 0.9;
    const uint32_t TEST_SET_SIZE = SET_SIZE - TRAIN_SET_SIZE;

    uint32_t i;
    vector<vector<double>> dataset;
    Result result{0, 0};
    // #pragma omp parallel for shared(result) private(i, dataset) schedule(static) num_threads(2)
    #pragma omp parallel for shared(result) private(i, dataset) schedule(static)
    for (i = TRAIN_SET_SIZE; i < TRAIN_SET_SIZE + TEST_SET_SIZE; ++i)
    {
        // Copy dataset for each test row
        dataset = letterData.attributes;

        // Calculate squares for every attribute
        uint32_t j;
        // #pragma omp parallel for private(j)
        for (j = 0; j < letterData.attributesAmount; ++j)
        {
            double testAttribute = dataset.at(j).at(i);
            uint32_t k;
            // #pragma omp parallel for private(k)
            for (k = 0; k < TRAIN_SET_SIZE; ++k)
            {
                double tmp = testAttribute - dataset.at(j).at(k);
                dataset.at(j).at(k) = tmp * tmp;
            }
        }
        
        uint32_t k;
        double minimalSum;
        char genre = '0';
        // #pragma omp parallel for shared(k, minimalSum, genre)
        // Sum each row & calculate square root
        for (k = 0; k < TRAIN_SET_SIZE; ++k)
        {
            double sum = 0.0;
            uint32_t a;
            // #pragma omp parallel for shared(dataset) private(a) reduction(+ : sum)
            for (a = 0; a < ATTRIBUTES; ++a)
            {
                sum += dataset.at(a).at(k);
            }

            sum = sqrt(sum);

            if (k == 0 || sum < minimalSum)
            {
                minimalSum = sum;
                genre = letterData.letters.at(k);
            }
        }

        // std::cout << "Row i = " << i << " detected as \n\t" << genre << "\nactual\n\t" << letters.at(i) << std::endl;
        if (genre == letterData.letters.at(i))
        {
            ++result.correct;
        }
    }

    result.all = TEST_SET_SIZE;

    return result;
}
