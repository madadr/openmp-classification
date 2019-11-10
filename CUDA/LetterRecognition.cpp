#include "LetterRecognition.hpp"

#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdint>
#include <set>

namespace
{
    using namespace std;
}

auto LetterRecognition::fetchData(const string& path) -> LetterData
{
    LetterData data;
    vector<vector<double>> matrix;
    matrix.resize(ATTRIBUTES);
    for (auto& attributes : matrix)
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
                matrix.at(position - 1).push_back(stod(stringValue));
            }
            position = (position + 1) % MATRIX_SIZE;
        }
    }

    // Flatten 2D array
    data.attributes.reserve(ATTRIBUTES * SET_SIZE);

    for (auto& column : matrix)
    {
        for (auto& element : column)
        {
            data.attributes.push_back(element);
        }
    }

    return data;
}

void LetterRecognition::Result::printOverallResult()
{
    double percentage = static_cast<double>(correct) / static_cast<double>(all) * 100.0;
    std::cout << "Accuracy: " << correct << "/" << all
        << ", Percentage: " << percentage << "%" << std::endl;
}

auto LetterRecognition::knn(LetterData& letterData) -> Result
{
    const uint32_t TRAIN_SET_SIZE = SET_SIZE * 0.9;
    const uint32_t TEST_SET_SIZE = SET_SIZE - TRAIN_SET_SIZE;

    uint32_t i;
    vector<double> dataset;
    Result result{0, 0};
    for (i = TRAIN_SET_SIZE; i < TRAIN_SET_SIZE + TEST_SET_SIZE; ++i)
    {
        // Copy dataset for each test row
        dataset = letterData.attributes;

        // Calculate squares for every attribute
        uint32_t j;
        for (j = 0; j < letterData.attributesAmount; ++j)
        {
            double testAttribute = dataset.at(j * SET_SIZE + i);
            uint32_t k;
            for (k = 0; k < TRAIN_SET_SIZE; ++k)
            {
                int index = j * SET_SIZE + k;
                double tmp = testAttribute - dataset.at(index);
                dataset.at(index) = tmp * tmp;
            }
        }
        
        uint32_t k;
        double minimalSum;
        char predictedGenre = '0';
        // Sum each row & calculate square root
        for (k = 0; k < TRAIN_SET_SIZE; ++k)
        {
            double sum = 0.0;
            uint32_t a;
            for (a = 0; a < ATTRIBUTES; ++a)
            {
                sum += dataset.at(a * SET_SIZE + k);
            }

            sum = sqrt(sum);

            if (k == 0 || sum < minimalSum)
            {
                minimalSum = sum;
                predictedGenre = letterData.letters.at(k);
            }
        }

        auto actualGenre = letterData.letters.at(i);

        if (predictedGenre == actualGenre)
        {
            ++result.correct;
        }
    }

    result.all = TEST_SET_SIZE;

    return result;
}

// auto LetterRecognition::knn(LetterData& letterData, uint32_t neighbours) -> Result
// {
//     const uint32_t TRAIN_SET_SIZE = SET_SIZE * 0.9;
//     const uint32_t TEST_SET_SIZE = SET_SIZE - TRAIN_SET_SIZE;

//     uint32_t i;
//     vector<double> dataset;
//     Result result{0, 0, {}};
//     for (i = TRAIN_SET_SIZE; i < TRAIN_SET_SIZE + TEST_SET_SIZE; ++i)
//     {
//         // Copy dataset for each test row
//         dataset = letterData.attributes;

//         // Calculate squares for every attribute
//         uint32_t j;
//         for (j = 0; j < letterData.attributesAmount; ++j)
//         {
//             double testAttribute = dataset.at(j * SET_SIZE + i);
//             uint32_t k;
//             for (k = 0; k < TRAIN_SET_SIZE; ++k)
//             {
//                 double tmp = testAttribute - dataset.at(j * SET_SIZE + k);
//                 dataset.at(j * SET_SIZE + k) = tmp * tmp;
//             }
//         }
        
//         set<pair<double, char>> nearestNeighbours;
//         uint32_t k;
//         // Sum each row & calculate square root
//         for (k = 0; k < TRAIN_SET_SIZE; ++k)
//         {
//             double sum = 0.0;
//             char genre = '0';
//             uint32_t a;
//             for (a = 0; a < ATTRIBUTES; ++a)
//             {
//                 sum += dataset.at(a * SET_SIZE + k);
//             }

//             sum = sqrt(sum);
//             genre = letterData.letters.at(k);

//             if (k < neighbours)
//             {
//                 nearestNeighbours.emplace(make_pair(sum, genre));
//             }
//             else if ((*--nearestNeighbours.end()).first > sum)
//             {
//                 nearestNeighbours.erase(--nearestNeighbours.end());
//                 nearestNeighbours.emplace(make_pair(sum, genre));
//             }
//         }

//         // Vote/decide which neighbour
//         auto predictedGenre = voteOnGenre(nearestNeighbours);
//         auto actualGenre = letterData.letters.at(i);       

//         // std::cout << "Actual:\t" << actualGenre << ", predicted\t" << predictedGenre << std::endl;
//         if (predictedGenre == actualGenre)
//         {
//             ++result.correct;
//         }
//     }

//     result.all = TEST_SET_SIZE;

//     return result;
// }

// char LetterRecognition::voteOnGenre(const set<pair<double, char>>& nearestNeighbours)
// {
//     map<char, uint32_t> occurencesMap;

//     for (const auto& entry : nearestNeighbours)
//     {
//         char neighbour = entry.second;
//         if (occurencesMap.find(neighbour) == occurencesMap.end())
//         {
//             occurencesMap[neighbour] = 1;
//         } else {
//             ++occurencesMap[neighbour];
//         }
//     }

//     char chosenChar;
//     uint32_t chosenCharOccurences = 0;
//     for (const auto& entry : occurencesMap)
//     {
//         // cout << entry.first << ":" << entry.second << endl;
//         if (entry.second > chosenCharOccurences)
//         {
//             chosenChar = entry.first;
//             chosenCharOccurences = entry.second;
//         }
//     }

//     return chosenChar;
// }
