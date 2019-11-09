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

auto LetterRecognition::fetchData(const string &path) -> LetterData
{
    LetterData data;
    data.attributes.resize(ATTRIBUTES);
    for (auto &attributes : data.attributes)
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
            }
            else
            {
                data.attributes.at(position - 1).push_back(stod(stringValue));
            }
            position = (position + 1) % MATRIX_SIZE;
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

void LetterRecognition::Result::printConfustionMatrix()
{
    for (const auto &entry : confusionMatrix)
    {
        double percentage = static_cast<double>(entry.second.first) / static_cast<double>(entry.second.first + entry.second.second) * 100.0;
        std::cout << "Letter: " << entry.first << ",\tpercentage: " << percentage << "%,\tcorrect: " << entry.second.first << ",\tincorrect: " << entry.second.second << std::endl;
    }
}

