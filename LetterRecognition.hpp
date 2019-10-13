#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace
{
    using namespace std;
}

class LetterRecognition
{
    uint32_t SET_SIZE = 20000;
    uint32_t ATTRIBUTES = 16;
    uint32_t MATRIX_SIZE = ATTRIBUTES + 1; // attributes + its class
public:
    struct LetterData
    {
        vector<vector<double>> attributes;
        vector<char> letters;
        uint32_t attributesAmount;
    };

    struct Result
    {
        uint32_t correct;
        uint32_t all;
        // TODO: Confusion matrix

        void print();
    };

    LetterData fetchData(const string& path);
    Result knn(LetterData& letterData);
    // Result knn(LetterData& letterData, uint32_t neighbours); // TODO
};
