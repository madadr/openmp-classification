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

    static LetterData fetchData(const string& path);
    static Result knn(LetterData& letterData);
    // static Result knn(LetterData& letterData, uint32_t neighbours); // TODO
};
