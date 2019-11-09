#pragma once

#include <vector>
#include <utility>

namespace
{
    using namespace std;
}

class Scalers
{
public:
    Scalers() : ROWS_AMOUNT{20000} {}
    void normalize(vector<double> &attributeSet);
    void normalizeCUDA(vector<double> &attributeSet);
private:
    const int ROWS_AMOUNT;

    vector<double> findMinMax(vector<double> &attributeSet);
};
