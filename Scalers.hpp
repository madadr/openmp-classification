#pragma once

#include <vector>
#include <utility>

namespace
{
    using namespace std;
}

class Scalers
{
    static pair<double, double> findMinMax(vector<double> &attributeSet);
    static pair<double, double> findAverageAndVariation(vector<double> &attributeSet);
public:
    static void normalize(vector<double> &attributeSet);
    static void standarize(vector<double> &attributeSet);
};
