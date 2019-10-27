#pragma once

#include <vector>
#include <utility>

#include "MpiWrapper.hpp"

namespace
{
    using namespace std;
}

class Scalers
{
public:
    Scalers(MpiWrapper& mpi) : ROWS_AMOUNT{20000}, mpiWrapper{mpi} {}
    void normalize(vector<double> &attributeSet);
    void standarize(vector<vector<double>>* attributeSet, int index);
private:
    const int ROWS_AMOUNT;
    MpiWrapper& mpiWrapper;

    pair<double, double> findMinMax(vector<double> &attributeSet);
    pair<double, double> findAverageAndVariation(vector<double> &attributeSet);
};
