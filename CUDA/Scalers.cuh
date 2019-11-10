#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
//#include <device_functions.h>

//#include <helper_functions.h>
//#include <helper_cuda.h>

#include <iostream>
#include <cstdio>
#include <cstdlib>

#include <vector>
#include <utility>

namespace
{
    using namespace std;
}

class Scalers
{
	// vector<double*> transformToRawPointer(vector<vector<double>>& attributes);
    //pair<double, double> findMinMax(vector<double> &attributeSet);
    // pair<double, double> findAverageAndVariation(vector<double> &attributeSet);    
public:
	void normalize(vector<double>& attributeSet);
    void standarize(vector<double> &attributeSet);
};
