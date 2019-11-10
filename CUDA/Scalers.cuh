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

struct Scalers
{
	void normalize(vector<double>& attributeSet);
    void standarize(vector<double> &attributeSet);
};
