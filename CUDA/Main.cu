#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <cstdint>
#include <iostream>

#include "LetterRecognition.hpp"
#include "Scalers.cuh"
#include "Stopwatch.cuh"
// #include "Stopwatch.hpp"

namespace
{
    using namespace std;
}

int main()
{
    LetterRecognition letterRecognition;
    Scalers scalers;
    Stopwatch timer;
    const string DATASET_PATH{"../csv/letter-recognition.csv"};

    auto letterData = letterRecognition.fetchData(DATASET_PATH);

    timer.start();
    scalers.normalize(letterData.attributes);
    timer.stop();
    timer.displayTime();
    for (int i = 0; i < 16; ++i)
    {
        cout << "min " << i << ": " << letterData.attributes[0 + i] << endl;
        cout << "max " << i << ": " << letterData.attributes[20000 + i] << endl;
    }

    // auto results = letterRecognition.knn(letterData);



    // letterRecognition.crossValidation(letterData, 5);
    // auto results = letterRecognition.knn(letterData, 5);
    
    // results.printConfustionMatrix();
    // results.printOverallResult();

    return 0;
}
