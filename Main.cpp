#include <cstdint>
#include <iostream>
#include <omp.h>

#include "Stopwatch.hpp"
#include "LetterRecognition.hpp"
#include "Scalers.hpp"

namespace
{
    using namespace std;
}

int main()
{
    LetterRecognition letterRecognition;
    Scalers scalers;
    StopWatch timer;
    const string DATASET_PATH{"csv/letter-recognition.csv"};

    auto letterData = letterRecognition.fetchData(DATASET_PATH);

    uint32_t i;
    // #pragma omp parallel for shared(letterData) private(i) num_threads(2)
    #pragma omp parallel for shared(letterData) private(i)
    for (i = 0; i < letterData.attributesAmount; ++i)
    {
        scalers.normalize(letterData.attributes.at(i));
        // scalers.standarize(letterData.attributes.at(i));
    }

    timer.start();
    auto results = letterRecognition.knn(letterData);
    // auto results = letterRecognition.knn(letterData, NEIGHBOURS); // TODO
    timer.stop();
    timer.displayTime();
    
    results.print();

    return 0;
}
