#include <cstdint>
#include <iostream>
#include <omp.h>

#include "MpiWrapper.hpp"
#include "Stopwatch.hpp"
#include "LetterRecognition.hpp"
#include "Scalers.hpp"

namespace
{
    using namespace std;
}

int main()
{
    MpiWrapper mpiWrapper;
    LetterRecognition letterRecognition;
    Scalers scalers(mpiWrapper);
    StopWatch timer;

    LetterRecognition::LetterData letterData;
    if (mpiWrapper.getWorldRank() == 0)
    {
        const string DATASET_PATH{"csv/letter-recognition.csv"};
        letterData = letterRecognition.fetchData(DATASET_PATH);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    timer.start();

    for (unsigned int i = 0; i < letterData.attributesAmount; ++i)
    {
        // scalers.normalize(valueSet);
        scalers.standarize(&letterData.attributes, i);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    timer.stop();
    timer.displayTime();
    // // TEST
    // if (mpiWrapper.getWorldRank() == 0)
    // {
    //     cout << letterData.attributes.at(0).at(0) << "\n[1] ";
    //     cout << letterData.attributes.at(0).at(1) << " ";
    //     cout << letterData.attributes.at(0).at(2) << " ";
    //     cout << letterData.attributes.at(0).at(3) << "\n[4999] ";
    //     cout << letterData.attributes.at(0).at(4999) << "\n[5000] ";
    //     cout << letterData.attributes.at(0).at(5000) << " ";
    //     cout << letterData.attributes.at(0).at(5001) << " ";
    //     cout << letterData.attributes.at(0).at(5002) << " ";
    //     cout << letterData.attributes.at(0).at(5003) << "\n";
    //     auto results = letterRecognition.knn(letterData);
    //     results.printOverallResult();
    // }
/*
    timer.start();
    // letterRecognition.crossValidation(letterData, 5);
    auto results = letterRecognition.knn(letterData);
    // auto results = letterRecognition.knn(letterData, 5);
    timer.stop();
    timer.displayTime();
*/  
    // results.printConfustionMatrix();
    // results.printOverallResult();

    // if (mpiWrapper.getWorldRank() == 0)
    // {
    //     auto results = letterRecognition.knn(letterData);
    //     results.printOverallResult();
    // }

    return 0;
}
