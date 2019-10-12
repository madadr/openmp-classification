#include <cstdint>

#include "Stopwatch.hpp"
#include "LetterRecognition.hpp"
#include "Scalers.hpp"

namespace
{
    using namespace std;

    static const string DATASET_PATH{"csv/letter-recognition.csv"};
}

int main()
{
    auto letterData = LetterRecognition::fetchData(DATASET_PATH);

    StopWatch timer;
    timer.start();
    for (uint32_t i = 0; i < letterData.attributesAmount; ++i)
    {
        Scalers::normalize(letterData.attributes.at(i));
        Scalers::standarize(letterData.attributes.at(i));
    }

    auto results = LetterRecognition::knn(letterData);
    // auto results = LetterRecognition::knn(letterData, NEIGHBOURS); // TODO
    timer.stop();
    timer.displayTime();
    
    results.print();

    return 0;
}
