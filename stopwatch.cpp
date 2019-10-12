#include "stopwatch.hpp"

#include <iostream>

void StopWatch::start()
{
    startTime = std::chrono::high_resolution_clock::now();

}

void StopWatch::stop()
{
    endTime = std::chrono::high_resolution_clock::now();
}

void StopWatch::displayTime()
{
    std::chrono::duration<double, std::milli> duration = endTime - startTime;
    std::cout << "took " << duration.count() << " ms" << std::endl;
}
