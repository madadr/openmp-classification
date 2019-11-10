#pragma once

class Stopwatch
{
private:
    cudaEvent_t startTime;
    cudaEvent_t stopTime;
    float time;
public:
    void start();
    void stop();
    void displayTime();
};
