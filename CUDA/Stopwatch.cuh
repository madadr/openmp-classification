#pragma once

class StopWatch
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
