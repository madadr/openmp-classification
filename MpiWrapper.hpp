#pragma once

class MpiWrapper
{
public:
    MpiWrapper();
    ~MpiWrapper();

    int getWorldRank();
    int getWorldSize();
private:
    int worldRank{0};
    int worldSize{0};
};
