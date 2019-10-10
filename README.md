# Iris dataset classification app
- Solution for classification problem solved in two languages: C++ (parallel with OpenMP) and Python (with sklearn).
- Iris dataset (UCI).
- KNN algorithm was used with 2 data preparation variants: data minmax normalization and linear standarization.

## Environment preparation
### C++ & OpenMP
#### Install g++ version 9 (c++20 experimental support)
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install g++-9 # v7 should be enough

#### Install OpenMP
sudo apt-get install libomp-dev

### Python
#### Install python
sudo apt install python3-pip

#### Install sklearn with dependencies
pip3 install -U scikit-learn[alldeps]

## Running apps
### Compile & execute C++ program
g++ -std=gnu++17 -fopenmp main.cpp -o main && ./main
or
g++-9 -std=gnu++2a -fopenmp main.cpp -o main && ./main

### Execute Python program
TBD

