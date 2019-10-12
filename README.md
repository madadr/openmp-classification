# Classification app
- Solution for classification problem solved in two languages: C++ (parallel with OpenMP) and Python (with sklearn).
- Letter recognition dataset (UCI) - https://archive.ics.uci.edu/ml/datasets/Letter+Recognition
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

## Dataset description
- Letter recognition dataset
- https://archive.ics.uci.edu/ml/datasets/Letter+Recognition
- CSV format:
  1. lettr capital letter (26 values from A to Z)
  2. x-box horizontal position of box (integer)
  3. y-box vertical position of box (integer)
  4. width width of box (integer)
  5. high height of box (integer)
  6. onpix total # on pixels (integer)
  7. x-bar mean x of on pixels in box (integer)
  8. y-bar mean y of on pixels in box (integer)
  9. x2bar mean x variance (integer)
  10. y2bar mean y variance (integer)
  11. xybar mean x y correlation (integer)
  12. x2ybr mean of x * x * y (integer)
  13. xy2br mean of x * y * y (integer)
  14. x-ege mean edge count left to right (integer)
  15. xegvy correlation of x-ege with y (integer)
  16. y-ege mean edge count bottom to top (integer)
  17. yegvx correlation of y-ege with x (integer)

