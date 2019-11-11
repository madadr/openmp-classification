# Parallel k-nearest neighbours algorithm
- Solution for KNN algorithm solved in parallel with two languages:
    - C++
        * OpenMP
        * MPI
        * CUDA
    - Python
        * sklearn
        * sklearn + MPI
- Letter recognition dataset (UCI)
    - `https://archive.ics.uci.edu/ml/datasets/Letter+Recognition`
- KNN algorithm was used with 2 data scalers:
    - minmax normalization
    - linear standarization.

## Specific requirements
- CUDA: GPU with compute compatibility at least 7.5; tested on GeForce RTX 2080 Ti

## Setup environment
### C++ (OpenMP, MPI, CUDA)
#### Install g++
```
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install g++
```

#### Install OpenMP
```
sudo apt-get install libomp-dev
```

#### Install MPI (Open MPI)
##### Recommended way
```
# Download MPI from https://www.open-mpi.org/software/ompi/v4.0/
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.2.tar.gz
tar -xvf openmpi-4.0.2.tar.gz
cd openmpi-4.0.2
# Follow instructions in INSTALL file
sudo ./configure
sudo make all install

# Before executing app
echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope # When receiving CMA support is not available due to restrictive ptrace settings when executing mpirun/mpiexec
```

##### Alternative way (faster, but not tested)
```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install libopenmpi-dev
sudo apt-get install g++
echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope # When receiving CMA support is not available due to restrictive ptrace settings when executing mpirun/mpiexec
```

#### Install CUDA
```
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
```

### Python
#### Install python
```
sudo apt install python3-pip
```

#### Install sklearn with dependencies
```
pip3 install -U scikit-learn[alldeps]
```

## Running apps
### Compile & execute C++ program
All C++ program have included running scripts in their directories with name `RunApp.sh`.
- `./OpenMP/RunApp.sh`
- `./MPI/RunApp.sh`
- `./CUDA/RunApp.sh`

### Execute Python program
```
python3 main.py
```

## Dataset description
- Letter recognition dataset
- `https://archive.ics.uci.edu/ml/datasets/Letter+Recognition`
- CSV format:
```
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
```
