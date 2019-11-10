echo "Cleaning"
rm app
echo "Compiling app"
nvcc="/usr/local/cuda/bin/nvcc"
$nvcc -Xcompiler -std=gnu++14 -rdc=true -o app main.cu  LetterRecognition.cpp Stopwatch.cu
echo "Running app"
./app
