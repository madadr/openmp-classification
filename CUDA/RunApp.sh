echo "Cleaning"
rm app
echo "Compiling app"
nvcc="/usr/local/cuda/bin/nvcc"
$nvcc  -arch=sm_75 -Xcompiler -std=gnu++14 -rdc=true -o app Main.cu Scalers.cu LetterRecognition.cu Stopwatch.cu
echo "Running app"
./app
