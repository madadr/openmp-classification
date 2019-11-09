echo "Cleaning"
rm app
echo "Compiling app"
nvcc="/usr/local/cuda/bin/nvcc"
$nvcc main.cu LetterRecognition.cpp Stopwatch.cu -o app
echo "Running app"
./app
