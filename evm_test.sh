#!/bin/sh

rm -rf ./Results

echo "Testing results!"

# Motion Magnification
./build/evm-cpp-linux ./vids/machine1.mp4 1 100 90 4.0 6.2 30 1
#./build/evm-cpp-linux ./vids/machine2.mp4 1 60 90 3.6 6.2 30 0.3
#./build/evm-cpp-linux ./vids/machine3.mp4 1 70 90 3.6 6.2 30 1
#./build/evm-cpp-linux ./vids/machine4.mp4 1 25 90 0.5 3.5 30 0