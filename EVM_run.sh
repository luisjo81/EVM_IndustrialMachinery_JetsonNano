echo " "
echo "Building......"
echo " "

rm -rf ./results
rm -rf ./logs

cd build
make
cd ..

echo " "
echo "Analyzing......"
echo " "

#Motion Magnification
#Video Sample 1
./build/evm-cpp-linux ./vids/machine1.mp4 1 100 90 4.0 6.2 30 1
#Video Sample 2
#./build/evm-cpp-linux ./vids/machine2.mp4 1 60 90 3.6 6.2 30 0.3
#Video Sample 3
#./build/evm-cpp-linux ./vids/machine3.mp4 1 70 90 3.6 6.2 30 1
#Video Sample 4
#./build/evm-cpp-linux ./vids/machine4.mp4 1 25 90 0.5 3.5 30 0

#python3 video_detection.py