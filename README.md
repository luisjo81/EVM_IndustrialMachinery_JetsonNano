
# Eulerian Video Magnification algorithm applied to industrial machinery monitoring

## :clipboard: Installation
The first thing we need to do is to update the system, to do so, open a new terminal and type the following commands:
```
sudo apt update
```
```
sudo apt install build-essential
```
```
sudo apt-get install cmake
```
```
sudo apt-get install gcc g++
```
After this, we will proceed to install all the other needed dependencies.

- Python
```
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
```
```
mkdir /python && cd /python
```
```
wget https://www.python.org/ftp/python/3.11.0/Python-3.11.0a7.tgz
```
```
tar -xvf Python-3.11.0a7.tgz
```
```
cd Python-3.11.0a7
./configure --enable-optimizations
```
```
sudo make install
```
- OpenCV

- Cloning Repository
```
git clone https://github.com/luisjo81/EVM_IndustrialMachinery_JetsonNano
```

## :computer: Run the program
Open a terminal in the cloned repository and the type the following command
```
./EVM_run.sh
```
This will apply the eulerian video magnification to a video in the vids directory. Once finished, there will be a processed video in the results directory and a logfile in the logs directory. 
Then, to see the movement detection and tracking, type these commands in the terminal:
```
cd src && python3 video_detection.py
```

## ➕: More
- [Eulerian Video Magnification (EVM)](http://people.csail.mit.edu/mrub/evm/)
- [Optimized Eulerian Video Magnification](https://github.com/kisung5/evm_c_plusplus.git)
- [Android Video Magnification](https://github.com/edmobe/android-video-magnification.git)
