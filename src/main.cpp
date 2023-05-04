#include <iostream>

#include <omp.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/core/cuda.hpp>

#include "processing_functions.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {

    if (argc != 9)
    {
       cerr << "usage: evm-cpp-linux <input_file> <method> <alpha> <level/lambda_c>";
       cerr << " <fl/r1> <fh/r2> <sampling_rate> <chrom_attenuation>" << endl;
       return -1;
    }
    
    // Get OpenCV build information for feature support
    // cout << getBuildInformation() << endl; 

    int status;
    // string dataDir = "./vid/";
    string name;
    string delimiter = "/";
    size_t last = 0; size_t next = 0;
    string resultsDir = "./Results/";
    string dataDir = argv[1];

    int method = stoi(argv[2]);
    float alpha = stof(argv[3]);
    int level = stoi(argv[4]);
    float lambda = stof(argv[4]);
    string fl_str = argv[5];
    string fh_str = argv[6];
    int samplingRate = stoi(argv[7]);
    float chromAttenuation = stof(argv[8]);
    float fl, fh;
    float enu, den;
    
    // utils::fs::remove_all(resultsDir);
    if (!utils::fs::createDirectory(resultsDir))
        cout << "Not able to create the directory or directory already created";
        cout << endl;

    if (!utils::fs::exists(dataDir)) {
        cerr << "The input file does not exist" << endl;
        return -1;
    }

    while ((next = dataDir.find(delimiter, last)) != string::npos) {
        last = next + 1;
    }

    name = dataDir.substr(last);
    name = name.substr(0, name.find("."));

    cout << "Cuda #N " << cuda::getCudaEnabledDeviceCount() << endl;
    cout << "OpenMP Max Threads: " <<  omp_get_max_threads() << endl;

    switch (method)
    {
    case 0:
        enu = stof(fl_str.substr(0, fl_str.find("/")));
        den = stof(fl_str.substr(fl_str.find("/")+1));
        fl = enu / den;
        enu = stof(fh_str.substr(0, fh_str.find("/")));
        den = stof(fh_str.substr(fh_str.find("/")+1));
        fh = enu / den;
        status = amplify_spatial_Gdown_temporal_ideal(dataDir, name, resultsDir, 
            alpha, level, fl, fh, samplingRate, chromAttenuation);
        break;
    
    case 1:
        fl = stof(fl_str);
        fh = stof(fh_str);
        status = amplify_spatial_lpyr_temporal_butter(dataDir, name, resultsDir,
            alpha, lambda, fl, fh, samplingRate, chromAttenuation);
        break;

    case 2:
        fl = stof(fl_str);
        fh = stof(fh_str);
        status = amplify_spatial_lpyr_temporal_ideal(dataDir, name, resultsDir,
            alpha, lambda, fl, fh, samplingRate, chromAttenuation);
        break;

    case 3:
        fl = stof(fl_str);
        fh = stof(fh_str);
        status = amplify_spatial_lpyr_temporal_iir(dataDir, name, resultsDir,
            alpha, lambda, fl, fh, chromAttenuation);
        break;
    
    default:
        cerr << "Incorrect selected method, options are:" << endl;
        cerr << "0 - Color magnification" << endl;
        cerr << "1 - Motion magnification Butterworth" << endl;
        cerr << "2 - Motion magnification Ideal" << endl;
        cerr << "3 - Motion magnification IIR (Does't work)" << endl;
        return -1;
    }

    if (status < 0) {
        cerr << "Processing was done with some errors" << endl;
        return -1;
    }
  
    return 0;
}