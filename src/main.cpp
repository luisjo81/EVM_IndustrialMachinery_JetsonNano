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
  
    int status;

    string name;
    string delimiter = "/";
    size_t last = 0; size_t next = 0;
    string resultsDir = "./results/";
    string logDir = "./logs/";
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
    
    if (!utils::fs::createDirectory(resultsDir))
        cout << "Not able to create the results directory or directory already created";
        cout << endl;

    if (!utils::fs::createDirectory(logDir))
        cout << "Not able to create the log directory or directory already created";
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


    switch (method)
    {   
    case 1:
        fl = stof(fl_str);
        fh = stof(fh_str);
        status = amplify_spatial_lpyr_temporal_butter(dataDir, name, resultsDir,
            alpha, lambda, fl, fh, samplingRate, chromAttenuation);
        break;
    
    default:
        cerr << "Incorrect selected method, options are:" << endl;
        cerr << "1 - Motion magnification Butterworth (only one available for this application)" << endl;
        return -1;
    }

    if (status < 0) {
        cerr << "Processing was done with some errors" << endl;
        return -1;
    }
  
    return 0;
}