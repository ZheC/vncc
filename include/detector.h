#ifndef DETECTOR_H
#define DETECTOR_H

#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <stdio.h>
#include <sstream>
#include <time.h>
#include <fstream>
#include <string>
#include "ncc.cuh"

typedef struct product {
  int rows;
  int cols;
  int id;
  float scale;
  Mat children_position;
} Output;

using namespace cv;
using namespace std;
		
void writeMatToFile(const Mat &image, char* file_name);
		
void read_yml(GpuMat& result, string yml_name, string var_name);
    	
void show_gpu_result(const GpuMat& image, int index, int height, string yml_name, string image_name);
                
void write_gpu_result(const GpuMat& image, int index, int height, string txt_name, string image_name);
    	
//vectorized normalized cross-correlation
void compute_response_map(const GpuMat& d_image, const GpuMat& temp_dst, const GpuMat& d_log, int temp_w, int temp_h, GpuMat& level_score);
        
//visulize the alignment result
void visualize_alignment(Mat &depth, float position[][3], int angle[][3], map<float, Output, greater<float> > max_map, vector<int>& temp_num, int m);

// convert 2d position back to 3D position
void convert2Dto3D( Mat depth_image, Mat intrinsics, map<float, Output, greater<float> > max_map, string object_name, cv::Vec3f& position, cv::Vec3f& angle);
		
// the controller of the program
bool DetectObject(Mat color_image, Mat depth_image, Mat intrinsics, string training_file_path, string object_name, cv::Vec3f& position, cv::Vec3f& angle);

#endif    	   	
