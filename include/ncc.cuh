#ifndef NCC_CUH
#define NCC_CUH
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/stream_accessor.hpp>
#include <opencv2/core/cuda_devptrs.hpp>
#include <cuda_runtime.h>
#include "stdio.h"
#include "cuda.h"

using namespace cv;
using namespace cv::gpu;     



        //transfrom the image matrix
		__global__ void gpu_resize(int width, int height, int w, int h, float scale, const PtrStepb img, PtrStepb result);
        
        
        __global__ void gpu_resize2(int width, int height, int w, int h, float scale, const PtrStepb img, PtrStepb result);
        
        
        
        //transfrom the image matrix
		__global__ void transform_img(int width, int height, PtrStepSzf img_vec, PtrStepSzf img_mean, PtrStepSzf img_sqsum, PtrStepSzf img_dst);
		
		
		
		//compute (x-avg)^2 of each pixel in the image
		__global__ void diff_img(int width, int height, const PtrStepSzf img_vec, PtrStepSzf img_mean,  PtrStepSzf img_sq);			
		
		
		//transform the colvolved image to huge matrix in one step.
		__global__ void mean_stdev(int w, int h, int bound_cols, int bound_rows, const PtrStepSzf image, PtrStepSzf img_mean, PtrStepSzf img_stdev, PtrStepSzf result);
		
		//transfrom each image patch into vector
        __global__ void transpose_img(int w, int h, int bound_cols, int bound_rows, const PtrStepf image, PtrStepSzf result, PtrStepSzf img_mean);      
        
        
        //convolve image with the LOG kernel without transforming the matrix
        __global__ void convolution(int log_w, int log_h, int bound_cols, int bound_rows, const PtrStepb image, PtrStepSzf log_k, PtrStepSzf result);
        
        //convolve template with the LOG kernel
        __global__ void convolve_temp(int log_w, int log_h, int bound_cols, int bound_rows, const PtrStepb image, PtrStepSzf log_k, PtrStepSzf templ, int n);      		
		
		
		
		//compute (x-avg)^2 of each pixel in the template
		__global__ void diff_temp(int width, int height, const PtrStepSzf temp_vec, PtrStepSzf temp_mean,  PtrStepSzf temp_sq);       
        
        
        
        //transfrom the template matrix
		__global__ void transform_temp(int width, int height, const PtrStepSzf temp_vec, PtrStepSzf temp_mean, PtrStepSzf temp_sqsum, PtrStepSzf temp_dst);
        
        
        
        //transfrom the template matrix
        __global__ void column_norm(int width, int height, const PtrStepSzf dst, PtrStepSzf col_norm);
        
        
        
        //copy the new ZY matrix
        //reduce_matrix<<<grid6, thread5, 0>>>(count, d_ZY.rows, d_ZY, new_ZY, column_list);
        __global__ void reduce_matrix(int width, int height, const PtrStepSzf dst, PtrStepSzf new_ZY, int* a);
        
        
        // calculate the column row.
        __global__ void calculate_list(int width, int height, const PtrStepSzf dst, int* column_list, int* count);
        
        
        //reduce_mat<<<0, thread6, 0>>>(count, d_ZY.rows, d_ZY, new_ZY, d_list);
        __global__ void reduce_mat(int width, int height, const PtrStepSzf dst, PtrStepSzf new_ZY, int* a);

#endif                     
