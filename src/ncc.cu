#include <opencv2/gpu/device/vec_math.hpp>
#include <opencv2/gpu/device/functional.hpp>
#include "ncc.cuh"   



        //transfrom the image matrix
		__global__ void gpu_resize(int width, int height, int w, int h, float scale, const PtrStepb img, PtrStepb result)
        {
            int s_i, s_j;
            float d_i, d_j;
			for (int i = 0; i < h; ++i)
                for (int j = 0; j < w; ++j){  
                 	s_i = int(i/scale);
                 	s_j = int(j/scale);
                 	
                 	d_i= i/scale - s_i;
                 	d_j= j/scale - s_j;
                 	
                    result.ptr(i)[j]= int( img.ptr(s_i)[s_j]*(1-d_i)*(1-d_j) + img.ptr(s_i+1)[s_j]*d_i*(1-d_j) + img.ptr(s_i)[s_j+1]*(1-d_i)*d_j + img.ptr(s_i+1)[s_j+1]*d_i*d_j );                                  	    
                 }       
        }
        
        
        __global__ void gpu_resize2(int width, int height, int w, int h, float scale, const PtrStepb img, PtrStepb result)
        {
            int s_i, s_j;
			for (int i = 0; i < h; ++i)
                for (int j = 0; j < w; ++j){  
                 	s_i = int(i/scale);
                 	s_j = int(j/scale);
                                 	
                    result.ptr(i)[j]= int( img.ptr(s_i)[s_j]);                                  	    
                 }       
        }
        
        
        
        //transfrom the image matrix
		__global__ void transform_img(int width, int height, PtrStepSzf img_vec, PtrStepSzf img_mean, PtrStepSzf img_sqsum, PtrStepSzf img_dst)
        {
            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;
            float stddev;
            //int total = width*height;
			if(x< width && y< height){
                  stddev = sqrtf( img_sqsum.ptr(0)[x] );
                  img_dst.ptr(y)[x] = (img_vec.ptr(y)[x] - img_mean.ptr(0)[x]) / stddev;                 	    
            }        
        }
		
		
		
		//compute (x-avg)^2 of each pixel in the image
		__global__ void diff_img(int width, int height, const PtrStepSzf img_vec, PtrStepSzf img_mean,  PtrStepSzf img_sq)
        {                   
            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;
			float diff;
			if(x< width && y< height){
                 diff= img_vec.ptr(y)[x] - img_mean.ptr(0)[x];
                 img_sq.ptr(y)[x] = diff * diff;      	    
			}
        }			
		
		
		//transform the colvolved image to huge matrix in one step.
		__global__ void mean_stdev(int w, int h, int bound_cols, int bound_rows, const PtrStepSzf image, PtrStepSzf img_mean, PtrStepSzf img_stdev, PtrStepSzf result)
        {

            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;
			int total = w*h;
			
			float sum=0;
			float pixel;
			if(x<bound_cols && y<bound_rows){
                for (int i = 0; i < h; ++i)
                    for (int j = 0; j < w; ++j){  
                    	pixel= image.ptr(y + i)[x + j] ;                     
                        //result.ptr(i*w+j)[y*bound_cols+x] = pixel; 
                        sum = sum + pixel;                  	    
                    }
                img_mean.ptr(y)[x] = sum/total;    
			}
			
			__syncthreads();
			
			
			float diff;
			float sqsum=0;
			float mean;
			if(x<bound_cols && y<bound_rows){
				mean = img_mean.ptr(y)[x];
                for (int i = 0; i < h; ++i)
                    for (int j = 0; j < w; ++j){  
                    	diff = image.ptr(y + i)[x + j] -mean;                      
                        sqsum = sqsum + diff * diff;                  	    
                    }
                //img_stdev.ptr(y)[x] = sqrtf(sqsum/total); 
                img_stdev.ptr(y)[x] = sqrtf(sqsum);   
			}
						
			__syncthreads();
			
			
			float stdev;
			if(x<bound_cols && y<bound_rows){
				mean = img_mean.ptr(y)[x];
				stdev = img_stdev.ptr(y)[x];
                for (int i = 0; i < h; ++i)
                    for (int j = 0; j < w; ++j){  
                    	pixel= image.ptr(y + i)[x + j] ;                     
                        result.ptr(i*w+j)[y*bound_cols+x] = (pixel - mean)/stdev;                	    
                    }    
			}
        }
		
		//mul_matrix<<<gridm, threadm, 0>>>( dst.cols, dst.rows, temp_dst.cols, img_dst, temp_dst, dst);
		// multiplay the image matrix and template matrix
		__global__ void mul_matrix(int w, int h, int num, const PtrStepf image, const PtrStepf temp, PtrStepSzf result)
		{
			
			int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;
			
			float sum=0;
			//float pixel;
			if(x<w && y<h){
                for (int i = 0; i < num; ++i){  
                    sum = sum + image.ptr(num)[x] * temp.ptr(y)[num];                                       	    
                }
                result.ptr(y)[x] = sum;    
			}
		
		}
		
		//transfrom each image patch into vector
        __global__ void transpose_img(int w, int h, int bound_cols, int bound_rows, const PtrStepf image, PtrStepSzf result, PtrStepSzf img_mean)
        {

            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;
			
			float sum=0;
			float pixel;
			if(x<bound_cols && y<bound_rows){
                for (int i = 0; i < h; ++i)
                    for (int j = 0; j < w; ++j){  
                    	pixel= image.ptr(y + i)[x + j] ;                     
                        result.ptr(i*w+j)[y*bound_cols+x] = pixel; 
                        sum = sum + pixel;                  	    
                    }
                img_mean.ptr(0)[y*bound_cols+x] = sum/(w*h);    
			}
        }       
        
        
        //convolve image with the LOG kernel without transforming the matrix
        __global__ void convolution(int log_w, int log_h, int bound_cols, int bound_rows, const PtrStepb image, PtrStepSzf log_k, PtrStepSzf result)
        {

            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;
			
			float sum=0;
			if(x<bound_cols && y<bound_rows){
                for (int i = 0; i < log_h; ++i)
                    for (int j = 0; j < log_w; ++j){  
                    	sum = sum + log_k.ptr(i)[j] * float( image.ptr(y + i)[x + j] );                     	                    }
                result.ptr(y)[x] = sum;    
			}
        }  
        
        
        //convolve template with the LOG kernel
        __global__ void convolve_temp(int log_w, int log_h, int bound_cols, int bound_rows, const PtrStepb image, PtrStepSzf log_k, PtrStepSzf templ, int n)
        {

            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;
			
			float sum=0;
			if(x<bound_cols && y<bound_rows){
                for (int i = 0; i < log_h; ++i)
                    for (int j = 0; j < log_w; ++j){  
                    	sum = sum + log_k.ptr(i)[j] * float( image.ptr(y + i)[x + j] );                     	                    }
                templ.ptr(n)[y*bound_cols+x] = sum;    
			}
        }         		
		
		
		
		//compute (x-avg)^2 of each pixel in the template
		__global__ void diff_temp(int width, int height, const PtrStepSzf temp_vec, PtrStepSzf temp_mean,  PtrStepSzf temp_sq)
        {                   
            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;
            
			float diff;
			if(x< width && y< height){
                 diff= temp_vec.ptr(y)[x] - temp_mean.ptr(0)[y];
                 temp_sq.ptr(y)[x] = diff * diff;      	    
			}
        }        
        
        
        
        //transfrom the template matrix
		__global__ void transform_temp(int width, int height, const PtrStepSzf temp_vec, PtrStepSzf temp_mean, PtrStepSzf temp_sqsum, PtrStepSzf temp_dst)
        {
            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;
            
            float stddev;
			if(x< width && y< height){
                  //stddev = sqrtf( temp_sqsum.ptr(0)[y] / width);
                  stddev = sqrtf( temp_sqsum.ptr(0)[y] );
                  temp_dst.ptr(y)[x] = (temp_vec.ptr(y)[x] - temp_mean.ptr(0)[y]) / stddev;                 	    
            }        
        } 
        
        
        
        //transfrom the template matrix
        __global__ void column_norm(int width, int height, const PtrStepSzf dst, PtrStepSzf col_norm)
        {
        
            int x = (blockIdx.x *16 + blockIdx.y)* 128 + threadIdx.x * 16 + threadIdx.y;
            
            float sum=0;
			if(x< width ){
				for(int y=0; y< height; y++){               
                  sum = sum + dst.ptr(y)[x] * dst.ptr(y)[x]; 
                  }
                  col_norm.ptr(0)[x]= sqrtf(sum);                               	    
            }        
        } 
        
        
        
        //copy the new ZY matrix
        //reduce_matrix<<<grid6, thread5, 0>>>(count, d_ZY.rows, d_ZY, new_ZY, column_list);
        __global__ void reduce_matrix(int width, int height, const PtrStepSzf dst, PtrStepSzf new_ZY, int* a)
        {
        
            int x = (blockIdx.x *16 + blockIdx.y)* 128 + threadIdx.x * 16 + threadIdx.y;
            
			if(x< width ){
				int num= a[x]; 
				for(int y=0; y< height; y++){ 				            
                  new_ZY.ptr(y)[x] = dst.ptr(y)[num];                  
                  }                              	    
            }        
        } 
        
        
        // calculate the column row.
        __global__ void calculate_list(int width, int height, const PtrStepSzf dst, int* column_list, int* count)
        {
			int i = -1;        
            for(int j=0; j< width; j++)
				if( dst.ptr(0)[j] >0.5 ){	
					i ++;
					column_list[i]=j;
				}                           	    
        	*count = i;
        }
        
        
        //reduce_mat<<<0, thread6, 0>>>(count, d_ZY.rows, d_ZY, new_ZY, d_list);
        __global__ void reduce_mat(int width, int height, const PtrStepSzf dst, PtrStepSzf new_ZY, int* a)
        {
        
            int x = threadIdx.x * 16 + threadIdx.y;
            
			if(x< width ){
				int num= a[x]; 
				for(int y=0; y< height; y++){ 				            
                  new_ZY.ptr(y)[x] = dst.ptr(y)[num];                  
                  }                              	    
            }        
        } 
                     
