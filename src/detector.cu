#include <opencv2/gpu/device/vec_math.hpp>
#include <opencv2/gpu/device/functional.hpp>

#include "detector.h"


		
		void writeMatToFile(const Mat &image, char* file_name) {
		 
			ofstream fout(file_name);
			float *ptr;
			ptr=(float *)image.data;

			 for(int i=0; i< image.rows; i++){
				for(int j=0; j<image.cols; j++){
					fout<<*ptr<<"\t";
					ptr++;
				}
				fout<<endl;
			}
			fout.close();
		}
		
		
		void read_yml(GpuMat& result, string yml_name, string var_name)
    	{
    	    Mat temp;
  	
			FileStorage fin( yml_name, FileStorage::READ);
			fin[var_name] >> temp;
			temp.convertTo(temp, CV_32FC1);
			result.upload(temp);    	    
    	}
    	
		

        void show_gpu_result(const GpuMat& image, int index, int height, string yml_name, string image_name)
    	{
    	    gpu::GpuMat d_score(Size(image.cols,1), CV_32FC1, (void*)(image.ptr<float>(index)), image.step);		  	  	  	
		  	Mat score(d_score);
		  	
		  	if(yml_name!=""){		  	
		  		FileStorage fb(yml_name, FileStorage::WRITE);
				fb << "score" << score;
				fb.release();
			  	}
			
			if(image_name!=""){
			    score=score.reshape(0, height);
		  		normalize( score, score, 0, 255, NORM_MINMAX, -1);
		  		imwrite(image_name,score);
		  		}
    	}
        
        
        
        void write_gpu_result(const GpuMat& image, int index, int height, string txt_name, string image_name)
    	{
    	    gpu::GpuMat d_score(Size(image.cols,1), CV_32FC1, (void*)(image.ptr<float>(index)), image.step);		  	  	  	
		  	Mat score(d_score);
		  	score=score.reshape(0, height);
		  	
		  	if(txt_name!=""){		  	
		  		
		  		char file_name[txt_name.size()+1];
		  		strcpy(file_name, txt_name.c_str());
		  		ofstream fout(file_name);
		  		
				float *ptr;
				ptr=(float *)score.data;

				for(int i=0; i< score.rows; i++){
					for(int j=0; j<score.cols; j++){
						fout<<*ptr<<"\t";
						ptr++;
					}
					fout<<endl;
				}
				fout.close();
			}
			
			if(image_name!=""){			    
		  		normalize( score, score, 0, 255, NORM_MINMAX, -1);
		  		imwrite(image_name,score);
		  		}
    	}
    	
        
        //vectorized normalized cross-correlation
        void compute_response_map(const GpuMat& d_image, const GpuMat& temp_dst, const GpuMat& d_log, int temp_w, int temp_h, GpuMat& level_score)
    	{	
			dim3 threads(16, 8);
        	
        	gpu::GpuMat d_img, log_img;
        	log_img.create(d_image.rows, d_image.cols, CV_32FC1);
		  	
		  	int img_w = d_image.cols - d_log.cols + 1;
		  	int img_h = d_image.rows - d_log.rows + 1;
		  	d_img.create(img_h, img_w, CV_32FC1);
		  	
		  	
		  	// prepare for transforming the image
		  	gpu::GpuMat img_dst;			
		  	int bound_rows= img_h - temp_h + 1;
  			int bound_cols= img_w - temp_w + 1;
		  	img_dst.create(temp_w * temp_h, bound_cols * bound_rows, CV_32FC1);
		  	
		  	gpu::GpuMat image_mean, image_stdev;			  	
		  	image_mean.create( bound_rows, bound_cols, CV_32FC1);
		  	image_stdev.create( bound_rows, bound_cols, CV_32FC1);

		  	gpu::GpuMat d_test;
		  	//d_test.create(temp_dst.rows, img_dst.cols, CV_32FC1);
		  	
		  	// convolve the image with the given LOG kernel
		  	dim3 grid3(divUp(d_image.cols, threads.x), divUp(d_image.rows, threads.y));	
		 	convolution<<<grid3, threads, 0>>>(d_log.cols, d_log.rows, img_w, img_h, d_image, d_log, d_img);	 			 	
		 			 		 	
		  	//transpose the image  	
		  	dim3 grid4(divUp(bound_cols, threads.x), divUp(bound_rows, threads.y));
		  	mean_stdev<<<grid4, threads, 0>>>(temp_w, temp_h, bound_cols, bound_rows, d_img, image_mean, image_stdev, img_dst);	 					
		  	// matrix multiplication  
		  
		  	gpu::gemm( temp_dst, img_dst, 1, d_test, 0, level_score, 0);
		  	
		  	/*cudaFree(log_img.ptr());
		  	cudaFree(d_img.ptr());
		  	cudaFree(image_mean.ptr());
		  	cudaFree(image_stdev.ptr());
		  	cudaFree(img_dst.ptr());*/
        }
        
 	
    	//visulize the alignment result
    	void visualize_alignment(Mat &depth, float position[][3], int angle[][3], map<float, Output, greater<float> > max_map, vector<int>& temp_num, int m) 
    	{
    		Mat templ, image;
    		char filename[50];
    		int ROW_NUM = 1080;  
    		int COL_NUM = 1920;
    		float temp_scale=0.5;
    		
    		Output output=max_map.begin()->second;
			
			//calculate the 2d object center		  	
		  	int rows= (output.rows + 45/output.scale)*2.5;
		  	int cols= (output.cols + 45/output.scale)*2.5;
			//cout<< "rows" << rows << " " << cols << endl;
	
			//convert 2d position back to 3d position 
			sprintf(filename, "1/frame%04d.txt", m);
			std::ifstream file(filename);

			for(int row=0; row<ROW_NUM; row++)  
				for (int col=0; col<COL_NUM; col++) 
					file >> depth.at<float>(row,col);
					//file >> depth[row][col];
			
			//resize( depth, depth, Size(depth.cols*0.4, depth.rows*0.4) );
	
			position[m][2]=depth.at<float>(rows, cols)/1000;
		  	position[m][0]=(cols-944.9)*(position[m][2]/1068.5);
		  	position[m][1]=(rows-549.5)*(position[m][2]/1275.9);
		  	
		  	cout << position[m][2] << " " << output.scale << endl;
		  	
		  	double min, max;
			minMaxIdx(depth, &min, &max);
		
			//calculate the angles
		  	int t=( output.id/6 ) % 3;
		  	int j=( output.id )%6; 
		  		
		  	angle[m][0]= (t-1)*15;
		  	angle[m][1]= 10*j+20;
		  	angle[m][2]= 0;
		  	//cout<< angle[m][0] << " " << angle[m][1] << " " << angle[m][2] << " " << endl;	
		  	
		  	temp_num.push_back(output.id+1);
		  	
		  	
		  	map < float, Output>::iterator iter= max_map.begin();
		  	  	
		  	for(int top=0; top<5; top++)
		  	{  	
		  		rows= iter->second.rows;
		  		cols= iter->second.cols;
		  		//printf("cols=%d,rows=%d\t", cols, rows);
		  		float img_scale = iter->second.scale;
		
				sprintf(filename, "model/template%03d.png",iter->second.id);
				templ = imread(filename,1);
		
				resize( templ, templ, Size(templ.cols*temp_scale/img_scale,templ.rows*temp_scale/img_scale) );
				sprintf(filename, "1/frame%04d.jpg", m);
				image = imread(filename,1);
				resize( image, image, Size(image.cols*0.4,image.rows*0.4) );
				//resize( image, image, Size(image.cols*0.8,image.rows*0.8) );
		
				for(int row = 0; row < templ.rows; row++)
			  		for (int col = 0; col < templ.cols; col++){
			   			if(templ.at<Vec3b>(row, col)[0]==0 && templ.at<Vec3b>(row, col)[1]==0 && templ.at<Vec3b>(row, col)[2]==0)
							continue;
						
				else{
					image.at<Vec3b>(row+rows, col+cols)[0]=templ.at<Vec3b>(row, col)[0];
					image.at<Vec3b>(row+rows, col+cols)[1]=templ.at<Vec3b>(row, col)[1];
					image.at<Vec3b>(row+rows, col+cols)[2]=templ.at<Vec3b>(row, col)[2];
				    //printf("%d",templ.at<Vec3b>(row, col)[0]);
				    }
				}
			
				//sprintf(filename, "frame%03d_top%d",m,top+1);
				//imshow(filename, image);
			
				sprintf(filename, "top5/frame%03d_top%d.png",m,top+1);
				imwrite(filename,image);
			
				//depth alignment
				sprintf(filename, "model/template%03d.png",iter->second.id);
				templ = imread(filename,0);
				templ.convertTo(templ, CV_8UC1);
				resize( templ, templ, Size(templ.cols*temp_scale/img_scale,templ.rows*temp_scale/img_scale) );
			
				Mat adjMap;
				depth.convertTo(adjMap, CV_8UC1, 255 / (max-min), -min);
				sprintf(filename, "1/depth%04d.png",m);
				imwrite(filename, adjMap);
				resize( adjMap, adjMap, Size(adjMap.cols*0.4,adjMap.rows*0.4) );
		
				for(int row = 0; row < templ.rows; row++)
			  		for (int col = 0; col < templ.cols; col++){
			   			if(templ.at<uchar>(row, col)==0)
							continue;
						
						else{
							adjMap.at<uchar>(row+rows, col+cols)=templ.at<uchar>(row, col);
				    		//printf("%d",templ.at<Vec3b>(row, col)[0]);
				    		}
				}
			
				//sprintf(filename, "frame%03d_top%d",m,top+1);
				//imshow(filename, image);
			
				sprintf(filename, "top5/depth%03d_top%d.png",m,top+1);
				imwrite(filename,adjMap);
			
			
				iter++;
			}
    	}
		
		
		// convert 2d position back to 3D position
		void convert2Dto3D( Mat depth_image, Mat intrinsics, map<float, Output, greater<float> > max_map, string object_name, cv::Vec3f& position, cv::Vec3f& angle){
		
			Output output=max_map.begin()->second;			
			//calculate the 2d object center		  	
		  	int rows= (output.rows + 45/output.scale)*2.5;
		  	int cols= (output.cols + 45/output.scale)*2.5;
		  	
		  	float p_3d[3];
		  	p_3d[2] =depth_image.at<float>(rows, cols)/1000;
            //cout<<"Depth reading at bowl is "<<p_3d[2]<<endl;
		  	p_3d[0]=(cols-intrinsics.at<float>(0,2))*(p_3d[2]/intrinsics.at<float>(0,0));
		  	//cout<<"Intrinsics val "<<intrinsics.at<float>(1,0)<<endl;
            p_3d[1]=(rows-intrinsics.at<float>(1,2))*(p_3d[2]/intrinsics.at<float>(1,1));
		  	
		  	position= Vec3f(p_3d[0], p_3d[1], p_3d[2]);
		  	//cout << position[m][2] << " " << output.scale << endl;
		
			//calculate the angles
		  	int t=( output.id/6 ) % 3;
		  	int j=( output.id )%6; 		  	
		  	angle = Vec3f( (t-1)*15, 10*j+20, 0);
		}
		
		
		
		// the controller of the program
		bool DetectObject(Mat color_image, Mat depth_image, Mat intrinsics, string training_file_path, string object_name, cv::Vec3f& position, cv::Vec3f& angle){
			Mat image, img, templ, img_display;  	
			gpu::GpuMat  d_image;
	
			double maxVal; Point maxLoc;	   	  	  	  	   
		  	vector<int> temp_num;
		  	dim3 threads(16, 8); 	  	
			 
			size_t time = clock(); 
				 
			gpu::GpuMat d_log;
			read_yml(d_log, training_file_path+"pre/log_19.yml", "c");
		
			gpu::GpuMat temp_level1;
			string filename=training_file_path+"pre/"+object_name+".yml";
			cout<<filename<<endl;
			read_yml(temp_level1, filename, "temp_level1"); 
			//read_yml(temp_level1, "pre/temp_level1.yml", "temp_level1"); 		
			int temp_w=72;
			int temp_h=72;
	
		  	vector<float> best_scale;      					
			cvtColor( color_image, image, CV_BGR2GRAY );		
			resize( image, image, Size(image.cols*0.4,image.rows*0.4) );	 	   		
		   	map<float, Output, greater<float> > max_map;
		   			   	
			// eight scales
		  for(int n=0; n<8; n++){ 
		
			//process the image
			float img_scale = 0.6 + 0.05*n;
			resize( image, img, Size(image.cols*img_scale,image.rows*img_scale) );
			img.copyTo( img_display ); 	 
		  	d_image.upload(img);  	
		  	
		  	int img_w = img.cols - d_log.cols - temp_w + 2;
		  	int img_h = img.rows - d_log.rows - temp_h + 2;
		  	
		  	//calculate the first level score
		  	gpu::GpuMat firstlevel_score;
		  	firstlevel_score.create(18, img_w*img_h,  CV_32FC1);
			compute_response_map(d_image, temp_level1, d_log, temp_w, temp_h, firstlevel_score);		  			  	

		  	 
		  	// multiple hypotheses 	 
			for(int v=0;v<9;v++){	// number of contours
				int interval=firstlevel_score.rows/9;
		  		int t=v*interval;	
		  		gpu::GpuMat vec_t(Size(firstlevel_score.cols,interval), CV_32FC1, (void*)(firstlevel_score.ptr<float>(t)), firstlevel_score.step);
		  		gpu::minMaxLoc( vec_t, NULL, &maxVal, NULL, &maxLoc);
		  		Output output;
		  	
		  		//cout<< "max" << maxLoc.x << endl;
		  		output.rows= (maxLoc.x/ img_w) /img_scale;
		  		output.cols= (maxLoc.x % img_w) /img_scale;
		  		output.id = maxLoc.y+t;
		  		output.scale = img_scale;
		  		//if(maxVal>0.98)
		  			max_map[maxVal] = output;
		  		}		  			  	
		  		
			} 
			
			//set a threshold for object detection
			if((max_map.begin()->first) > 0.55){
				
				//convert 2d position back to 3D position
				convert2Dto3D( depth_image, intrinsics, max_map, object_name, position, angle);		  		
		
				printf("Runtime: %f ms\n", (double(clock() - time)/CLOCKS_PER_SEC*1000.0));
	
				return 1;
			}
			else	
				return 0;
					
		}
	
