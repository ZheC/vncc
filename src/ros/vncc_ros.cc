#include "ncc.cuh"
#include "detector.h"

#include <ros/ros.h>
#include <ros/forwards.h>
#include <ros/single_subscriber_publisher.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <visualization_msgs/Marker.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <cv_bridge/cv_bridge.h>
#include "geometry_msgs/Pose.h"
#include <sstream>
#include <fstream>


#include <vncc/GetDetections.h>

using namespace std;

//Global variables needed across function calls
cv::Mat currentDepthImage;
cv::Mat currentColorImage;
cv::Mat goodDepthImage;
sensor_msgs::CameraInfo currentCameraInfo;
bool has_color_image_;
bool has_depth_image_;
bool has_camera_info_ = false;

ros::Publisher pose_publisher;
std::string training_file_path;

// Callback for camera info
void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& camera_info)
{
	currentCameraInfo = (*camera_info);
	has_camera_info_ = true;
}

//Callback for depth image
void depthImageCallback(const sensor_msgs::ImageConstPtr& msg)
{
	cv_bridge::CvImagePtr subscribed_ptr;
    try
	{
		subscribed_ptr = cv_bridge::toCvCopy(msg,"32FC1");
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
	}
	currentDepthImage = subscribed_ptr->image;
    
    has_depth_image_ = true;


//Callback for color image
void colorImageCallback(const sensor_msgs::ImageConstPtr& msg)
{
	cv_bridge::CvImagePtr subscribed_ptr;
	try
	{
		subscribed_ptr = cv_bridge::toCvCopy(msg, "bgr8");
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
	}

	currentColorImage = subscribed_ptr->image;
	has_color_image_ = true;
}

//
bool GetDetectionsService(vncc::GetDetections::Request &req,
						  vncc::GetDetections::Response &resp)
{
	std::string object_name = req.object_name;
	ROS_INFO_STREAM("Running VNCC Detections Service for "<<object_name);

	cv::Matx33f intrinsics_mat(currentCameraInfo.K[0], 0, currentCameraInfo.K[2],
                           0, currentCameraInfo.K[4], currentCameraInfo.K[5],
                           0, 0, 1);

    cv::Mat intrinsics(intrinsics_mat);    
    cout<<intrinsics<<endl;
	//Detect plate
	Vec3f position;
	Vec3f angle;

	//Check if object present
	bool has_object = DetectObject(currentColorImage,currentDepthImage,intrinsics,
								  training_file_path,object_name,position,angle);

	if(has_object)
	{
		//Get pose and append to response
		vncc::Detection detection;
		detection.header.stamp = ros::Time::now();
		detection.header.frame_id = object_name;

		//Get pose of object
		detection.pt.x = position[0];
		detection.pt.y = position[1];
		detection.pt.z = position[2];

		detection.roll = angle[0];
		detection.pitch = angle[1];
		detection.yaw = angle[2];

		resp.detections.push_back(detection);
		resp.ok = true;

		ROS_INFO_STREAM("Found "<<object_name<<" !");
	}

	else
	{
		ROS_INFO_STREAM("did not find "<<object_name<<" !");
		resp.ok==false;
	}

	return true;

}


int main(int argc, char **argv)
{
	ros::init(argc, argv, "vncc");
	ros::NodeHandle nh("~");

	training_file_path = std::string(argv[1]);

	image_transport::ImageTransport depth_it(nh);
	// rename the image topic before using
	image_transport::Subscriber depth_sub = 
	depth_it.subscribe("depth_image", 1, depthImageCallback);  

	image_transport::ImageTransport color_it(nh);
	// rename the image topic before using
	image_transport::Subscriber color_sub = 
	color_it.subscribe("color_image", 1, colorImageCallback);

	// camera_info topic, rename this too
	ros::Subscriber info_subscriber = nh.subscribe("camera_info", 10, 
	&cameraInfoCallback);


	//Setup services
	ros::ServiceServer detection_server = nh.advertiseService(std::string("get_vncc_detections"), &GetDetectionsService);



	ros::spin();
	return 0;
}

