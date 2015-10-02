VNCC
====

This package is a ROS wrapper for the VNCC (Vectorized Normalized Cross Correlation) method for object pose estimation. The core algorithm was developed by Zhe Cao and the wrapper was written by Shushman Choudhury, both of Carnegie Mellon University.

###Dependencies
* Ubuntu 14.04
* ROS Indigo
* [catkin_tools](https://catkin-tools.readthedocs.org/en/latest/)
* OpenCV 2.4.0 or higher
* CUDA 6.0 or higher
* NVIDIA GPU (with 8GB memory or higher

###Folder structure
* `include` - The header files
* `launch` - Has an example launch file
* `msg` - For the message type for detection
* `pre` - Contains training files for the currently modelled objects
* `src` - Has the main algorithmic code, the ROS node and the python wrapper
* `srv` - The service call definition

The other files are standard required files for catkin packages. For any queries about the code in `include` or the .cu files in `src`, please contact Zhe Cao. For queries about the wrapper or running the code, please contact Shushman Choudhury.


###Running vncc

Please read the 'Note on OpenCV' below before building the package.

(It is assumed the user has basic knowledge of ROS. Also, please build the package with `catkin build` for convenience)

Once built, the `vncc` node can be launched by using a launch file. An example launch file is located in `vncc/launch/vncc_estimator.launch`. Please remap the parameters for camera topics AND the system-wide path to the `vncc/pre` folder where the training files are stored. The command for launching it is

`roslaunch vncc vncc_estimator.launch`

The terminal window where the launch file is run should remain open. Now you can query the method for detections via the python wrapper. An example of the same is in `vncc/src/vncc_wrapper/example.py`. This call can be made repeatedly while the launch file's terminal is executing.


### Note on OpenCV

Assuming you have installed the full ROS Indigo, it automatically installs a version of OpenCV that does NOT have GPU support. You need to install an additional OpenCV with GPU support in a local place, that you then link to only with vncc. Do not install this OpenCV with GPU support system wide, as that may affect your `iai_kinect2` package!

* Download OpenCV 2.4.10 from [SourceForge](http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.10/)
* Unzip the folder and enter it

```
mkdir build
cd build
```

For this next step, please observe the instructions [here](http://docs.opencv.org/doc/tutorials/introduction/linux_install/linux_install.html#building-opencv-from-source-using-cmake-using-the-command-line). Please add the `CMAKE_INSTALL_PREFIX` flag with appropriate options to the command below when you execute it.

```
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON  -D WITH_V4L=ON -D WITH_OPENGL=ON -D CUDA_ARCH_BIN=3.0 -D CUDA_ARCH_PTX=3.0 -D WITH_CUDA=ON -D WITH_VTK=ON -D WITH_CUBLAS=ON ..
make -j4
sudo checkinstall make install
```

Answer the questions of checkinstall (they prompt most answers), give your package a suitable name and wait for it to finish.

Hereafter, modify the line for `find_package (OpenCV REQUIRED CONFIG PATHS ...)` which is COMMENTED OUT currently in the CMakeLists.txt file. Then build the vncc package.

##A further note on CMakeLists

Depending on the location of cuda in your system, you may have to change some other lines in the CMakeLists.txt file as well. Please contact Shushman if there are problems building.