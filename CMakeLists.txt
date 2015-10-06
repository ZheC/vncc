cmake_minimum_required(VERSION 2.8.3)
project(vncc)


find_package( CUDA )
find_package( OpenCV REQUIRED CONFIG PATHS ~/local NO_DEFAULT_PATH)

#find_package( OpenCV REQUIRED)

set(GENCODE -gencode=arch=compute_30,code=sm_30)
set(GENCODE ${GENCODE} -gencode=arch=compute_35,code=compute_35)
set(GENCODE ${GENCODE} -gencode=arch=compute_20,code=sm_20)
set(CUDA_NVCC_FLAGS_DEBUG "-G;-g")
set(CUDA_NVCC_FLAGS_RELEASE "-O2")



include_directories(${OpenCV_INCLUDE_DIRS})
message("Shushman: ${OpenCV_INCLUDE_DIRS}")
include_directories( /usr/local/cuda/include )


find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  std_msgs
  sensor_msgs
  image_transport
  roscpp
  cmake_modules
  message_generation
  geometry_msgs
)



add_message_files(
    DIRECTORY
    msg
    FILES
    Detection.msg
)

add_service_files(
    DIRECTORY
    srv
    FILES
    GetDetections.srv
)

catkin_python_setup()

generate_messages(
  DEPENDENCIES
  ${PROJECT_NAME}
	std_msgs
  geometry_msgs
)


catkin_package(
	INCLUDE_DIRS include
  LIBRARIES vncc_cuda
  CATKIN_DEPENDS message_runtime roscpp std_msgs geometry_msgs
  DEPENDS system_lib
)

include_directories(
    include/
    ${catkin_INCLUDE_DIRS}
)

cuda_add_library(vncc_cuda SHARED src/detector.cu src/ncc.cu)

target_link_libraries(
    vncc_cuda
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)


add_executable(vncc_estimator src/ros/vncc_ros.cc)
target_link_libraries(vncc_estimator ${catkin_LIBRARIES})
target_link_libraries(vncc_estimator ${OpenCV_LIBS})
target_link_libraries(vncc_estimator ${CUDA_LIBRARIES} vncc_cuda)
add_dependencies( vncc_estimator ${PROJECT_NAME}_generate_messages_cpp )
add_dependencies( vncc_estimator ${PROJECT_NAME}_generate_messages_py )


install(DIRECTORY include/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)


install(TARGETS vncc_estimator
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
