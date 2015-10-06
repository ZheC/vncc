import rospy
import vncc.srv
import vncc.msg
import math
import tf
import tf.transformations as transformations
import numpy
import os.path


def MsgToPose(msg):

	#Parse the ROS message to a 4x4 pose format

	#Get translation and rotation (from Euler angles)
	pose = transformations.euler_matrix(msg.roll,msg.pitch,msg.yaw) 

        pose[0,3] = msg.pt.x
        pose[1,3] = msg.pt.y
        pose[2,3] = msg.pt.z
    
	return pose


class VnccDetector(object):

	def __init__(self,service_namespace='/vncc',
				detection_frame=None,world_frame=None):

		#For getting transforms in world frame
		if detection_frame is not None and world_frame is not None:
			self.listener = tf.TransformListener()
		else:
			self.listener = None

		self.detection_frame = detection_frame
		self.world_frame = world_frame
                self.service_namespace = service_namespace

	def LocalToWorld(self,pose):
		#Get pose w.r.t world frame
		self.listener.waitForTransform(self.world_frame,self.detection_frame,
                                       rospy.Time(),rospy.Duration(10))
		t, r = self.listener.lookupTransform(self.world_frame,self.detection_frame,rospy.Time(0))

		#Get relative transform between frames
		offset_to_world = numpy.matrix(transformations.quaternion_matrix(r))
		offset_to_world[0,3] = t[0]
                offset_to_world[1,3] = t[1]
                offset_to_world[2,3] = t[2]

		#Compose with pose to get pose in world frame
		result = numpy.array(numpy.dot(offset_to_world, pose))

		return result


	def GetDetection(self,obj_name):

		#Call detection service for a particular object
		detect_vncc = rospy.ServiceProxy(self.service_namespace+'/get_vncc_detections',
										vncc.srv.GetDetections)

		detect_resp = detect_vncc(object_name=obj_name)

		if detect_resp.ok == False:
			return None
		#Assumes one instance of object
                result = MsgToPose(detect_resp.detections[0])
		#if (self.detection_frame is not None and self.world_frame is not None):
		    #	result = self.LocalToWorld(result)

		return result
