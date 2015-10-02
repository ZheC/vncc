from vncc_wrapper import detector

#Get detector
det = detector.VnccDetector()

#Query pose of object
bowl_pose = det.GetDetection('bowl_pose')
plate_pose = det.GetDetection('plate_pose')