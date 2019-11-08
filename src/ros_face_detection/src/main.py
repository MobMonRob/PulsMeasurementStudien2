#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('ros_face_detection')
import sys
import rospy
import cv2
import os
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("/face_detection/image_raw",Image,queue_size = 10)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/webcam/image_raw", Image,self.callback)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(os.path.dirname(os.path.realpath(__file__)) + "/../resources/cascade.xml")

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped_image = cv_image[y: y + h / 3, x: x + w]
        # cv2.imshow("Cropped", cropped_image)
        # cv2.waitKey(3)

    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)

def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
