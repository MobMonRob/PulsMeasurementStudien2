#!/usr/bin/env python
# -*- encoding: utf-8 -*-

__version__ = "0.1.1"

import os
import sys
import time
import logging
import pexpect
import argparse
import rospy
from sensor_msgs.msg import Image


class PulseHeadMovement:

    def __init__(self, topic):
        self.topic = topic

    def run(self):
        rospy.Subscriber(self.topic, Image, self.pulse_callback)
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down")

    def pulse_callback(self, forehead):
        rospy.loginfo("Hello")


def main():
    rospy.init_node('head_movement_listener', anonymous=False, log_level=rospy.DEBUG)
    topic = rospy.get_param("~topic", "/face_detection/forehead")
    rospy.loginfo("Listening on topic '" + topic + "'")
    pulse = PulseHeadMovement(topic)
    pulse.run()


if __name__ == "__main__":
    sys.argv = rospy.myargv()
    main()
