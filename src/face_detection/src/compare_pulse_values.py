#!/usr/bin/env python
# -*- encoding: utf-8 -*-

__version__ = "0.1.1"
from face_detection.msg import Pulse
import sys
import rospy

class ComparePulseValues:

    def __init__(self, topic, topicToCompare):
        self.topic = topic
        self.topicToCompare = topicToCompare
        self.pulse = None
        self.pulseToCompare = None
        self.error = None

    def run(self):
        rospy.Subscriber(self.topic, Pulse, self.pulse_callback)
        rospy.Subscriber(self.topicToCompare, Pulse, self.pulse_to_compare_callback)
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down")

    def pulse_callback(self, pulse):
        self.calculate_error(topic=True, pulse=pulse)

    def pulse_to_compare_callback(self, pulse):
        self.calculate_error(topic=False, pulse=pulse)

    def calculate_error(self, topic, pulse):
        if topic is True:
            self.pulse = pulse
        else:
            self.pulseToCompare = pulse

        absolute_error = abs(self.pulseToCompare-self.pulse)
        self.error = absolute_error/self.pulse


def main():
    rospy.init_node('compare', anonymus=False, log_level=rospy.DEBUG)
    topic = rospy.get_param("~topic", "/pulsgurt")
    rospy.loginfo("Listening on topic '" + topic + "'")
    topicToCompare = rospy.get_param("~topicToCompare", "/head_movement_pulse")
    rospy.loginfo("Listening on topic '" + topicToCompare + "'")
    pulse = ComparePulseValues(topic, topicToCompare)
    pulse.run()


if __name__ == "__main__":
    sys.argv = rospy.myargv()
    main()