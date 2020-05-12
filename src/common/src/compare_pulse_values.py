#!/usr/bin/env python
# -*- encoding: utf-8 -*-

__version__ = "0.1.1"
from common.msg import Pulse
from common.msg import Error
import sys
import rospy

class ComparePulseValues:

    def __init__(self, topic, topic_to_compare):
        # set up ROS publisher
        self.pub = rospy.Publisher('compare_pulse_values', Error, queue_size=10)
        # sequence of published error values, published with each error message
        self.published_error_value_sequence = 0
        self.topic = topic
        self.topic_to_compare = topic_to_compare
        self.pulse = None
        self.pulseToCompare = None
        self.error = None
        self.timestamp = None

    def run(self):
        rospy.Subscriber(self.topic, Pulse, self.pulse_callback)
        rospy.Subscriber(self.topic_to_compare, Pulse, self.pulse_to_compare_callback)
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
            self.pulse = pulse.pulse
        else:
            self.pulseToCompare = pulse.pulse

        if self.pulseToCompare is not None and self.pulse is not None:
            self.timestamp = pulse.time
            absolute_error = abs(self.pulseToCompare-self.pulse)
            self.error = (absolute_error/self.pulse)*100
            self.publish_error()

    def publish_error(self):
        rospy.loginfo("[ComparePulseValues] Error: "+str(self.error))
        msg_to_publish = Error()
        msg_to_publish.error = self.error
        msg_to_publish.time.stamp = rospy.Time.now()
        msg_to_publish.time.seq = self.published_error_value_sequence
        self.pub.publish(msg_to_publish)
        self.published_error_value_sequence += 1

def main():
    rospy.init_node("compare", anonymous=False, log_level=rospy.DEBUG)

    topic = rospy.get_param("~topic", "/pulsgurt")
    rospy.loginfo("[ComparePulseValues] Listening on topic '" + topic + "'")

    topic_to_compare = rospy.get_param("~topicToCompare", "/head_movement_pulse")
    rospy.loginfo("[ComparePulseValues] Listening on topic '" + topic_to_compare + "'")

    pulse = ComparePulseValues(topic, topic_to_compare)
    pulse.run()


if __name__ == "__main__":
    sys.argv = rospy.myargv()
    main()
