#!/usr/bin/env python
# -*- encoding: utf-8 -*-

__version__ = "0.1.1"

from common.msg import Pulse
from common.msg import Error
from datetime import datetime
import sys
import rospy
import csv
import os


class ComparePulseValues:

    def __init__(self, topic, topic_to_compare):
        # set up ROS publisher
        self.pub = rospy.Publisher('/compare_pulse_values', Error, queue_size=10)
        # sequence of published error values, published with each error message
        self.published_error_value_sequence = 0
        self.topic = topic
        self.topic_to_compare = topic_to_compare
        self.pulse = None
        self.pulseToCompare = None
        self.error = None
        self.timestamp = None
        self.date = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

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
            absolute_error = abs(self.pulseToCompare - self.pulse)
            self.error = (absolute_error / self.pulse) * 100
            time = rospy.Time.now()
            self.publish_error(time)
            self.write_to_csv(time)

    def publish_error(self, time):
        rospy.loginfo("[ComparePulseValues] Calculated error: " + str(self.error))
        msg_to_publish = Error()
        msg_to_publish.error = self.error
        msg_to_publish.time.stamp = time
        msg_to_publish.time.seq = self.published_error_value_sequence
        self.pub.publish(msg_to_publish)

        self.published_error_value_sequence += 1

    def write_to_csv(self, time):
        topic_csv = ""
        topic_to_compare_csv = ""

        if self.topic == "/pulse_chest_strap":
            topic_csv = "pulse_chest_strap"
        elif self.topic == "/ecg":
            topic_csv = "ecg"

        if self.topic_to_compare == "/pulse_head_movement":
            topic_to_compare_csv = "pulse_head_movement"

        filename = "pulse_measurement/compare/" + topic_csv + "_" + topic_to_compare_csv + "_compare_" + self.date + ".csv"

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        csv_file = open(filename, "a+")

        writer = csv.writer(csv_file)
        writer.writerow([time, self.error])


def main():
    rospy.init_node("compare", anonymous=False, log_level=rospy.DEBUG)

    topic = rospy.get_param("~topic", "/pulse_chest_strap")
    rospy.loginfo("[ComparePulseValues] Listening on topic '" + topic + "'")

    topic_to_compare = rospy.get_param("~topic_to_compare", "/pulse_head_movement/pulse")
    rospy.loginfo("[ComparePulseValues] Listening on topic '" + topic_to_compare + "'")

    pulse = ComparePulseValues(topic, topic_to_compare)
    pulse.run()


if __name__ == "__main__":
    sys.argv = rospy.myargv()
    main()
