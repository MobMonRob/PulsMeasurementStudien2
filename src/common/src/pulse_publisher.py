from common.msg import Pulse

import csv
import rospy


class PulsePublisher:

    def __init__(self, name):
        self.name = name
        self.topic = "/" + name
        self.publisher = rospy.Publisher(self.topic, Pulse, queue_size=10)
        self.sequence = 0

    def publish(self, pulse, timestamp):
        rospy.loginfo("[PulsePublisher] Publishing pulse ('" + self.topic + "'): " + str(pulse))
        self.publish_to_ros(pulse, timestamp)
        self.write_to_csv(pulse, timestamp)

    def publish_to_ros(self, pulse, timestamp):
        ros_msg = Pulse()
        ros_msg.pulse = pulse
        ros_msg.time.stamp = timestamp
        ros_msg.time.seq = self.sequence

        self.publisher.publish(ros_msg)
        self.sequence += 1

    def write_to_csv(self, pulse, timestamp):
        csv_file = open(self.name + "_pulse.csv", "a+")
        writer = csv.writer(csv_file)
        writer.writerow([timestamp, pulse])
