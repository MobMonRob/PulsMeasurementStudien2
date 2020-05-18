from common.msg import Pulse
from datetime import datetime

import csv
import os
import rospy


class PulsePublisher:

    def __init__(self, name):
        self.name = name
        self.topic = "/" + name
        self.publisher = rospy.Publisher(self.topic, Pulse, queue_size=10)
        self.sequence = 0
        self.date = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    def publish(self, pulse, timestamp):
        """
        Publishes the pulse value to ROS and also writes it into a csv file.
        :param pulse: The pulse value to publish in ROS and to write into a csv file.
        :param timestamp: The timestamp corresponding to the pulse value.
        """
        rospy.loginfo("[PulsePublisher] Publishing pulse ('" + self.topic + "'): " + str(pulse))
        self.publish_to_ros(pulse, timestamp)
        self.write_to_csv(pulse, timestamp)

    def publish_to_ros(self, pulse, timestamp):
        """
        Publishes the pulse value to ROS using self.topic as topic.
        :param pulse: The pulse value to publish in ROS.
        :param timestamp: The timestamp corresponding to the pulse value.
        """
        ros_msg = Pulse()
        ros_msg.pulse = pulse
        ros_msg.time.stamp = timestamp
        ros_msg.time.seq = self.sequence

        self.publisher.publish(ros_msg)
        self.sequence += 1

    def write_to_csv(self, pulse, timestamp):
        """
        Writes the pulse value into a csv file.
        The file will be created inside the ROS_HOME directory using the name of the publisher
        and the start date of the script.
        :param pulse: The pulse value to write.
        :param timestamp: The timestamp corresponding to the pulse value.
        """
        filename = "pulse_measurement/" + self.name + "/pulses_" + self.date + ".csv"

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        csv_file = open(filename, "a+")
        writer = csv.writer(csv_file)
        writer.writerow([timestamp, pulse])
