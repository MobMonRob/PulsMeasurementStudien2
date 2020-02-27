#!/usr/bin/env python
import datetime
import sys

import numpy as np
import rospy
import cv2

from cv2 import moveWindow
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from interface import plotXY, imshow, waitKey, destroyWindow
from processors import GetPulse


class PulseApp(object):
    def __init__(self, topic):
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.bridge = CvBridge()

        self.w, self.h = 0, 0
        self.pressed = 0

        # Basically, everything that isn't part of the GUI
        self.processor = GetPulse()

        # Init parameters for the cardiac data plot
        self.bpm_plot = False
        self.plot_title = "Data display - raw signal (top) and PSD (bottom)"

        # Maps keystrokes to specified methods
        # (A GUI window must have focus for these to work)
        self.key_controls = {"d": self.toggle_display_plot, "f": self.write_csv}

    def write_csv(self):
        fn = "Webcam-pulse" + str(datetime.datetime.now())
        fn = fn.replace(":", "_").replace(".", "_")
        data = np.vstack((self.processor.times, self.processor.samples)).T
        np.savetxt(fn + ".csv", data, delimiter=',')
        print("Writing csv")

    def toggle_display_plot(self):
        if self.bpm_plot:
            print("bpm plot disabled")
            self.bpm_plot = False
            destroyWindow(self.plot_title)
        else:
            print("bpm plot enabled")
            self.bpm_plot = True
            self.make_bpm_plot()
            moveWindow(self.plot_title, self.w, 0)

    def make_bpm_plot(self):
        plotXY([[self.processor.times,
                 self.processor.samples],
                [self.processor.freqs,
                 self.processor.fft]],
               labels=[False, True],
               showmax=[False, "bpm"],
               label_ndigits=[0, 0],
               showmax_digits=[0, 1],
               skip=[3, 3],
               name=self.plot_title,
               bg=self.processor.slices[0])

    def key_handler(self):
        self.pressed = waitKey(10) & 255  # wait for keypress for 10 ms
        if self.pressed == 27:  # exit program on 'esc'
            print("Exiting")
            rospy.signal_shutdown('Exiting')

        for key in self.key_controls.keys():
            if chr(self.pressed) == key:
                self.key_controls[key]()

    def callback(self, data):
        try:
            # Get current image frame from ros message
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        self.h, self.w, _c = frame.shape

        # display unaltered frame
        # imshow("Original", frame)

        # set current image frame to the processor's input
        self.processor.frame_in = frame
        # process the image frame to perform all needed analysis
        self.processor.run()
        # collect the output frame for display
        output_frame = self.processor.frame_out

        # show the processed/annotated output frame
        # imshow("Processed", output_frame)

        # create and/or update the raw data display if needed
        if self.bpm_plot:
            self.make_bpm_plot()

        # handle any key presses
        self.key_handler()


def main(args):
    rospy.init_node('webcam_pulse_detector', anonymous=True, log_level=rospy.DEBUG)

    # Get ROS topic from launch parameter
    topic = rospy.get_param("~topic", "/webcam/image_raw")
    rospy.loginfo("Listening on topic '" + topic + "'")

    PulseApp(topic)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv)
