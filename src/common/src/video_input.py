#!/usr/bin/env python
from bdf_processor import BdfProcessor
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

import cv2
import rospy
import time


class VideoInput:

    def __init__(self, topic, cascade_file):
        self.topic = topic
        self.cascade_file = cascade_file
        self.bridge = CvBridge()
        self.bdf_processor = None
        self.frame_callback = None
        self.video_file = None
        self.video_fps = None
        self.total_video_frames = None
        self.video_duration = None
        self.frame_count = 0
        self.start_time = None
        self.fps_start_time = 0

    def run(self, video_file, bdf_file):
        self.video_file = video_file

        # Start bdf processor
        if bdf_file and bdf_file != "None":
            self.bdf_processor = BdfProcessor(bdf_file)
            self.bdf_processor.run()

        self.fps_start_time = time.time()
        self.start_time = rospy.Time.now()

        if self.video_file:
            capture = cv2.VideoCapture(self.video_file)
            self.video_fps = int(capture.get(cv2.CAP_PROP_FPS))
            self.total_video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_duration = self.total_video_frames / float(self.video_fps)

            while capture.isOpened() and not rospy.is_shutdown() and self.frame_count < self.total_video_frames:
                ret, frame = capture.read()
                self.on_image_frame(frame, convert=False)

            capture.release()
        else:
            rospy.Subscriber(self.topic, Image, self.on_image_frame)

    def on_image_frame(self, data, convert=True):
        self.frame_count += 1
        self.calculate_fps()
        timestamp = self.get_timestamp()

        if convert:
            try:
                # Convert ROS image to OpenCV image
                cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                rospy.logerr(e)
                return
        else:
            cv_image = data

        if self.bdf_processor:
            self.bdf_processor.process_frame(self.frame_count, self.video_fps, timestamp)

        if self.frame_callback:
            self.frame_callback(cv_image, timestamp)

    def calculate_fps(self):
        if self.frame_count % 60 is 0:
            fps_end_time = time.time()
            seconds = fps_end_time - self.fps_start_time
            fps = 60 / seconds
            rospy.loginfo("[VideoInput] Estimated FPS: " + str(fps) + " (Measured timespan: " + str(seconds) + "s)")
            self.fps_start_time = time.time()

    def get_timestamp(self):
        if not self.video_file:
            return rospy.Time.now() - self.start_time

        percentage = self.frame_count / float(self.total_video_frames)
        offset = (percentage * self.video_duration)
        return rospy.Time.from_sec(0) + rospy.Duration.from_sec(offset)
