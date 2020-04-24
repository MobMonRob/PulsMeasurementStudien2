from mne.preprocessing.ecg import qrs_detector
from face_detection.msg import ECG

import matplotlib.pyplot as plt
import numpy as np
import pyedflib
import rospy


class BdfProcessor:

    def __init__(self, bdf_file):
        self.bdf_file = bdf_file
        self.pulse_sequence = 0
        self.signal = None
        self.peaks = None
        self.frequency = None
        self.heart_rates = None
        self.total_average_heart_rate = None
        self.ecg_publisher = rospy.Publisher("/face_detection/ecg", ECG, queue_size=10)

    def run(self):
        self.signal, self.frequency = self.get_signal(name='EXG2')
        self.total_average_heart_rate, self.peaks = self.estimate_average_heartrate(self.signal, self.frequency)
        rospy.loginfo("Total average heart rate: " + str(self.total_average_heart_rate))

        self.heart_rates = self.calculate_heart_rates(self.peaks, self.frequency)

        # self.plot_signal(self.signal, self.frequency, 'EXG2')
        # plt.tight_layout()
        # plt.show()

    def process_frame(self, frame_count, fps, timestamp):
        if self.pulse_sequence > len(self.heart_rates) - 1:
            return

        signal_position = frame_count / float(fps) * self.frequency

        if signal_position >= self.heart_rates[self.pulse_sequence][1]:
            self.publish_pulse(self.heart_rates[self.pulse_sequence][0], timestamp)

    def get_signal(self, name='EXG2'):
        reader = pyedflib.EdfReader(self.bdf_file)

        # Get index of ECG channel
        index = reader.getSignalLabels().index(name)
        # Get sample frequency of ECG
        frequency = reader.getSampleFrequency(index)

        # Get index of status channel
        status_index = reader.getSignalLabels().index('Status')
        # Read status signal
        status = reader.readSignal(status_index, 0).round().astype('int').nonzero()[0]

        # Determine start end end of video file in signal with status bits
        video_start = status[0]
        video_end = status[-1]

        # Read ECG signal and return as tuple with sample frequency
        return reader.readSignal(index, video_start, video_end - video_start), frequency

    def calculate_heart_rates(self, peaks, frequency):
        rates = (frequency * 60) / np.diff(peaks)
        # Remove instantaneous rates which are lower than 30, higher than 240
        selector = (rates > 30) & (rates < 240)
        rates = rates[selector]

        heart_rates = []
        hr = rates[:10]
        heart_rates.append((hr.mean(), peaks[10]))

        for index, rate in enumerate(rates[10:]):
            hr[:-1] = hr[1:]
            hr[-1] = rate
            heart_rates.append((hr.mean(), peaks[index + 11]))

        return heart_rates

    def publish_pulse(self, pulse, time):
        ros_msg = ECG()
        ros_msg.pulse = pulse
        ros_msg.time.stamp = time
        ros_msg.time.seq = self.pulse_sequence

        self.ecg_publisher.publish(ros_msg)
        self.pulse_sequence += 1

    def estimate_average_heartrate(self, signal, sampling_frequency):
        peaks = qrs_detector(sampling_frequency, signal)
        instantaneous_rates = (sampling_frequency * 60) / np.diff(peaks)

        # remove instantaneous rates which are lower than 30, higher than 240
        selector = (instantaneous_rates > 30) & (instantaneous_rates < 240)
        return float(np.nan_to_num(instantaneous_rates[selector].mean())), peaks

    def plot_signal(self, signal, sampling_frequency, channel_name):
        avg, peaks = self.estimate_average_heartrate(signal, sampling_frequency)
        ax = plt.gca()
        ax.plot(np.arange(0, len(signal) / sampling_frequency, 1 / sampling_frequency), signal, label='Raw signal')
        xmin, xmax, ymin, ymax = plt.axis()
        ax.vlines(peaks / sampling_frequency, ymin, ymax, colors='r', label='P-T QRS detector')
        plt.xlim(0, len(signal) / sampling_frequency)
        plt.ylabel('uV')
        plt.xlabel('time (s)')
        plt.title('Channel %s - Average heart-rate = %d bpm' % (channel_name, avg))
        ax.grid(True)
        ax.legend(loc='best', fancybox=True, framealpha=0.5)

        return avg, peaks
