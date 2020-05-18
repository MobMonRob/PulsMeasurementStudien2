from mne.preprocessing.ecg import qrs_detector
from pulse_publisher import PulsePublisher

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
        self.publisher = PulsePublisher("ecg")

    def run(self):
        self.signal, self.frequency = self.get_signal(name='EXG2')
        self.total_average_heart_rate, self.peaks = self.estimate_average_heartrate(self.signal, self.frequency)
        rospy.loginfo("[BdfProcessor] Total average heart rate: " + str(self.total_average_heart_rate))
        self.heart_rates = self.calculate_heart_rates(self.peaks, self.frequency)

        # Uncomment to plot the ECG signal
        # self.plot_signal(self.signal, self.frequency, 'EXG2')
        # plt.tight_layout()
        # plt.show()

    def process_frame(self, frame_count, video_fps, timestamp):
        """
        Synchronizes the processing of the video frames and the publishing of the ECG pulse values.
        :param frame_count: The index of the current processed frame.
        :param video_fps: The fps of the provided video
        :param timestamp: The timestamp of the current processed frame.
        """
        if self.pulse_sequence > len(self.heart_rates) - 1:
            return

        signal_position = frame_count / float(video_fps) * self.frequency

        if signal_position >= self.heart_rates[self.pulse_sequence][1]:
            self.publisher.publish(self.heart_rates[self.pulse_sequence][0], timestamp)
            self.pulse_sequence += 1

    def get_signal(self, name='EXG2'):
        """
        Extracts the ECG signal out of the BDF file.
        :param name: The name of the channel to extract the signal from.
        """

        reader = pyedflib.EdfReader(self.bdf_file)

        # Get index of ECG channel
        index = reader.getSignalLabels().index(name)
        # Get sample frequency of ECG
        frequency = reader.getSampleFrequency(index)

        # Get index of status channel
        status_index = reader.getSignalLabels().index('Status')
        # Read status signal
        status = reader.readSignal(status_index, 0).round().astype('int').nonzero()[0]

        # Determine start of the video file in signal with status bits
        video_start = status[0]

        # Determine the end of the video file. It seems like the status bits are set to 0 before the videos finishing
        # Therefore ignore the end of the signal and just cut the signal at the beginning.
        # You can ignore if the signal is longer than the video, the remaining values will not be published.
        # video_end = status[-1]
        # video_length = video_end - video_start

        # Read ECG signal and return as tuple with sample frequency
        return reader.readSignal(index, video_start), frequency

    def calculate_heart_rates(self, peaks, sampling_frequency):
        """
        Calculates multiple heart rates out of the signal in equidistant time slots.
        :param peaks: The detected peaks in the signal.
        :param sampling_frequency: The sampling frequency of the signal.
        :return: The calculated heart rates.
        """
        rates = (sampling_frequency * 60) / np.diff(peaks)
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

    def estimate_average_heartrate(self, signal, sampling_frequency):
        """
        Calculates the total average heart rate of the signal.
        :param peaks: The detected peaks in the signal.
        :param sampling_frequency: The sampling frequency of the signal.
        :return: The calculated average heart rate.
        """
        peaks = qrs_detector(sampling_frequency, signal)
        instantaneous_rates = (sampling_frequency * 60) / np.diff(peaks)

        # remove instantaneous rates which are lower than 30, higher than 240
        selector = (instantaneous_rates > 30) & (instantaneous_rates < 240)
        return float(np.nan_to_num(instantaneous_rates[selector].mean())), peaks

    def plot_signal(self, signal, sampling_frequency, channel_name):
        """
        Plots the ECG signal.
        :param signal: The signal to plot.
        :param sampling_frequency: The sampling frequency of the signal.
        :param channel_name: The channel name of the signal in the BDF file.
        """
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
