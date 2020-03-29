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
        self.ecg_publisher = rospy.Publisher("/face_detection/ecg", ECG, queue_size=10)

    def run(self):
        signal, frequency, duration = self.get_signal(name='EXG2')
        avg_hr, peaks = self.plot_signal(signal, frequency, 'EXG2')
        rospy.loginfo("Average heart rate: " + str(avg_hr))

        # plt.tight_layout()
        # plt.show()

        peaks = qrs_detector(frequency, signal)
        rates = self.get_instantaneous_rates(peaks, frequency)
        self.calculate_heart_rates(rates, peaks, len(signal), duration)

    def calculate_heart_rates(self, rates, peaks, length, duration):
        hr = rates[:20]

        for index, rate in enumerate(rates[20:]):
            hr = np.roll(hr, -1)
            hr[-1] = rate

            percentage = peaks[index + 21] / float(length)
            offset = (percentage * duration)
            time = rospy.Time.now() + rospy.Duration.from_sec(offset)
            self.publish_pulse(hr.mean(), time)

    def get_instantaneous_rates(self, peaks, frequency):
        rates = (frequency * 60) / np.diff(peaks)

        # Remove instantaneous rates which are lower than 30, higher than 240
        selector = (rates > 30) & (rates < 240)
        return rates[selector]

    def publish_pulse(self, pulse, time):
        ros_msg = ECG()
        ros_msg.pulse = pulse
        ros_msg.time.stamp = time
        ros_msg.time.seq = self.pulse_sequence

        self.ecg_publisher.publish(ros_msg)
        self.pulse_sequence += 1

    def get_signal(self, name='EXG2'):
        # Read signal
        reader = pyedflib.EdfReader(self.bdf_file)
        index = reader.getSignalLabels().index(name)
        frequency = reader.samplefrequency(index)
        signal = reader.readSignal(index, digital=False)

        # Filter end of signal
        length = len(signal)
        selector = signal < -100
        signal = signal[selector]

        # Calculate new duration with filtered signal
        percentage = len(signal) / float(length)
        duration = reader.getFileDuration() * percentage
        return signal, frequency, duration

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

    def estimate_average_heartrate(self, signal, sampling_frequency):
        peaks = qrs_detector(sampling_frequency, signal)
        instantaneous_rates = (sampling_frequency * 60) / np.diff(peaks)

        # remove instantaneous rates which are lower than 30, higher than 240
        selector = (instantaneous_rates > 30) & (instantaneous_rates < 240)
        return float(np.nan_to_num(instantaneous_rates[selector].mean())), peaks

