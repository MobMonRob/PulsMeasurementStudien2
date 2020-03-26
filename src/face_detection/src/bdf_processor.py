from mne.preprocessing.ecg import qrs_detector

import matplotlib.pyplot as plt
import numpy as np
import os
import pyedflib


class BdfProcessor:

    def __init__(self, bdf_file):
        self.bdf_file = bdf_file

    def run(self):
        estimates = []

        for i, channel in enumerate(('EXG1', 'EXG2', 'EXG3')):
            plt.subplot(3, 1, i + 1)
            signal, freq = self.bdf_load_signal(name=channel)
            avg_hr, peaks = self.plot_signal(signal, freq, channel)
            estimates.append(avg_hr)

        plt.tight_layout()
        plt.show()

    def bdf_load_signal(self, name='EXG2', start=None, end=None):
        if not os.path.exists(self.bdf_file):  # or the EdfReader will crash the interpreter
            raise IOError("file `%s' does not exist" % self.bdf_file)

        with pyedflib.EdfReader(self.bdf_file) as e:
            # get the status information, so we how the video is synchronized
            status_index = e.getSignalLabels().index('Status')
            sample_frequency = e.samplefrequency(status_index)
            status_size = e.samples_in_file(status_index)
            status = np.zeros((status_size,), dtype='float64')
            e.readsignal(status_index, 0, status_size, status)
            status = status.round().astype('int')
            nz_status = status.nonzero()[0]

            # because we're interested in the video bits, make sure to get data
            # from that period only
            video_start = nz_status[0]
            video_end = nz_status[-1]

            # retrieve information from this rather chaotic API
            index = e.getSignalLabels().index(name)
            sample_frequency = e.samplefrequency(index)

            video_start_seconds = video_start / sample_frequency

            if start is not None:
                start += video_start_seconds
                start *= sample_frequency
                if start < video_start: start = video_start
                start = int(start)
            else:
                start = video_start

            if end is not None:
                end += video_start_seconds
                end *= sample_frequency
                if end > video_end: end = video_end
                end = int(end)
            else:
                end = video_end

            # now read the data into a numpy array (read everything)
            container = np.zeros((end - start,), dtype='float64')
            e.readsignal(index, start, end - start, container)

            return container, sample_frequency

    def estimate_average_heartrate(self, s, sampling_frequency):
        peaks = qrs_detector(sampling_frequency, s)
        instantaneous_rates = (sampling_frequency * 60) / np.diff(peaks)

        # remove instantaneous rates which are lower than 30, higher than 240
        selector = (instantaneous_rates > 30) & (instantaneous_rates < 240)
        return float(np.nan_to_num(instantaneous_rates[selector].mean())), peaks

    def plot_signal(self, s, sampling_frequency, channel_name):
        avg, peaks = self.estimate_average_heartrate(s, sampling_frequency)

        ax = plt.gca()
        ax.plot(np.arange(0, len(s) / sampling_frequency, 1 / sampling_frequency), s, label='Raw signal')
        xmin, xmax, ymin, ymax = plt.axis()
        ax.vlines(peaks / sampling_frequency, ymin, ymax, colors='r', label='P-T QRS detector')
        plt.xlim(0, len(s) / sampling_frequency)
        plt.ylabel('uV')
        plt.xlabel('time (s)')
        plt.title('Channel %s - Average heart-rate = %d bpm' % (channel_name, avg))
        ax.grid(True)
        ax.legend(loc='best', fancybox=True, framealpha=0.5)

        return avg, peaks
