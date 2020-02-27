import numpy as np
import time
import cv2
import pylab
import os


class GetPulse(object):

    def __init__(self):
        self.frame_in = np.zeros((10, 10))
        self.frame_out = np.zeros((10, 10))
        self.fps = 0
        self.buffer_size = 250
        self.data_buffer = []
        self.times = []
        self.ttimes = []
        self.samples = []
        self.freqs = []
        self.fft = []
        self.slices = [[0]]
        self.t0 = time.time()
        self.bpms = []
        self.bpm = 0
        self.face_cascade = cv2.CascadeClassifier(
            os.path.dirname(os.path.realpath(__file__)) + "/../resources/cascade.xml")

        self.last_center = np.array([0, 0])
        self.last_wh = np.array([0, 0])
        self.output_dim = 13

        self.idx = 1
        self.find_faces = True

    def get_subface_means(self):
        v1 = np.mean(self.frame_in[:, :, 0])
        v2 = np.mean(self.frame_in[:, :, 1])
        v3 = np.mean(self.frame_in[:, :, 2])

        return (v1 + v2 + v3) / 3.

    def plot(self):
        data = np.array(self.data_buffer).T
        np.savetxt("data.dat", data)
        np.savetxt("times.dat", self.times)
        freqs = 60. * self.freqs
        idx = np.where((freqs > 50) & (freqs < 180))
        pylab.figure()
        n = data.shape[0]
        for k in xrange(n):
            pylab.subplot(n, 1, k + 1)
            pylab.plot(self.times, data[k])
        pylab.savefig("data.png")
        pylab.figure()
        for k in xrange(self.output_dim):
            pylab.subplot(self.output_dim, 1, k + 1)
            pylab.plot(self.times, self.pcadata[k])
        pylab.savefig("data_pca.png")

        pylab.figure()
        for k in xrange(self.output_dim):
            pylab.subplot(self.output_dim, 1, k + 1)
            pylab.plot(freqs[idx], self.fft[k][idx])
        pylab.savefig("data_fft.png")
        quit()

    def run(self):
        self.times.append(time.time() - self.t0)
        self.frame_out = self.frame_in
        self.gray = cv2.equalizeHist(cv2.cvtColor(self.frame_in, cv2.COLOR_BGR2GRAY))
        col = (100, 255, 100)
        # cv2.putText(self.frame_out, "Press 'S' to restart", (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        # cv2.putText(self.frame_out, "Press 'D' to toggle data plot", (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, col)
        # cv2.putText(self.frame_out, "Press 'Esc' to quit", (10, 75), cv2.FONT_HERSHEY_PLAIN, 1.5, col)

        vals = self.get_subface_means()

        self.data_buffer.append(vals)
        L = len(self.data_buffer)
        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            L = self.buffer_size

        processed = np.array(self.data_buffer)
        self.samples = processed
        if L > 10:
            self.output_dim = processed.shape[0]

            self.fps = float(L) / (self.times[-1] - self.times[0])
            even_times = np.linspace(self.times[0], self.times[-1], L)
            interpolated = np.interp(even_times, self.times, processed)
            interpolated = np.hamming(L) * interpolated
            interpolated = interpolated - np.mean(interpolated)
            raw = np.fft.rfft(interpolated)
            phase = np.angle(raw)
            self.fft = np.abs(raw)
            self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)

            freqs = 60. * self.freqs
            idx = np.where((freqs > 20) & (freqs < 240))

            pruned = self.fft[idx]
            phase = phase[idx]

            pfreq = freqs[idx]
            self.freqs = pfreq
            self.fft = pruned
            idx2 = np.argmax(pruned)

            self.bpm = self.freqs[idx2]
            self.idx += 1

            self.frame_out = self.frame_in

            gap = (self.buffer_size - L) / self.fps
            # self.bpms.append(bpm)
            # self.ttimes.append(time.time())
            if gap:
                print("(Estimated: %0.1f bpm, wait %0.0f s)" % (self.bpm, gap))
            else:
                print("(Estimated: %0.1f bpm)" % self.bpm)

            # cv2.putText(self.frame_out, text, (int(x - w / 2), int(y)), cv2.FONT_HERSHEY_PLAIN, tsize, col)
