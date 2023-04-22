import wave
import numpy as np
import scipy.signal

class Recording():

    __slots__ = ['recording', 'name']

    def __init__(self, path):
        self.recording = wave.open(path, 'rb')
        self.name = path.split('/')[-1][:-4]

    def __str__(self):
        return f"""
Audio Clip Summary
Name:       {self.name}

Duration:   {self.duration:.2f}
Frequency:  {self.frequency}
Samples:    {self.number_of_samples}
Channels:   {self.number_of_channels}

Detailed Metrics
Silent Ratio:   {self.calculate_sr():.2f}
VDR:            {self.calculate_vdr():.2f}
LSTER:          {self.calculate_lster():.2f}
HZCRR:          {self.calculate_hzcrr():.2f}
        """

    @property
    def duration(self):
        return self.number_of_samples/self.frequency

    @property
    def frequency(self):
        return self.recording.getframerate()

    @property
    def number_of_samples(self):
        return self.recording.getnframes()

    @property
    def number_of_channels(self):
        return self.recording.getnchannels()

    @property
    def samples(self):
        self.recording.rewind()
        frame_buffer = self.recording.readframes(-1)
        samples = np.frombuffer(frame_buffer, dtype=np.int16)
        return np.array(samples, dtype=np.int32)

    def get_volume_array(self, frame_length=0.02, hop_length=0.01):
        frame_size = int(self.frequency * frame_length)
        hop_size = int(self.frequency * hop_length)

        volume = []
        samples = self.samples
        for i in range(0, self.number_of_samples - frame_size, hop_size):
            frame = samples[i:i+frame_size]
            volume.append(np.sqrt(np.mean(frame**2)))
        max_value = np.max(np.abs(volume))
        volume = volume / max_value

        return volume

    def get_ste_array(self, frame_length=0.02, hop_length=0.01):
        frame_size = int(self.frequency * frame_length)
        hop_size = int(self.frequency * hop_length)

        ste = []
        samples = self.samples
        for i in range(0, self.number_of_samples - frame_size, hop_size):
            frame = samples[i:i+frame_size]
            ste.append(np.mean(frame**2))

        return ste

    def get_zcr_array(self, frame_length=0.02, hop_length=0.01):
        frame_size = int(self.frequency * frame_length)
        hop_size = int(self.frequency * hop_length)

        zcr = []
        samples = self.samples
        for i in range(0, self.number_of_samples-frame_size, hop_size):
            frame = samples[i:i+frame_size]
            zc = np.sum(np.abs(np.diff(np.sign(frame)))) / (2*frame_size)
            zcr.append(zc)

        return zcr

    def get_silence_array(self, frame_length=0.02, hop_length=0.01, volume_threshold=0.02, zcr_threshold=0.1):
        zcr_array = self.get_zcr_array(frame_length, hop_length)
        volume_array = self.get_volume_array(frame_length, hop_length)

        silence_array = [False] * len(zcr_array)
        for i in range(len(zcr_array)):
            if zcr_array[i] < zcr_threshold and volume_array[i] < volume_threshold:
                silence_array[i] = True

        return silence_array

    def calculate_sr(self, frame_length=0.02, hop_length=0.01, volume_threshold=0.02, zcr_threshold=0.1):
        silence_array = self.get_silence_array(
            frame_length, hop_length, volume_threshold, zcr_threshold)

        silence_duration = np.sum(silence_array) * hop_length
        return silence_duration / self.duration

    def calculate_vdr(self, frame_length=0.02, hop_length=0.01):
        volume = self.get_volume_array()
        max_volume = np.max(volume)
        min_volume = np.min(volume)
        return (max_volume - min_volume) / max_volume

    def calculate_lster(self, frame_length=0.02, hop_length=0.01):
        ste = self.get_ste_array(frame_length, hop_length)

        # moving average
        frames_per_one_second_window = int(1 / frame_length)
        mean_ste = running_mean(ste, frames_per_one_second_window)

        return np.mean(np.sign(mean_ste / 2 - ste) + 1) / 2

    def calculate_hzcrr(self, frame_length=0.02, hop_length=0.01):
        zcr = self.get_zcr_array(frame_length, hop_length)

        # moving average
        frames_per_one_second_window = int(1 / frame_length)
        mean_zcr = running_mean(zcr, frames_per_one_second_window)

        return np.mean(np.sign(zcr - 1.5 * mean_zcr) + 1) / 2
    
    def FC(self, window):
        windowed_sample = window(len(self.samples))
        freqs = np.fft.fftfreq(self.number_of_samples, d=1/self.frequency)
        freqs = freqs[:self.number_of_samples//2]
        fft = np.fft.fft(self.samples* windowed_sample)
        fc = np.sum(freqs * np.abs(fft)) / np.sum(np.abs(fft))
        return fc
    
    def BW(self, window):
        windowed_sample = window(len(self.samples))
        power_spectrum = np.abs(np.fft.fft(self.samples*windowed_sample))**2
        freqs = np.fft.fftfreq(len(power_spectrum), d=1/self.frequency)
        bandwidth = np.sum(power_spectrum * (freqs >= 0)) / np.sum(power_spectrum)
        return bandwidth

    def BE(self, window):
        freq_ranges = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500),
               (500, 600), (600, 700), (700, 800), (800, 900), (900, 1000)]
        windowed_sample = window(len(self.samples))
        power_spectrum = np.abs(np.fft.fft(self.samples*windowed_sample))**2
        freqs = np.fft.fftfreq(len(power_spectrum), d=1/self.frequency)
        be_values = []
        for freq_range in freq_ranges:
            low, high = freq_range
            band_mask = np.logical_and(freqs >= low, freqs < high)
            band_power = np.sum(power_spectrum[band_mask])
            be_values.append(band_power)
        #normalizacja BE do przedzialu [0,1]
        be_values /= np.sum(be_values)
        return be_values, freq_ranges

    def BER(self, window):
        be_values, freq_ranges = self.BE(window)
        ber = np.max(be_values) / np.sum(be_values)
        return ber, freq_ranges
    

    def SFM(self, window): #TODO
        spectrum = np.fft.fft(self.samples)
        windowed = window(len(self.samples))
        spectrum *= windowed
        powered_spectrum = np.abs(spectrum)**2
        signal_len = len(self.samples)
        freqs = np.fft.fftfreq(len(self.samples), d=1.0/self.frequency)
        positive_freqs = freqs[:len(freqs)//2]
        sfm_values = []
        for i in range(positive_freqs):
            segment = powered_spectrum[i]
            arithmetic_mean = np.mean(np.abs(segment))
            geometric_mean = np.exp(np.mean(np.log(np.abs(segment))))
            SFM = geometric_mean / arithmetic_mean
            sfm_values.append(SFM)
        return sfm_values, positive_freqs

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    mean = (cumsum[N:] - cumsum[:-N]) / float(N)
    return np.concatenate([[mean[0]] * (N-1), mean])


def main():
    recording = Recording('recordings/female.wav')
    recording.get_volume_array()


if __name__ == '__main__':
    main()
