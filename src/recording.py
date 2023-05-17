import wave
import numpy as np
import scipy.signal.windows as wn

class Recording():

    __slots__ = ['recording', 'name', 'window_function']

    def __init__(self, path):
        self.recording = wave.open(path, 'rb')
        self.name = path.split('/')[-1][:-4]
        self.window_function = self.identity

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

    @property
    def windowed_samples(self):
        window = self.window_function(self.number_of_samples)
        return self.samples*window
    
    @property
    def fft_magnitude_spectrum_signal(self):
        fft_result = np.fft.fft(self.windowed_samples)
        return np.abs(fft_result)

    @property
    def fft_magnitude_spectrum_frames(self):
        fft_result = np.fft.fft(self.get_frames(), axis=1)
        return np.abs(fft_result)

    @property
    def fft_freqs_signal(self):
        freq_vals = np.fft.fftfreq(
            len(self.windowed_samples),
            1 / self.frequency
        )
        return freq_vals

    @property
    def fft_freqs_frames(self):
        freq_vals = [np.fft.fftfreq(len(frame), 1 / self.frequency)
                     for frame in self.get_frames()]
        return np.array(freq_vals)
    
    def set_window_function(self, function):
        if function=='identity':
            self.window_function = self.identity
        elif function=='hamming':
            self.window_function = self.hamming
        elif function=='hanning':
            self.window_function = self.hanning
        elif function=='triangle':
            self.window_function = self.triangle
        elif function=='blackman':
            self.window_function = self.blackman
        elif function=='taylor':
            self.window_function = self.taylor

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

    def get_frames(self, frame_length=0.02, hop_length=0.01):
        frame_size = int(self.frequency * frame_length)
        hop_size = int(self.frequency * hop_length)
        samples = self.windowed_samples
        frames=[]
        for i in range(0, self.number_of_samples - frame_size, hop_size):
            frame = samples[i:i+frame_size]
            frames.append(frame)
        return frames
    

    
    # __FREQ__ANALYSIS__

    def hamming(self, x):
        return wn.hamming(x)

    def hanning(self, x):
        return wn.hanning(x)
        
    def triangle(self, x):
        return wn.triang(x)
    
    def blackman(self, x):
        return wn.blackman(x)
    
    def taylor(self, x):
        return  wn.taylor(x)
    
    def identity(self, x):
        return np.ones(x)
    
    def volume2(self):
        return np.mean(self.fft_magnitude_spectrum_frames ** 2, axis=1)
    
    def FC(self):
        numerator = np.sum(
            self.fft_freqs_frames * self.fft_magnitude_spectrum_frames,
            axis=1
        )
        denominator = np.sum(self.fft_magnitude_spectrum_frames, axis=1)
        return numerator / denominator
    
    def BW(self):
        fc = self.FC()
        squared_spectrum_magnitude = self.fft_magnitude_spectrum_frames ** 2
        numerator_inner = \
            (self.fft_magnitude_spectrum_frames - fc.reshape(-1, 1)) ** 2 * squared_spectrum_magnitude
        return np.sqrt(
            np.sum(numerator_inner, axis=1) /
            np.sum(squared_spectrum_magnitude, axis=1)
        )
    
    def SFM(self):
        geometric_mean = np.exp(np.mean(np.log(
            self.fft_magnitude_spectrum_frames ** 2), axis=1))
        arithmetic_mean = np.mean(self.fft_magnitude_spectrum_frames ** 2, axis=1)
        arithmetic_mean = np.where(arithmetic_mean == 0, 1, arithmetic_mean)
        return geometric_mean / arithmetic_mean

    def SCF(self):
        max_squared_magnitude = np.max(self.fft_magnitude_spectrum_frames ** 2, axis=1)
        arithmetic_mean = np.mean(self.fft_magnitude_spectrum_frames ** 2, axis=1)
        return max_squared_magnitude / arithmetic_mean
    
    def BE(self, freq_lower_bound, freq_upper_bound):
        freqs_masks = [np.where((freqs_frame >= freq_lower_bound) &
                                    (freqs_frame <= freq_upper_bound))
                           for freqs_frame in self.fft_freqs_frames]
        relevant_magnitudes_squared = [frame[mask] ** 2 for frame, mask in
                                  zip(self.fft_magnitude_spectrum_frames, freqs_masks)]
        return np.array([np.sum(frame_magn_squared) for frame_magn_squared in relevant_magnitudes_squared])
    
    def _ERSB(self, freq_lower_bound, freq_upper_bound):
        return self.BE(freq_lower_bound, freq_upper_bound) / \
            np.sum(self.fft_magnitude_spectrum_frames ** 2, axis=1)
    
    def ERSB1(self):
        return self._ERSB(0, 630)

    def ERSB2(self):
        return self._ERSB(630, 1720)

    def ERSB3(self):
        return self._ERSB(1720, 4400)
    
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    mean = (cumsum[N:] - cumsum[:-N]) / float(N)
    return np.concatenate([[mean[0]] * (N-1), mean])

def main():
    recording = Recording('recordings/female.wav')
    recording.get_volume_array()


if __name__ == '__main__':
    main()
