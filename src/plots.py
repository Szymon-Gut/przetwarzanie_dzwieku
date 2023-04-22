from src.recording import Recording
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sc

def plot_amplitude(recording: Recording):
    times = np.linspace(0, recording.duration, num=recording.number_of_samples)

    plt.plot(times, recording.samples)
    plt.title('Amplitude')
    plt.ylabel('Signal Value')
    plt.xlabel('Time (s)')
    plt.xlim(0, recording.duration)

    plt.figure(figsize=(15, 5))
    plt.show()


def plot_volume(recording: Recording, frame_length=0.02, hop_length=0.01):
    volume_array = recording.get_volume_array(frame_length, hop_length)
    hop_size = int(recording.frequency * hop_length)
    times_frames = np.arange(0, len(volume_array)) * \
        hop_size / recording.frequency

    plt.plot(times_frames, volume_array)
    plt.title('Volume per frame')
    plt.ylabel('Volume')
    plt.xlabel('Time (s)')

    plt.figure(figsize=(15, 5))
    plt.show()


def plot_ste(recording: Recording, frame_length=0.02, hop_length=0.01):
    ste_array = recording.get_ste_array(frame_length, hop_length)
    hop_size = int(recording.frequency * hop_length)
    times_frames = np.arange(0, len(ste_array)) * \
        hop_size / recording.frequency

    plt.plot(times_frames, ste_array)
    plt.title('Short Time Energy (STE)')
    plt.ylabel('STE')
    plt.xlabel('Time (s)')

    plt.figure(figsize=(15, 5))
    plt.show()


def plot_zcr(recording: Recording, frame_length=0.02, hop_length=0.01):
    zcr_array = recording.get_zcr_array(frame_length, hop_length)
    hop_size = int(recording.frequency * hop_length)
    times_frames = np.arange(0, len(zcr_array)) * \
        hop_size / recording.frequency

    plt.plot(times_frames, zcr_array)
    plt.xlabel('Time (s)')
    plt.ylabel('ZCR')

    plt.figure(figsize=(15, 5))
    plt.show()


def plot_silence(recording: Recording, frame_length=0.02, hop_length=0.01, volume_threshold=0.02, zcr_threshold=0.1):
    volume_array = recording.get_volume_array(frame_length, hop_length)
    zcr_array = recording.get_zcr_array(frame_length, hop_length)
    silence_array = recording.get_silence_array(
        frame_length, hop_length, volume_threshold, zcr_threshold)

    hop_size = int(recording.frequency * hop_length)
    times_frames = np.arange(0, len(zcr_array)) * \
        hop_size / recording.frequency

    plt.plot(times_frames, zcr_array, color="green", label="zcr")
    plt.plot(times_frames, volume_array, color="blue", label="volume")
    plt.bar(times_frames, silence_array, alpha=0.3, width=0.01,
            color="red", label="detected silence")
    plt.title('Detected Silence')
    plt.ylabel('Volume/STE/Silence Rate')
    plt.xlabel('Time (s)')
    plt.legend()

    plt.figure(figsize=(15, 5))
    plt.show()


def plot_frequency_centroid(recording: Recording, window):
    windowed_sample = window(len(recording.samples))
    freqs = np.fft.fftfreq(recording.number_of_samples, d=1/recording.frequency)
    freqs = freqs[:recording.number_of_samples//2]
    fft = np.fft.fft(recording.samples) * windowed_sample
    fc= recording.FC()
    plt.figure(figsize=(8,4))
    plt.plot(freqs, fft, color='blue')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('FFT Spectrum')

    plt.axvline(x=fc, color='red', linestyle='--', linewidth=2, label='Frequency Centroid')

    plt.legend()
    plt.show()


def plot_effective_bandwith(recording:Recording, window):
    windowed_sample = window(len(recording.samples))
    power_spectrum = np.abs(np.fft.fft(recording.samples)*windowed_sample)**2
    freqs = np.fft.fftfreq(len(power_spectrum), d=1/recording.frequency)
    bandwidth = recording.BW(window)
    plt.figure(figsize=(8,4))
    plt.plot(freqs, power_spectrum, color='blue')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectrum')
    plt.title('Power Spectrum')
    plt.axvline(x=bandwidth, color='red', linestyle='--', linewidth=2, label='Effective Bandwidth')

    plt.legend()
    plt.show()


def plot_band_energy(recording:Recording, window):
    be_values, freq_ranges = recording.BE()
    fig, ax = plt.subplots()
    ax.bar(range(len(freq_ranges)), be_values)
    ax.set_xticks(range(len(freq_ranges)))
    ax.set_xticklabels([f"{r[0]}-{r[1]} Hz" for r in freq_ranges], rotation=45)
    ax.set_ylabel("Band Energy")
    ax.set_xlabel("Frequency Range")

    plt.show()


def plot_band_energy_ratio(recording:Recording, window):
     
    ber, freq_ranges = recording.BER(window)
    plt.bar(["BER", ""], [ber, 1-ber])
    plt.ylim([0, 1])
    plt.title("Band Energy Ratio (BER)")
    plt.ylabel("Ratio")
    plt.show()


def plot_sfm(recording:Recording, window): #TODO
    sfm_values, positive_freqs = recording.SFM(window)
    plt.scatter(positive_freqs, sfm_values, s=1)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Spectral Flatness Measure')
    plt.show()


def plot_spectrogram(recording:Recording, frame_length, frame_overlap, window):
    data = recording.samples
    frame_step = frame_length - frame_overlap
    num_frames = int(np.ceil((len(data) - frame_length + frame_overlap) / frame_step))
    frames = np.zeros((num_frames, frame_length))
    for i in range(num_frames):
        start = i * frame_step
        end = start + frame_length
        if end > len(data):
            frames[i, :len(data)-start] = data[start:]
        else:
            frames[i, :] = data[start:end]
    windowed_frames = frames * window(frame_length)
    frame_fft = np.fft.fft(windowed_frames, axis=1)[:,:frame_length//2+1]
    frame_amp = np.abs(frame_fft)
    frame_db = 20 * np.log10(frame_amp)
    x = np.arange(num_frames) * frame_step / len(data)
    y = np.arange(frame_length//2+1) / frame_length * 44100
    plt.pcolormesh(x, y, frame_db.T, cmap='viridis')
    plt.xlabel('Czas [s]')
    plt.ylabel('Częstotliwość [Hz]')
    plt.ylim([0, 44100/2])
    plt.colorbar()
    plt.show()


def plot_vocal_tract_freq(recording:Recording, window):
    # obliczanie cepstrum
    signal = recording.samples
    sample_rate = recording.frequency
    fft_signal = np.fft.fft(signal*window(len(signal)))
    log_spectrum = np.log(np.abs(fft_signal))
    cepstrum = np.fft.ifft(log_spectrum)

    # wybór indeksów dla częstotliwości formantów
    min_idx = int(sample_rate / 400)
    max_idx = int(sample_rate / 50)
    cep_cut = cepstrum[min_idx:max_idx]
    peaks_idx = np.argmax(np.abs(cep_cut))
    formants_idx = np.zeros(4, dtype=int)
    formants_idx[0] = min_idx + peaks_idx
    cep_cut[peaks_idx] = -1000

    for i in range(1, 4):
        peaks_idx = np.argmax(np.abs(cep_cut))
        formants_idx[i] = min_idx + peaks_idx
        cep_cut[peaks_idx] = -1000

    # konwersja indeksów częstotliwości na częstotliwości w Hz
    formants_hz = sample_rate / formants_idx

    # rysowanie wykresu
    plt.plot(np.arange(len(signal)) / sample_rate, signal)
    plt.xlim([0, len(signal) / sample_rate])
    plt.xlabel('Czas (s)')
    plt.ylabel('Amplituda')
    for i, f in enumerate(formants_hz):
        plt.axvline(x=1/f, color='r', linestyle='--')
        plt.text(1/f + 0.001, 0.8*(i+1), f'{f:.0f} Hz', color='r')

    plt.title('Wykres częstotliwości formantów krtaniowych')
    plt.show()