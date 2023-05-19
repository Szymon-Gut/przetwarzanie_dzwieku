from src.recording import Recording
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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

def plot_frequency(recording: Recording, threshold=-0.1, alpha=0.1):
    sp = recording.fft_magnitude_spectrum_signal
    non_zero = np.where(sp > threshold)[0]
    freq = recording.fft_freqs_signal
    ax = sns.scatterplot(x=freq[non_zero], y=sp[non_zero], alpha=alpha)
    ax.set(xlabel = 'Frequency [Hz]', ylabel='Magnitude')

def plot_frame_frequency(recording: Recording, frame_idx = 0, threshold=-0.1, alpha=0.1):
    sp = recording.fft_magnitude_spectrum_frames[frame_idx, :]
    non_zero = np.where(sp > threshold)[0]
    freq = recording.fft_freqs_frames[frame_idx]
    ax = sns.scatterplot(x=freq[non_zero], y=sp[non_zero], alpha=alpha)
    ax.set(xlabel = 'Frequency [Hz]', ylabel='Magnitude')


def plot_FC(recording: Recording, alpha=1.0):
    fc = recording.FC()
    frames = np.arange(1, len(fc)+1)
    ax = sns.lineplot(x=frames, y=fc, alpha=alpha)
    ax.set(xlabel = 'Frame', ylabel='FC')

def plot_BW(recording: Recording, alpha=1.0):
    bw = recording.BW()
    frames = np.arange(1, len(bw)+1)
    ax = sns.lineplot(x=frames, y=bw, alpha=alpha)
    ax.set(xlabel = 'Frame', ylabel='BW')

def plot_SFM(recording: Recording, alpha=1.0):
    sfm = recording.SFM()
    frames = np.arange(1, len(sfm)+1)
    ax = sns.lineplot(x=frames, y=sfm, alpha=alpha)
    ax.set(xlabel = 'Frame', ylabel='SFM')

def plot_SCF(recording: Recording, alpha=1.0):
    scf = recording.SCF()
    frames = np.arange(1, len(scf)+1)
    ax = sns.lineplot(x=frames, y=scf, alpha=alpha)
    ax.set(xlabel = 'Frame', ylabel='SCF')

def plot_ERSB(recording: Recording, alpha=0.8):
    ersb1 = recording.ERSB1()
    ersb2 = recording.ERSB2()
    ersb3 = recording.ERSB3()
    frames = np.arange(1, len(ersb1)+1)
    ax = sns.lineplot(x=frames, y=ersb1, alpha=alpha, label='ERSB1')
    sns.lineplot(ax, x=frames, y=ersb2, alpha=alpha, label='ERSB2')
    sns.lineplot(ax, x=frames, y=ersb3, alpha=alpha, label='ERSB3')
    ax.set(xlabel = 'Frame', ylabel='ERSB')

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
    x = np.arange(num_frames) * frame_step / len(data) * recording.duration
    y = np.arange(frame_length//2+1) / frame_length * 44100
    plt.pcolormesh(x, y, frame_db.T, cmap='viridis')
    plt.xlabel('Czas [s]')
    plt.ylabel('Częstotliwość [Hz]')
    plt.ylim([0, 44100/2])
    plt.colorbar()
    plt.show()
