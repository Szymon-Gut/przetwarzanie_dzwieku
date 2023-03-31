from src.recording import Recording
import matplotlib.pyplot as plt
import numpy as np


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
