import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

WIDTH = 15
HEIGHT = 4
SAMPLE_RATE = 22050
DPI = 300


def plot_chromagram(sounds_path, sounds_labels):
    for sound_path, sound_label in zip(sounds_path, sounds_labels):
        plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI)

        audio_time_series, sample_rate = librosa.load(sound_path)
        stft = np.abs(librosa.stft(audio_time_series))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)

        librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
        plt.colorbar()
        plt.title(sound_label.title())

        plt.savefig(sound_path + '.png', bbox_inches='tight')


def plot_wave(sounds_path, sounds_labels):
    for sound_path, sound_label in zip(sounds_path, sounds_labels):
        plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI)

        audio_time_series, sample_rate = librosa.load(sound_path)

        librosa.display.waveplot(np.array(audio_time_series), sr=SAMPLE_RATE)

        plt.title(sound_label.title())

        plt.savefig(sound_path + '.png', bbox_inches='tight')
