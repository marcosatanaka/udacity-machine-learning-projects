import numpy as np
import librosa


def get_features(sounds_paths):
    features = np.empty((0, 12))

    for sound_path in sounds_paths:
        try:
            chroma = get_chroma(sound_path)
        except Exception as e:
            print "Error encountered while parsing file: ", sound_path
            raise

        ext_features = np.hstack([chroma])
        features = np.vstack([features, ext_features])

    return np.array(features)


def get_chroma(sound_path):
    audio_time_series, sample_rate = librosa.load(sound_path)
    stft = np.abs(librosa.stft(audio_time_series))

    # chroma_stft: retorna a quantidade de cada chroma a cada frame
    # [0] 0.3, 0.3, 0.4
    # [1] 0.7, 0.7, 0.6
    # [2] 0.0, 0.0, 0.0
    #
    # fazemos transpose e media para fazer a media de cada chroma
    # [0] 0.3
    # [1] 0.6
    # [2] 0.0
    return np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
