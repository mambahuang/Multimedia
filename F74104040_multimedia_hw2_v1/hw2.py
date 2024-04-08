import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

audio_file = "multimedia_hw2.mp3"

# load the audio signal and its sample rate
signal, rate = librosa.load(audio_file)

'''Problem 1: Waveform'''
plt.figure(figsize=(10, 4))
librosa.display.waveshow(signal, sr=rate, alpha=0.5, color='b')
plt.title('Raw Waveform')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.savefig('./img/waveform.png')

'''Problem 2: Energy Contour'''
energy = librosa.feature.rms(y=signal)

plt.figure(figsize=(10, 4))
# librosa.display.waveshow(signal, sr=rate, alpha=0.5, color='b')  # Plot the waveform
plt.plot(librosa.times_like(energy), energy[0], color='r', label='Energy')  # Plot the energy contour
plt.title('Energy Contour')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend()
plt.tight_layout()
plt.savefig('./img/energy_contour.png')

'''Alternative method to calculate energy'''
frame_length = int(0.01 * rate)  # 10 ms
energy = np.array([sum(abs(signal[i:i+frame_length]**2)) for i in range(0, len(signal), frame_length)])
plt.figure(figsize=(10, 4))
plt.plot(librosa.times_like(energy), energy, color='r', label='Energy')
plt.title('Energy Contour (Alternative)')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend()
plt.tight_layout()
plt.savefig('./img/energy_contour_alternative.png')

'''Problem 3: Zero-crossing rate Contour'''
zcr = librosa.feature.zero_crossing_rate(signal)

plt.figure(figsize=(10, 4))
plt.plot(librosa.times_like(zcr), zcr[0], color='g', label='ZCR')
plt.title('Zero-crossing Rate Contour')
plt.xlabel('Time')
plt.ylabel('Zero-crossing Rate')
plt.legend()
plt.tight_layout()
plt.savefig('./img/zcr_contour.png')

'''Problem 4: End-point detection'''
# Compute short-term energy
frame_length = int(0.01 * rate)  # 10 ms
energy = np.array([sum(abs(signal[i:i+frame_length]**2)) for i in range(0, len(signal), frame_length)])

# Set a threshold for energy
energy_threshold = np.mean(energy) * 1.3  # Adjust this multiplier as needed

# Perform endpoint detection
speech_segments = []
segment_start = None
for i in range(len(energy)):
    if energy[i] > energy_threshold:
        if segment_start is None:
            segment_start = i
    elif segment_start is not None:
        speech_segments.append((segment_start, i))
        segment_start = None

# Plot the waveform and detected speech segments
plt.figure(figsize=(10, 4))
librosa.display.waveshow(signal, sr=rate, alpha=0.5, color='b')
for segment in speech_segments:
    plt.axvspan(segment[0] * frame_length / rate, segment[1] * frame_length / rate, color='r', alpha=0.3)
plt.title('Endpoint Detection')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.savefig('./img/endpoint_detection.png')

'''Problem 5: Pitch Contour'''
f0, voice_flag, voiced_probs = librosa.pyin(signal, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

plt.figure(figsize=(10, 4))
plt.plot(librosa.times_like(f0), f0, label='Pitch (Hz)', color='b')
plt.title('Pitch Contour')
plt.xlabel('Time')
plt.ylabel('Frequency (Hz)')
plt.legend()
plt.tight_layout()
plt.savefig('./img/pitch_contour.png')

'''Problem 6: Spectrogram'''
spectrogram = np.abs(librosa.stft(signal))
spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(spectrogram_db, sr=rate, x_axis='time', y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()
plt.show()
plt.savefig('./img/spectrogram.png')