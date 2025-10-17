import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf  # Library to save audio as .wav

sample_rate = 22050
# Load the audio file
audio, sr = librosa.load('reconstructed_audio.wav', sr=sample_rate)

# Compute the Short-Time Fourier Transform (STFT)
D = librosa.stft(audio, n_fft=2048, hop_length=512, win_length=2048)

# Get the magnitude and phase of the STFT
magnitude, phase = librosa.magphase(D)

# Visualize the STFT magnitude spectrogram
plt.figure(figsize=(12, 6))
librosa.display.specshow(
    librosa.amplitude_to_db(magnitude, ref=np.max),  # Convert magnitude to dB scale
    sr=sr,
    hop_length=512,
    x_axis='time',
    y_axis='log'  # Use a logarithmic frequency scale
)
plt.colorbar(format='%+2.0f dB')
plt.title('STFT Spectrogram of the Reconstructed Audio File')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.grid(True)
#plt.show()

# Combine magnitude and phase for reconstruction
reconstructed_stft = magnitude * phase

# Use librosa.istft to reconstruct the signal and ensure it matches the original length
reconstructed_signal = librosa.istft(reconstructed_stft, hop_length=512, win_length=2048, length=len(audio))

# Normalize the reconstructed signal to avoid clipping when saving
reconstructed_signal = reconstructed_signal / np.max(np.abs(reconstructed_signal))

# Save the reconstructed signal as a .wav file
output_file = 'stft.wav'
sf.write(output_file, reconstructed_signal, sr)
print(f"Reconstructed audio saved to {output_file}")

# Compare Original vs Reconstructed Signal (STFT version)
mse = np.mean((audio - reconstructed_signal) ** 2)
print(f"Mean Squared Error (MSE) between original and reconstructed signal: {mse:.6f}")

# Plot Original vs Reconstructed Signal
t = np.linspace(0, len(audio) / sr, len(audio))
plt.figure(figsize=(12, 6))
plt.plot(t, audio, label='Original Signal', alpha=0.7, color='blue')
plt.plot(t, reconstructed_signal, label='Reconstructed Signal (STFT)', alpha=0.7, color='orange')
plt.title('Original Signal vs Reconstructed Signal (STFT)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()


