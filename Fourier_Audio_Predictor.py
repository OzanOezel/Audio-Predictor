"""
This is an updated version of the Fourier-Based regressor created by Marco Gori.
In this version, the aim is to input a song that has some gaps, then using Fourier-Based Regression,
predict those gaps and reconstruct the audio file. This code requires the installation of the music
library "Librosa".
-Ozan Özel
"""

############################################################################
# Fourier-Based regression - Linear system with eigenvalues inside the
# unitary circle
# On-line gradient descent is used
# Plain LMS with fixed set of orthonormal sin/cos
############################################################################
#
# by Marco Gori 24th of November 2024
#


import numpy as np
import matplotlib.pyplot as plt
import sys
import math
from numpy import linalg as LA
import random
from random import *

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from prompt_toolkit.widgets import HorizontalLine

#
# Set up for plotting
#

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

def update_line(g, x, y):
    g.set_xdata(np.append(g.get_xdata(), x))
    g.set_ydata(np.append(g.get_ydata(), y))
    ax.relim()
    ax.autoscale_view(True,True,True)
    plt.draw()
    plt.pause(0.0001)


#prepare the  plot
ax = plt.gca()
ax.set_autoscale_on(True)
#label of axes
steps = np.linspace(-1, 2, 100)
plt.ylabel('output')
plt.xlabel('time')
target, = plt.plot([], [])
output, = plt.plot([], [])
weights, = plt.plot([], [])

#Loading audio file

sampling_rate = 22050

audio, sr = librosa.load('Audio_with_missing_segment_.m4a', sr=sampling_rate)

#Finding Gaps in the audio
a = np.zeros(len(audio)) #a marks where the audio has gaps. 1 for gap, 0 for else
for i in range(len(audio)):
    if audio[i] == 0:
        a[i] = 1

#Finding where the gaps start and end
moving_avg = np.convolve(a, np.ones(1000) / 1000, mode='same') #Using convolution because there may also be 0's
                                                               #within the gap and the aim is to cover the whole gap
                                                               #from start to finish.
regions = moving_avg > 0

starting_points = []
ending_points = []

for i in range(1, len(regions)):
    if regions[i] and not regions[i - 1]:  # Start of a region
        starting_points.append(i)
    elif not regions[i] and regions[i - 1]:  # End of a region
        ending_points.append(i - 1)

reconst_length = ending_points[0] - starting_points[0] #length of the missing gap

horizon = sr*2 #2 seconds before the break (sampling rate*2)
order=950
w_range = 1
T = 2300 # set the period
z = np.zeros(horizon)           # allocate/initialize all the outputs
y = np.zeros(horizon)           # allocate the target
x = np.zeros(order)             # allocate/initialize all the virtual inputs
w= w_range*np.random.rand(horizon,order)
omega = np.zeros(order//2)
#a = np.zeros((T,p))            # allocate/initialize all the model parameters
eta = 0.001
mu = 0.01

#Starting point for algorithm:
training_start = ending_points[0] - horizon

#
for i in range(order//2):
    omega[i] = 2*(np.pi/T)*i
#

for t in range(1,horizon-1):
    for i in range(order//2):
        x[2*i] = np.cos(t*omega[i])
        x[2*i+1] = np.sin(t*omega[i])
    z[t] = np.dot(w[t,],x)
    #print(audio[t])
    if (t<(horizon-reconst_length)): # Stop updating the weights just before the gap
        w[t+1,] = w[t,] - eta*(z[t] - audio[training_start+t])*x - mu*(w[t,]-w[t-1,]) #online gradient descent
    else:
        w[t+1,] = w[t,]
    #
    norm_w = LA.norm(w[t+1,])
    #print(norm_w)
    if t>(horizon-reconst_length-1000):
    #if t>horizon//2 - 1000:
    #if t>0:
        update_line(target, t, audio[training_start+t])
        update_line(output, t, z[t])
        update_line(weights, t, norm_w)
    #print(z)

plt.show()

# Reshape z so that only the part that coincides the gap is used
z = z[-reconst_length:]

#I realised that scaling is not necessary
"""
# Scaling the volume of the created signal into the original audio
audio_segment = audio[starting_points[0]-50:starting_points[0]]
volume = np.mean(np.abs(audio_segment))
z_max = np.mean(np.abs(z))  # max of the reconstructed signal

scaling_factor = volume / z_max
"""
scaling_factor = 1
z_scaled = z * scaling_factor

#Making sure there are no peaks in the audio that can harm the ears
for i in range(len(z_scaled)):
    if z_scaled[i] > 0.9:
        z_scaled[i] = 0.9
    elif z_scaled[i] < -0.9:
        z_scaled[i] = -0.9

#Saving the predicted signal z
output_file = 'predicted_signal.wav'
sf.write(output_file, z_scaled, sr)
print(f"Predicted signal saved to {output_file}")

#Plot of the created signal z
t = np.linspace(0, len(z_scaled) / sr, len(z_scaled))
plt.figure(figsize=(12, 6))
plt.plot(t, z_scaled, label='Predicted Signal', alpha=0.7, color='orange')
plt.title('Predicted Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()


# Insert the predicted signal into the gap in the audio
for i in range(reconst_length):
    audio[starting_points[0] + i] = z_scaled[i]

#Plot and save the reconstructed audio
t = np.linspace(0, len(audio) / sr, len(audio))
plt.figure(figsize=(12, 6))
plt.plot(t, audio, label='Reconstructed audio', alpha=0.7, color='orange')
plt.title('Reconstructed Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

#Save the reconstructed audio
output_file = 'reconstructed_audio.wav'
sf.write(output_file, audio, sr)
print(f"Reconstructed audio saved to {output_file}")
