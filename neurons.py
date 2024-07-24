import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# CSV-DataFrame
df = pd.read_csv('neuronal_spikes.csv')

# The activity is the sum of all spikes.
activity = df.drop(columns=['time']).sum(axis=1)
spikes = df.drop(columns=['time']).values

A = activity.values
time = df['time'].values

print("Activity:", A)
print("Time:", time)

plt.figure(figsize=(12, 10))

# First subplot: Total activity over time
plt.subplot(211)
plt.plot(time, A, marker='o', linestyle='-', color='b')
plt.xlabel('Time')
plt.ylabel('Activity')
plt.title('Total Neuron Activity Over Time')
plt.grid(True)

# Second subplot: Spikes of each neuron over time
plt.subplot(212)
for time_index in range(spikes.shape[0]):
    active_neurons = np.where(spikes[time_index, :] == 1)[0]
    plt.scatter(np.full(active_neurons.shape, time_index), active_neurons, s=10)

plt.xlabel('Time')
plt.ylabel('Neurons')
plt.title('Neuronal Activity Over Time')
plt.grid(True)

plt.tight_layout()
plt.show()