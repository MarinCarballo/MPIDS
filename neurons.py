import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV file
data = pd.read_csv('neuronal_spikes.csv')

# Extract time and neuronal activity data
time = data['time'].values
neuronal_activity = data.drop(columns=['time']).values

# Plotting the activity of neurons over time
plt.figure(figsize=(10, 6))

# Iterate over each neuron and plot its activity
for neuron_index in range(neuronal_activity.shape[1]):
    spikes = np.where(neuronal_activity[:, neuron_index] == 1)[0]
    plt.scatter(spikes, np.full(spikes.shape, neuron_index), label=f'Neuron {neuron_index + 1}', s=10)

plt.xlabel('Time')
plt.ylabel('Neurons')
plt.title('Neuronal Activity Over Time')
plt.legend(loc='best',fontsize=12)
plt.show()
