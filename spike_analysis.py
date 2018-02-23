import pickle
import numpy as np
from neuronunit.tests.dynamics import ISITest, LocalVariationTest
import pyspike
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

import pyspike as spk
with open('pickles/membrane_dynamics_file.p', 'rb') as f:
  mdf1 = pickle.load(f)
  print(mdf1)

with open('pickles/membrane_dynamics_hippocampome_file.p', 'rb') as f:
  mdf2 = pickle.load(f)
  print(mdf2)

try:
    with open('membrane_dynamics_balanced_file.p', 'rb') as f:
       mdf3 = pickle.load(f)
       print(mdf3)

    with open('membrane_dynamics_file.p','rb') as f:
       mdf = pickle.load(f)

    with open('pickles/membrane_dynamics_balanced_file.p', 'rb') as f:
       mdf0 = pickle.load(f)
       print(mdf0)

except:
   pass



# first load the data, interval ending time = 4000, start=0 (default)
spike_trains_txt = spk.load_spike_trains_from_txt("PySpike_testdata.txt", 4000)

wrangled_trains = []
for spiketrain in mdf1.spiketrains:
    y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
    pspikes = pyspike.SpikeTrain(spiketrain,edges=(0,max(spiketrain)))
    wrangled_trains.append(pspikes)
    print(pspikes)

    """ Class representing spike trains for the PySpike Module.
    def __init__(self, spike_times, edges, is_sorted=True):
    Constructs the SpikeTrain.

    :param spike_times: ordered array of spike times.
    :param edges: The edges of the spike train. Given as a pair of floats
                  (T0, T1) or a single float T1, where then T0=0 is
                  assumed.
    :param is_sorted: If `False`, the spike times will sorted by `np.sort`.

    """


spike_trains = wrangled_trains
print(spike_trains)
#plt.figure()
isi_distance = spk.isi_distance_matrix(spike_trains)
plt.imshow(isi_distance, interpolation='none')
plt.title("ISI-distance")

#plt.figure()
spike_distance = spk.spike_distance_matrix(spike_trains, interval=(0, 1000))
plt.imshow(spike_distance, interpolation='none')
plt.title("SPIKE-distance, T=0-1000")

#plt.figure()
spike_sync = spk.spike_sync_matrix(spike_trains, interval=(2000, 4000))
plt.imshow(spike_sync, interpolation='none')
plt.title("SPIKE-Sync, T=2000-4000")

plt.savefig('the distances')

'''
T_max = 0.01
window = np.sin(np.pi*times/T_max)**2
signal *= [window]

nyquist = int(T_max/2)
frequencies = frequencies[:nyquist]
magnitude = magnitude[:nyquist]



df = 1/T_max
F_max = 1/dt
frequencies = np.arange(0,F_max,df)
frequencies[int(N/2):] -= F_max



from scipy.signal import periodogram
frequencies, power = periodogram(signal,fs=1/dt)
def plot_periodogram(frequencies,power):
    plt.figure(figsize=(10,4))
    plt.plot(frequencies,power)
    plt.xlabel('Frequency ($Hz$)')
    plt.ylabel('Power ($V^2/Hz$)') # Note that power is now
                                   # a normalized density
plot_periodogram(frequencies,power)
'''
