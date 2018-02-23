import pickle
import numpy as np
from neuronunit.tests.dynamics import ISITest, LocalVariationTest

with open('membrane_dynamics_file.p','rb') as f:
   mdf = pickle.load(f)
try:
   with open('membrane_dynamics_balanced_file.p', 'wb') as f:
      mdf0 = pickle.load(f)
      print(mdf0)
except:
   pass

print(mdf)


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
