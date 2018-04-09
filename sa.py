import matplotlib as mpl
mpl.use('Agg')

# coding: utf-8
import os
import matplotlib.pyplot as plt
mpl.use('Agg')

#plt.use('Agg')
import seaborn as sns
mpl.use('Agg')

os.system('pip install pyspike')
os.system('pip install natsort')

import pickle
import numpy as np
import pyspike

#from elephant import conv
#import natsort
from natsort import natsorted, ns

import elephant.conversion as conv
import pyspike as spk
import glob
from elephant.spectral import welch_cohere
import elephant
import elephant.conversion as conv
import neo as n
import quantities as pq

import quantities as pq
import quantities as pq

from elephant.statistics import cv
import matplotlib.pyplot as plt
import elephant
import numpy as np


import pyspike as spk
'''
with open('pickles/qi.p', 'rb') as f:
  mdf1 = pickle.load(f)
print(mdf1)
'''

def iter_plot(md):

    k, mdf1 = md
    ass = mdf1.analogsignals[0]
    vm_spiking = []
    vm_not_spiking = []

    spike_trains = []
    binary_trains = []
    max_spikes = 0

    cnt = 0
    for spiketrain in mdf1.spiketrains:
        #spiketrain = mdf1.spiketrains[index]
        y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
        # argument edges is the time interval you want to be considered.
        pspikes = pyspike.SpikeTrain(spiketrain,edges=(0,len(ass)))
        spike_trains.append(pspikes)
        if len(spiketrain) > max_spikes:
            max_spikes = len(spiketrain)

        if np.max(ass[spiketrain.annotations['source_id']]) > 0.0:
            vm_spiking.append(ass[spiketrain.annotations['source_id']])
        else:
            vm_not_spiking.append(ass[spiketrain.annotations['source_id']])
        cnt+= 1

    for spiketrain in mdf1.spiketrains:
        x = conv.BinnedSpikeTrain(spiketrain, binsize=1 * pq.ms, t_start=0 * pq.s)
        binary_trains.append(x)
    end_floor = np.floor(float(mdf1.t_stop))
    dt = float(mdf1.t_stop) % end_floor
    mdf1.t_start
    v = mdf1.take_slice_of_analogsignalarray_by_unit()
    t_axis = np.arange(float(mdf1.t_start), float(mdf1.t_stop), dt)
    mdf1.events
    smaller_list = []
    for a in ass:
        #for i in ass:
        if np.min(a) <-100.0 or np.max(a) > 100.0:
            break
        else:
            smaller_list.append(a)
    assert len(smaller_list) < len(ass)
    plt.figure()
    plt.clf()
    for s in smaller_list:
        plt.plot([i for i in range(0,len(s))],s)
    plt.savefig(str('weight_')+str(k)+'analogsignals'+'.png');plt.close()

    plt.figure()
    plt.clf()
    plt.plot([i for i in range(0,len(vm_not_spiking[110]))],vm_not_spiking[110])
    plt.savefig(str('weight_')+str(k)+'eespecific_analogsignals'+'.png');plt.close()


    plt.figure()
    plt.clf()
    plt.plot([i for i in range(0,len(vm_not_spiking[8]))],vm_not_spiking[55])
    plt.savefig(str('weight_')+str(k)+'iispecific_analogsignals'+'.png');plt.close()



    cvs = [0 for i in range(0,len(spike_trains))]
    cvsd = {}
    for i,j in enumerate(spike_trains):
        cva = cv(j)
        if np.isnan(cva) or cva == 0:
            cvs[i] = 0
            cvsd[i] = 0
        else:
            cvs[i] = cva
            cvsd[i] = cva

    import pickle
    with open(str('weight_')+str(k)+'coefficients_of_variation.p','wb') as f:
       pickle.dump([cvs,cvsd],f)


    cvs = [i for i in cvs if i!=0 ]
    x_axis = [i for i in range(0,len(cvs))]

    plt.clf()
    fig, axes = plt.subplots()
    axes.set_title('Coefficient of Variation Versus Neuron {}'.format(i))
    axes.set_xlabel('Neuron number axis')
    axes.set_ylabel('CV estimate axis')
    plt.scatter(x_axis,cvs)
    fig.tight_layout()
    plt.savefig(str('weight_')+str(k)+'cvs'+'.png');plt.close()

    '''
    plt.clf()
    fig, axes = plt.subplots()
    axes.set_title('ISI histogram over all neuronal outputs {}'.format(i))
    axes.set_xlabel('Spike interinterval ranges')
    axes.set_ylabel('Power in Inteval range')
    unested_sp = []
    for s in spike_trains:
        unested_sp.extend(s)
    import elephant
    isi_hist = elephant.statistics.isi(sorted(unested_sp))
    isi_hist = [i for i in isi_hist if i!=0 ]
    plt.scatter([i for i in range(0,len(isi_hist))],isi_hist)
    plt.savefig(str('weight_')+str(k)+'ISI_hist'+'.png');plt.close()

    plt.clf()
    fig, axes = plt.subplots()
    axes.set_title('ISI histogram over all neuronal outputs {}'.format(i))
    axes.set_xlabel('Spike interinterval ranges')
    axes.set_ylabel('Power in Inteval range')
    unested_sp = []
    max_spikes = 0
    for s in spike_trains:
        unested_sp.extend(s)
        if len(s) > max_spikes:
            max_spikes = len(s)

    plt.clf()

    isi_hist = elephant.statistics.isi(sorted(unested_sp))
    isi_hist = [i for i in isi_hist if i<=10 and i>0.15 ]
    plt.hist(isi_hist,bins=10)
    plt.savefig(str('weight_')+str(k)+'ISI_hist_bin_10'+'.png');plt.close()
    plt.close()
    '''
    spike_trains = []
    ass = mdf1.analogsignals[0]
    tstop = mdf1.t_stop
    vm_spiking = []
    for spiketrain in mdf1.spiketrains:
        vm_spiking.append(mdf1.analogsignals[0][spiketrain.annotations['source_id']])
        y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']

        # argument edges is the time interval you want to be considered.
        pspikes = pyspike.SpikeTrain(spiketrain,edges=(0,tstop))
        spike_trains.append(pspikes)

    # plot the spike times

    plt.clf()
    for (i, spike_train) in enumerate(spike_trains):
        plt.scatter(spike_train, i*np.ones_like(spike_train), marker='|')
    plt.axis('tight')
    plt.savefig(str('weight_')+str(k)+'raster_plot'+'.png');plt.close()

    plt.clf()

    #for (i, spike_train) in enumerate(spike_trains):
    #    plt.plot(t, i * np.ones_like(tspike_train, 'k.', markersize=2)
    #plt.axis('tight')
    #plt.savefig(str('weight_')+str(k)+'colourful_raster_plot'+'.png');plt.close()

    # profile of the first two spike trains
    f = spk.isi_profile(spike_trains, indices=[0, 1])
    x, y = f.get_plottable_data()


    plt.figure()
    plt.plot(x, np.abs(y), '--k', label="ISI-profile")
    print("ISI-distance: %.8f" % f.avrg())
    f = spk.spike_profile(spike_trains, indices=[0, 1])
    x, y = f.get_plottable_data()
    plt.plot(x, y, '-b', label="SPIKE-profile")
    print("SPIKE-distance: %.8f" % f.avrg())
    plt.legend(loc="upper left")
    plt.savefig(str('weight_')+str(k)+'ISI_distance_bivariate'+'.png');plt.close()



    plt.figure()

    f = spk.spike_sync_profile(spike_trains[0], spike_trains[1])
    x, y = f.get_plottable_data()
    plt.plot(x, y, '--ok', label="SPIKE-SYNC profile")
    print("Average:", f.avrg())


    f = spk.spike_profile(spike_trains[0], spike_trains[1])
    x, y = f.get_plottable_data()

    plt.plot(x, y, '-b', label="SPIKE-profile")

    plt.axis([0, 4000, -0.1, 1.1])
    plt.legend(loc="center right")
    plt.clf()
    plt.figure()

    plt.subplot(211)

    f = spk.spike_sync_profile(spike_trains)
    x, y = f.get_plottable_data()
    plt.plot(x, y, '-b', alpha=0.7, label="SPIKE-Sync profile")

    x1, y1 = f.get_plottable_data(averaging_window_size=50)
    plt.plot(x1, y1, '-k', lw=2.5, label="averaged SPIKE-Sync profile")

    plt.subplot(212)

    f_psth = spk.psth(spike_trains, bin_size=50.0)
    x, y = f_psth.get_plottable_data()
    plt.plot(x, y, '-k', alpha=1.0, label="PSTH")


    print("Average:", f.avrg())
    plt.savefig(str('weight_')+str(k)+'multivariate_PSTH'+'.png');plt.close()


    plt.clf()
    plt.figure()

    f_psth = spk.psth(spike_trains, bin_size=50.0)
    x, y = f_psth.get_plottable_data()
    plt.plot(x, y, '-k', alpha=1.0, label="PSTH")


    plt.savefig(str('weight_')+str(k)+'exclusively_PSTH'+'.png');plt.close()


    plt.figure()
    isi_distance = spk.isi_distance_matrix(spike_trains)
    plt.imshow(isi_distance, interpolation='none')
    plt.title("ISI-distance")
    plt.savefig(str('weight_')+str(k)+'ISI_distance'+'.png');plt.close()

    #plt.show()

    plt.figure()
    plt.clf()
    import seaborn as sns

    sns.set()
    sns.clustermap(isi_distance)#,vmin=-,vmax=1);

    plt.savefig(str('weight_')+str(k)+'cluster_isi_distance'+'.png');plt.close()



    plt.figure()
    spike_distance = spk.spike_distance_matrix(spike_trains, interval=(0, float(tstop)))


    import pickle
    with open('spike_distance_matrix.p','wb') as f:
       pickle.dump(spike_distance,f)

    plt.imshow(spike_distance, interpolation='none')
    plt.title("SPIKE-distance, T=0-1000")


    plt.savefig(str('weight_')+str(k)+'spike_distance_matrix'+'.png');plt.close()
    plt.figure()
    plt.clf()
    sns.set()
    sns.clustermap(spike_distance);
    plt.savefig(str('weight_')+str(k)+'cluster_spike_distance'+'.png');plt.close()


    plt.figure()
    spike_sync = spk.spike_sync_matrix(spike_trains, interval=(0, float(tstop)))
    plt.imshow(spike_sync, interpolation='none')

    import numpy
    a = numpy.asarray(spike_sync)
    numpy.savetxt("spike_sync_matrix.csv", a, delimiter=",")


    plt.figure()
    plt.clf()

    sns.clustermap(spike_sync);

    plt.savefig(str('weight_')+str(k)+'cluster_spike_sync_distance'+'.png');plt.close()

    import elephant
    from scipy.signal import periodogram
    #dt = 0.0025
    #frequencies, power = periodogram(ass,fs=1/dt)
    frequencies, power = elephant.spectral.welch_psd(ass)
    mfreq = np.mean(frequencies)
    print(frequencies)
    import pickle
    with open(str(k)+'mfreq.p','rb') as f:
       pickle.dump(f,mfreq)

    def plot_periodogram(frequencies,power):
        plt.figure(figsize=(10,4))
        plt.plot(frequencies,power)
        plt.xlabel('Frequency ($Hz$)')
        plt.ylabel('Power ($V^2/Hz$)') # Note that power is now
                                       # a normalized density
        plt.savefig(str('weight_')+str(k)+'cluster_spike_sync_distance'+'.png');plt.close()

    lens = np.shape(ass.as_array())[1]
    coherance_matrix = np.zeros(shape=(lens,lens), dtype=float)
    for i in range(0,lens):
        for j in range(0,lens):
            if i!=j:
                x = ass.as_array()[i]
                y = ass.as_array()[j]
                coh = welch_cohere(x,y)
                if np.mean(coh) != 0:
                    coherance_matrix[i,j] = np.mean(coh)
    plt.figure()
    plt.clf()
    from matplotlib.colors import LogNorm
    plt.imshow(coherance_matrix, interpolation='none',norm=LogNorm())
    plt.title("Coherance matrix")
    plt.savefig(str('weight_')+str(k)+str(mfreq)+'.png');plt.close()
    mdf1 = None
    coh = None
iter_distances = natsorted(glob.glob('pickles/qi*.p'))
mdfloop = {}
weight_gain_factors = {1:None,3:None,9:None,15:None,20:None}

for k,i in enumerate(iter_distances):
    with open(i, 'rb') as f:
      mdfloop[k] = pickle.load(f)

for k,mdf1 in mdfloop.items():
    print(mdf1,k)

  
titems = [ (k,mdf1) for k,mdf1 in mdfloop.items() ]
print(titems)
print(iter_distances)
#print(titems)
import dask.bag as db
#for i in titems:
#    _ = iter_plot(i)

import pdb
print(titems,'titems')

#pdb.set_trace()
#    k, mdf1 = md
grid = db.from_sequence(titems,npartitions = 4)
_ = list(db.map(iter_plot,grid).compute());



import pickle
with open('bool_matrix.p','rb') as f:
   m = pickle.load(f)
print(type(m))
print(m)
import networkx as nx
G = nx.DiGraph(m)
in_degree = G.in_degree()
top_in = sorted(([ (v,k) for k, v in in_degree.items() ]))
in_hub = top_in[-1][1]
out_degree = G.out_degree()
top_out = sorted(([ (v,k) for k, v in out_degree.items() ]))
out_hub = top_out[-1][1]
sns.clustermap(m);
plt.figure()
plt.clf()
plt.imshow(m, interpolation='none')
plt.title("connection matrix")

mean_out = np.mean(list(out_degree.values()))
mean_in = np.mean(list(in_degree.values()))

mean_conns = int(np.floor(mean_in + mean_out)/2)

k = 2 # number of neighbouring nodes to wire.
p = 0.25 # probability of instead wiring to a random long range destination.
ne = np.shape(m)[0]# size of small world network
print(ne)
swe = nx.watts_strogatz_graph(ne,mean_conns,0.25)

#import networkx as nx
Gswe = nx.DiGraph(swe)
in_degree = Gswe.in_degree()
top_in = sorted(([ (v,k) for k, v in in_degree.items() ]))
in_hub = top_in[-1][1]
out_degree = Gswe.out_degree()
top_out = sorted(([ (v,k) for k, v in out_degree.items() ]))
out_hub = top_out[-1][1]


lswe = Gswe.adjacency_list()
excit_sm = np.zeros(shape=(ne,ne), dtype=bool)
#inhib_sm = np.zeros(shape=(ne,ne), dtype=bool)

for i,j in enumerate(lswe):
    for k in j:
        excit_sm[i,k] = int(1)

plt.figure()
plt.clf()
plt.imshow(excit_sm, interpolation='none')
plt.title("Small World connection matrix")
plt.show()

import numpy
a = numpy.asarray(excit_sm)
numpy.savetxt("small_world.csv", a, delimiter=",")
