import matplotlib as mpl
mpl.use('Agg')

# coding: utf-8
import os
import matplotlib.pyplot as plt
#mpl.use('Agg')

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
from neo.core import analogsignal
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

def iter_plot0(md):
    import seaborn as sns

    index, mdf1 = md
    weight_gain_factors = {0:1,1:3,2:9,3:15,4:20}
    k = weight_gain_factors[index]
    #print(len(mdf1.segments),'length of block')

    ass = mdf1.analogsignals[0]

    time_points = ass.times
    avg = np.mean(ass, axis=0)  # Average over signals of Segment

    plt.figure()
    plt.plot([i for i in range(0,len(avg))], avg)
    plt.title("Peak response in segment {0}".format(avg.max()))
    plt.savefig('prs.png')
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
    #v = mdf1.take_slice_of_analogsignalarray_by_unit()
    t_axis = np.arange(float(mdf1.t_start), float(mdf1.t_stop), dt)
    plt.figure()
    plt.clf()

    plt.figure()
    plt.clf()
    cleaned = []
    data = np.array(mdf1.analogsignals[0].as_array().T)
    print(data)
    for i,vm in enumerate(data):
        if np.max(vm) > 900.0 or np.min(vm) < - 900.0:
            pass
        else:
            plt.plot(ass.times,vm)#,label='neuron identifier '+str(i)))
            cleaned.append(vm)
            #vm = s#.as_array()[:,i]

    assert len(cleaned) < len(ass)

    print(len(cleaned))
    plt.title('neuron $V_{m}$')
    #plt.legend(loc="upper left")
    plt.savefig(str('weight_')+str(k)+'analogsignals'+'.png');
    plt.xlabel('$ms$')
    plt.ylabel('$mV$')

    plt.close()


        #pass

    plt.figure()
    plt.clf()
    plt.plot([i for i in range(0,len(vm_not_spiking[110]))],vm_not_spiking[110])
    plt.xlabel('$ms$')
    plt.ylabel('$mV$')
    plt.savefig(str('weight_')+str(k)+'eespecific_analogsignals'+'.png');
    plt.close()


    plt.figure()
    plt.clf()
    plt.plot([i for i in range(0,len(vm_not_spiking[110]))],vm_not_spiking[55])
    plt.xlabel('$ms$')
    plt.ylabel('$mV$')

    plt.savefig(str('weight_')+str(k)+'iispecific_analogsignals'+'.png');
    plt.close()

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
    cells = [i for i in range(0,len(cvs))]

    plt.clf()
    fig, axes = plt.subplots()
    axes.set_title('Coefficient of Variation Versus Neuron')
    axes.set_xlabel('Neuron number')
    axes.set_ylabel('CV estimate')
    mcv = np.mean(cvs)
    plt.scatter(cells,cvs)
    fig.tight_layout()
    plt.savefig(str('weight_')+str(k)+'cvs_mean_'+str(mcv)+'.png');
    plt.close()
    '''
    import pandas as pd
    d = {'coefficent_of_variation': cvs, 'cells': cells}
    df = pd.DataFrame(data=d)

    ax = sns.regplot(x='cells', y='coefficent_of_variation', data=df)#, fit_reg=False)
    plt.savefig(str('weight_')+str(k)+'cvs_regexp_'+str(mcv)+'.png');
    plt.close()
    '''

    spike_trains = []
    ass = mdf1.analogsignals[0]
    tstop = mdf1.t_stop
    #assert tstop == 2000
    tstop = 2000
    vm_spiking = []

    for spiketrain in mdf1.spiketrains:
        vm_spiking.append(mdf1.analogsignals[0][spiketrain.annotations['source_id']])
        y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']

        # argument edges is the time interval you want to be considered.
        pspikes = pyspike.SpikeTrain(spiketrain,edges=(0,tstop))
        spike_trains.append(pspikes)

    # plot the spike times

    #for (i, spike_train) in enumerate(spike_trains):
    #plt.scatter(spike_train, i*np.ones_like(spike_train), marker='|')

    plt.clf()
    for (i, spike_train) in enumerate(spike_trains):
        plt.scatter(spike_train, i*np.ones_like(spike_train), marker='.')
    #plt.axis('tight')
    plt.savefig(str('weight_')+str(k)+'raster_plot'+'.png');
    plt.close()

    #plt.clf()

    #for (i, spike_train) in enumerate(spike_trains):
    #    plt.plot(spike_train, i * np.ones_like(tspike_train, 'k.', markersize=2))
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
    plt.title("SPIKE-distance, T=0-2000")


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

def iter_plot1(md):
    index, mdf1 = md
    weight_gain_factors = {0:1,1:3,2:9,3:15,4:20}
    k = weight_gain_factors[index]
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

    import elephant
    from scipy.signal import periodogram
    #dt = 0.0025
    #frequencies, power = periodogram(ass,fs=1/dt)
    frequencies, power = elephant.spectral.welch_psd(ass)
    #mfreq = np.mean(frequencies)
    #import pdb; pdb.set_trace()
    #print(frequencies)
    mfreq = frequencies[np.where(power==np.max(power))[0][0]]
    import pickle
    with open(str(k)+'_'+str(mfreq)+'_'+'mfreq.p','wb') as f:
       pickle.dump(mfreq,f)

    import pickle
    f = open('9_25000.0 Hz_mfreq.p','rb')
    blah = pickle.load(f)
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
    #plt.imshow(coherance_matrix, interpolation='none',norm=cbar_kws)
    sns.heatmap(coherance_matrix)#,cbar_kws=cbar_kws)
    plt.title("Coherance Matrix")
    plt.savefig(str('Coherance_matrix_weight_')+str(k)+str('freq_')+str(mfreq)+'.png');
    plt.close()

    import numpy
    a = numpy.asarray(coherance_matrix)
    numpy.savetxt("coherance_matrix.csv", a, delimiter=",")
    mdf1 = None
    coh = None


iter_distances = natsorted(glob.glob('pickles/qi*.p'))
mdfloop = {}

for k,i in enumerate(iter_distances):
    with open(i, 'rb') as f:
        from neo.core import analogsignal
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

#for t in titems:
#    iter_plot1(t)

grid = db.from_sequence(titems,npartitions = 3)
_ = list(db.map(iter_plot0,grid).compute());

grid = db.from_sequence(titems,npartitions = 3)
_ = list(db.map(iter_plot1,grid).compute());

#_ = List(iter_plot0,titems))

import pca

from sciunit.utils import NotebookTools
NotebookTools.do_notebook('Distribution')

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
