import pyNN.neuron as sim
nproc = sim.num_processes()
node = sim.rank()
print(nproc)
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
#matplotlib.use('Agg')
mpl.rcParams.update({'font.size':16})

#import mpi4py
threads  = sim.rank()
rngseed  = 98765
parallel_safe = True
extra = {'threads' : threads}
import os
import pandas as pd
import sys
import numpy as np
from pyNN.neuron import STDPMechanism
import copy
from pyNN.random import RandomDistribution, NumpyRNG
import pyNN.neuron as neuron
from pyNN.neuron import h
from pyNN.neuron import StandardCellType, ParameterSpace
from pyNN.random import RandomDistribution, NumpyRNG
from pyNN.neuron import STDPMechanism, SpikePairRule, AdditiveWeightDependence, FromListConnector, TsodyksMarkramSynapse
from pyNN.neuron import Projection, OneToOneConnector
from numpy import arange
import pyNN
from pyNN.utility import get_simulator, init_logging, normalized_filename
import random
import socket
#from neuronunit.optimization import get_neab
import networkx as nx
sim = pyNN.neuron

installs = ['bbp_client','neuron','mpi4py','xlrd','pyNN','seaborn','lazyarray','neo','neuron','brian2']
def install_deps(i):
  '''
  Hack in dependencies into to sys.path
  '''
  import os
  if i not in sys.path:
    os.system('pip install '+str(i))
'''
System admin:
_ = list(map(install_deps,installs))
import os
#Compile NEUORN mod files.
temp = os.getcwd()
os.chdir('/opt/conda/lib/python3.5/site-packages/pyNN/neuron/nmodl')
get_ipython().system('nrnivmodl')
os.chdir(temp)
# Get some hippocampus connectivity data, based on a conversation with
# academic researchers on GH:
# https://github.com/Hippocampome-Org/GraphTheory/issues?q=is%3Aissue+is%3Aclosed
# scrape hippocamome connectivity data, that I intend to use to program neuromorphic hardware.
# conditionally get files if they don't exist.
'''

path_xl = '_hybrid_connectivity_matrix_20171103_092033.xlsx'
if not os.path.exists(path_xl):
    os.system('wget https://github.com/Hippocampome-Org/GraphTheory/files/1657258/_hybrid_connectivity_matrix_20171103_092033.xlsx')

xl = pd.ExcelFile(path_xl)
dfEE = xl.parse()
dfEE.loc[0].keys()
dfm = dfEE.as_matrix()

rcls = dfm[:,:1] # real cell labels.
rcls = rcls[1:]
rcls = { k:v for k,v in enumerate(rcls) } # real cell labels, cast to dictionary
import pickle
with open('cell_names.p','wb') as f:
    pickle.dump(rcls,f)
import pandas as pd
pd.DataFrame(rcls).to_csv('cell_names.csv', index=False)

filtered = dfm[:,3:]
filtered = filtered[1:]
rng = NumpyRNG(seed=64754)
delay_distr = RandomDistribution('normal', [45, 1e-1], rng=rng)

index_exc = [ i for i,d in enumerate(dfm) if '+' in d[0] ]
index_inh = [ i for i,d in enumerate(dfm) if '-' in d[0] ]

EElist = []
IIlist = []
EIlist = []
IElist = []

for i,j in enumerate(filtered):
  for k,xaxis in enumerate(j):
    if xaxis==1 or xaxis ==2:
      source = i
      target = k
      delay = delay_distr.next()
      weight = 11.0
      if target in index_inh:
          EIlist.append((source,target,delay,weight))
      else:
          EElist.append((source,target,delay,weight))

    if xaxis==-1 or xaxis ==-2:
      source = i
      target = k
      delay = delay_distr.next()
      weight = 11.0
      if target in index_exc:
          IElist.append((source,target,delay,weight))
      else:
          IIlist.append((source,target,delay,weight))

internal_conn_ee = sim.FromListConnector(EElist)
internal_conn_ii = sim.FromListConnector(IIlist)
internal_conn_ei = sim.FromListConnector(EIlist)
internal_conn_ie = sim.FromListConnector(IElist)

ml = len(filtered[1])+1
pre_exc = []
post_exc = []
pre_inh = []
post_inh = []


rng = NumpyRNG(seed=64754)
delay_distr = RandomDistribution('normal', [45, 1e-1], rng=rng)


plot_EE = np.zeros(shape=(ml,ml), dtype=bool)
plot_II = np.zeros(shape=(ml,ml), dtype=bool)


plot_EI = np.zeros(shape=(ml,ml), dtype=bool)
plot_IE = np.zeros(shape=(ml,ml), dtype=bool)

for i in EElist:
    plot_EE[i[0],i[1]] = int(0)
    if i[0]!=i[1]: # exclude self connections
        plot_EE[i[0],i[1]] = int(1)
        pre_exc.append(i[0])
        post_exc.append(i[1])

assert len(pre_exc) == len(post_exc)
for i in IIlist:
    plot_II[i[0],i[1]] = int(0)
    if i[0]!=i[1]:
        plot_II[i[0],i[1]] = int(1)
        pre_inh.append(i[0])
        post_inh.append(i[1])

for i in IElist:
    plot_IE[i[0],i[1]] = int(0)
    if i[0]!=i[1]: # exclude self connections
        plot_IE[i[0],i[1]] = int(1)
        pre_inh.append(i[0])
        post_inh.append(i[1])

for i in EIlist:
    plot_EI[i[0],i[1]] = int(0)
    if i[0]!=i[1]:
        plot_EI[i[0],i[1]] = int(1)
        pre_exc.append(i[0])
        post_exc.append(i[1])

plot_excit = plot_EI + plot_EE
plot_inhib = plot_IE + plot_II

assert len(pre_inh) == len(post_inh)

num_exc = [ i for i,e in enumerate(plot_excit) if sum(e) > 0 ]
num_inh = [ y for y,i in enumerate(plot_inhib) if sum(i) > 0 ]

# the network is dominated by inhibitory neurons, which is unusual for modellers.
assert num_inh > num_exc
#assert sum([i for i in filtered]) < 0

assert len(num_exc) < ml
assert len(num_inh) < ml

import pickle
with open('connections.p','wb') as f:
   pickle.dump([post_inh,pre_inh,pre_exc,post_exc],f)


# # Plot all the Projection pairs as a connection matrix (Excitatory and Inhibitory Connections)

import pickle
with open('graph_inhib.p','wb') as f:
   pickle.dump(plot_inhib,f)


import pickle
with open('graph_excit.p','wb') as f:
   pickle.dump(plot_excit,f)


#sns.pairplot(df, hue="species")
from scipy.sparse import coo_matrix
m = np.matrix(filtered[1:])

bool_matrix = np.add(plot_excit,plot_inhib)
with open('bool_matrix.p','wb') as f:
   pickle.dump(bool_matrix,f)

if not isinstance(m, coo_matrix):
    m = coo_matrix(m)


G = nx.DiGraph(plot_excit)
in_degree = G.in_degree()
top_in = sorted(([ (v,k) for k, v in in_degree.items() ]))
in_hub = top_in[-1][1]
out_degree = G.out_degree()
top_out = sorted(([ (v,k) for k, v in out_degree.items() ]))
out_hub = top_out[-1][1]



mean_out = np.mean(list(out_degree.values()))
mean_in = np.mean(list(in_degree.values()))

mean_conns = int(mean_in + mean_out/2)

k = 2 # number of neighbouig nodes to wire.
p = 0.25 # probability of instead wiring to a random long range destination.
ne = len(plot_excit)# size of small world network
small_world_ring_excit = nx.watts_strogatz_graph(ne,mean_conns,0.25)



k = 2 # number of neighbouring nodes to wire.
p = 0.25 # probability of instead wiring to a random long range destination.
ni = len(plot_inhib)# size of small world network
small_world_ring_inhib   = nx.watts_strogatz_graph(ni,mean_conns,0.25)

first_set = set(pre_exc)
second_set = set(post_exc)

assert len(set(pre_exc)) < len(pre_exc)
print(len(set(pre_exc)),len(pre_exc))

nproc = sim.num_processes()
nproc = 8
host_name = socket.gethostname()
node_id = sim.setup(timestep=0.01, min_delay=1.0, **extra)
print("Host #%d is on %s" % (node_id + 1, host_name))
print("%s Initialising the simulator with %d thread(s)..." % (node_id, extra['threads']))


#assert len(num_exc) < ml
#assert len(num_inh) < ml
pop_size = len(num_exc)+len(num_inh)

pop_exc =  sim.Population(len(num_exc), sim.Izhikevich(a=0.02, b=0.2, c=-65, d=8, i_offset=0))
pop_inh = sim.Population(len(num_inh), sim.Izhikevich(a=0.02, b=0.25, c=-65, d=2, i_offset=0))

num_exc = [ i for i,e in enumerate(plot_excit) if sum(e) > 0 ]
num_inh = [ y for y,i in enumerate(plot_inhib) if sum(i) > 0 ]

#import pdb; pdb.set_trace()
all_cells = pop_exc + pop_inh
#assert len(all_cells) == (len(pop_exc) + len(pop_inh))

pop_pre_exc = all_cells[list(set(pre_exc))]
pop_post_exc = all_cells[list(set(post_exc))]
pop_pre_inh = all_cells[list(set(pre_inh))]
pop_post_inh =  all_cells[list(set(post_inh))]
print(pop_pre_exc)

#assert len(pop_pre_exc) !=0
#assert len(pop_pre_inh) !=0



for pe in pop_exc:
    r = random.uniform(0.0, 1.0)
    pe.set_parameters(a=0.02, b=0.2, c=-65+15*r, d=8-r**2, i_offset=0)

for pe in pop_inh:
    r = random.uniform(0.0, 1.0)
    pe.set_parameters(a=0.02+0.08*r, b=0.2-0.05*r, c=-65, d= 2, i_offset=0)


num_exc = [ y for y,e in enumerate(plot_excit) if sum(e) > 0 ]
num_inh = [ y for y,i in enumerate(plot_inhib) if sum(i) > 0 ]


NEXC = len(num_exc)
NINH = len(num_inh)

#What I need to do.
# I need to create 4 new matrices.
# one for ee,
# one for ie,
# one for ii,
# one for ie
ee = np.delete(plot_excit,num_inh,0)
ee = np.delete(ee,num_inh,1)
#import pdb.set_trace()
#np.shape(ee)
#ei = np.delete(plot_excit,num_exc,0)
ei = np.delete(plot_excit,num_exc,1)
assert ei.any() == True

ii = np.delete(plot_inhib,num_exc,0)
ii = np.delete(ii,num_exc,1)

ie = np.delete(plot_inhib,num_inh,1)
assert ie.any() == True

#ie = np.delete(ie,num_inh,1)
'''
ie_sources = []
ei_targets = []
ei_sources = []
ie_targets = []


for i,j in enumerate(ei):
    ei_sources.append(i)
    for k in j:
        print(j)
        ei_targets.append(k)


for i,j in enumerate(ie):
    ie_sources.append(i)
    print(j)
    for k in j:
        ie_targets.append(k)


all_exc = all_cells[num_exc]
all_inh = all_cells[num_inh]
ie_sources = all_cells[ie_sources]
print(ie_sources)
ei_sources = all_cells[ei_sources]
print(ei_sources)
ie_targets = all_cells[ie_targets]
print(ie_sources)
ei_targets = all_cells[ei_targets]
print(ei_sources)
'''

'''
internal_conn_ee = sim.ArrayConnector(ee)
internal_conn_ii = sim.ArrayConnector(ii)
internal_conn_ie = sim.ArrayConnector(ie)
internal_conn_ei = sim.ArrayConnector(ie)
'''
all_conns = plot_excit + plot_inhib

all_exc = all_cells[num_exc]
all_inh = all_cells[num_inh]

print(len(all_cells))
rng = NumpyRNG(seed=64754)

print(EElist)
ees = all_cells[[e[0] for e in EElist]]
eet = all_cells[[e[1] for e in EElist]]

exc_distr = RandomDistribution('normal', [4.125, 10e-1], rng=rng)
exc_syn = sim.StaticSynapse(weight=exc_distr, delay=delay_distr)
prj_exc_exc = sim.Projection(ees, eet, internal_conn_ee, exc_syn, receptor_type='excitatory')
#import pdb; pdb.set_trace()


inh_distr = RandomDistribution('normal', [5, 2.1e-4], rng=rng)
inh_syn = sim.StaticSynapse(weight=inh_distr, delay=delay_distr)

iis = all_cells[[e[0] for e in IIlist]]
iit = all_cells[[e[1] for e in IIlist]]

rng = NumpyRNG(seed=64754)
delay_distr = RandomDistribution('normal', [50, 100e-3], rng=rng)
prj_inh_inh = sim.Projection(all_inh, all_inh, internal_conn_ii, inh_syn, receptor_type='inhibitory')

prj_inh_exc = sim.Projection(all_inh, all_exc, internal_conn_ie, exc_syn, receptor_type='excitatory')
inh_distr = RandomDistribution('normal', [5, 2.1e-4], rng=rng)
inh_syn = sim.StaticSynapse(weight=inh_distr, delay=delay_distr)

rng = NumpyRNG(seed=64754)
delay_distr = RandomDistribution('normal', [50, 100e-3], rng=rng)
prj_exc_inh = sim.Projection(ei_sources, ei_targets, internal_conn_ei, inh_syn, receptor_type='inhibitory')


#import pdb; pdb.set_trace()


'''
Old
prj_exc_exc = sim.Projection(all_exc, all_exc, internal_conn_e, exc_syn, receptor_type='excitatory')
inh_distr = RandomDistribution('normal', [5, 2.1e-4], rng=rng)
inh_syn = sim.StaticSynapse(weight=inh_distr, delay=delay_distr)
rng = NumpyRNG(seed=64754)
delay_distr = RandomDistribution('normal', [50, 100e-3], rng=rng)
prj_inh_inh = sim.Projection(all_inh, all_inh, internal_conn_i, inh_syn, receptor_type='inhibitory')
'''
# Variation in propogation delays are very important for self sustaininig network activity.
# more so in point neurons which don't have internal propogation times.

stdp = STDPMechanism(
          weight=4.0, #0.02,  # this is the initial value of the weight
          delay="0.2 + 0.01*d",
          timing_dependence=SpikePairRule(tau_plus=20.0, tau_minus=20.0,
                                          A_plus=0.01, A_minus=0.012),
          weight_dependence=AdditiveWeightDependence(w_min=0.01, w_max=10.0))



exc_targets = all_cells[pre_exc]
exc_srcs = all_cells[post_exc]
inh_srcs = all_cells[pre_inh] # = []
inh_targets = all_cells[post_inh] # = []
exc_cells = all_cells[index_exc]
inh_cells = all_cells[index_inh]

top_3_hubs = all_cells[top_out[-3:][1]]

ext_stim = sim.Population(len(all_cells), sim.SpikeSourcePoisson(rate=7.5, duration=6000.0), label="expoisson")
rconn = 0.9
ext_conn = sim.FixedProbabilityConnector(rconn)
ext_syn = sim.StaticSynapse(weight=5.925)
connections = {}
connections['ext'] = sim.Projection(ext_stim, all_cells, ext_conn, ext_syn, receptor_type='excitatory')

exc_spike_times = [
    6000+250,
    6000+500,
    6000+520,
    6000+540,
    6000+1250,
]

inh_spike_times = [
    6000+750,
    6000+1000,
    6000+1020,
    6000+1040,
    6000+1250,
]

stimulus_exc = sim.Population(1, sim.SpikeSourceArray, {
    'spike_times': exc_spike_times})
stimulus_inh = sim.Population(1, sim.SpikeSourceArray, {
    'spike_times': inh_spike_times})


connector = sim.OneToOneConnector()
ext_syn = sim.StaticSynapse(weight=5.925)

projections = [
    sim.Projection(stimulus_exc, top_3_hubs, connector, ext_syn, receptor_type='excitatory'),
    sim.Projection(stimulus_inh, top_3_hubs, connector, ext_syn, receptor_type='inhibitory'),
]


##
# Setup and run a simulation. Note there is no current injection into the neuron.
# All cells in the network are in a quiescent state, so its not a surprise that xthere are no spikes
##

neurons = all_cells
sim = pyNN.neuron
arange = np.arange
import re

neurons.record(['v','spikes'])  # , 'u'])
neurons.initialize(v=-65.0, u=-14.0)


# === Run the simulation =====================================================

sim.run(20000.0)

data = neurons.get_data().segments[0]
with open('pickles/qi.p', 'wb') as f:
    pickle.dump(data,f)
'''
from pyNN.utility.plotting import Figure, Panel, comparison_plot, plot_spiketrains
data = neurons.get_data().segments[0]
v = data.filter(name="v")
for i in v:
  Figure(
    Panel(i, ylabel="Membrane potential (mV)", xticks=True,
          xlabel="Time (ms)", yticks=True),
    #Panel(u, ylabel="u variable (units?)"),
    annotations="Simulated with"
  )
#Figure.savefig('voltage_time.png')

import pickle

#data = neurons.get_data().segments[0]
v0 =  neurons[(0,)].get_data().segments[0].filter(name="v")[0]
v1 =  neurons[(1,)].get_data().segments[0].filter(name="v")[0]
#plt.clf()
plt.plot(v0,range(0,len(v0)))
plt.plot(v1,range(0,len(v0)))
plt.show()

def plot_spiketrains(segment):
  """
  Plots the spikes of all the cells in the given segments
  """
  for spiketrain in segment.spiketrains:
      print(spiketrain)
      y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
      plt.plot(spiketrain, y, 'b')
      plt.ylabel('Neuron number')
      plt.xlabel('Spikes')
  plt.savefig('raster_plot.png')
spikes = neurons.get_data("spikes").segments[0]
data = neurons.get_data().segments[0]
plot_spiketrains(data)


Figure(
    Panel(v0, ylabel="Membrane potential (mV)", xticks=True,
          xlabel="Time (ms)", yticks=True),
    #Panel(u, ylabel="u variable (units?)"),
    annotations="Simulated with"
)

Figure(
    Panel(v1, ylabel="Membrane potential (mV)", xticks=True,
          xlabel="Time (ms)", yticks=True),
    #Panel(u, ylabel="u variable (units?)"),
    annotations="Simulated with"
)
'''
