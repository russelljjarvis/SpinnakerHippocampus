
import matplotlib
matplotlib.use('Agg')
from natsort import natsorted, ns
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
#!pip install ipyvolume
from mpl_toolkits.mplot3d import Axes3D
#import ipyvolume.pylab as p3

def report_mean_var(data):
    for i in range(data.shape[1]):
        column = data[:,i]
        print("Dimension %d has mean %.2g and variance %.3g" % \
              (i+1,column.mean(),column.var()))

def variance_explained(df,pca):
    #pca.fit(df.values)
    n_components = min(*df.shape)
    if pca.n_components:
        n_components = min(n_components,pca.n_components)
    for i in range(n_components):
        print("PC %d explains %.3g%% of the variance" % (i+1,100*pca.explained_variance_ratio_[i]))



def iter_pca(md):
    index, mdf1 = md
    ass = mdf1.analogsignals[0]
    lens = np.shape(ass.as_array()[:,1])[0]


    end_floor = np.floor(float(mdf1.t_stop))
    dt = float(mdf1.t_stop) % end_floor
    t_axis = np.arange(float(mdf1.t_start), float(mdf1.t_stop), dt)
    #t_axis = t_axis[0:-2]
    the_last_trace = mdf1.analogsignals[0].as_array()[:,121]


    plt.figure()
    plt.clf()
    cleaned = []
    data = np.array(mdf1.analogsignals[0].as_array().T)
    print(data)
    for i,vm in enumerate(data):
        if i<43:
            if np.max(vm) > 900.0 or np.min(vm) < - 900.0:
                pass
            else:
                plt.plot(ass.times,vm)#,label='neuron identifier '+str(i)))
                cleaned.append(vm)

    print(len(cleaned))
    plt.title('All Neurons $V_{m}$ versus Time')
    plt.xlabel('Time $(ms)$')
    plt.ylabel('Voltage $(mV)$')
    plt.savefig(str('excitatory_weight_')+str(index)+'_analogsignals'+'.png');
    plt.close()


    lens = len(cleaned)
    if lens >10:
        pca = PCA()
        data = np.array(cleaned)#np.array(mdf1.analogsignals[0].as_array().T)
        pca = PCA(n_components=3).fit(data)
        data_projected = np.dot(pca.components_,data.T).T

        print(pca.components_.shape_,'components_')
        plt.plot([i for i in pca.components_.shape_[0]],pca.components_.shape_[0])
        plt.title('PCA features to extract')
        plt.xlabel('Dimension directions')
        plt.ylabel('Weights for dimension after rotation')
        plt.savefig(str('excitatory_weight_')+str(index)+'_analogsignals'+'.png');
        plt.close()

        signals = np.dot(data.T,data_projected)
        signals = signals.T

        plt.figure()
        plt.clf()
        for i,s in enumerate(signals):
            vm = s
            if i < 3:
                plt.plot(ass.times,vm,label='PCA component: '+str(i))
            else:
                pass

        plt.title('$V_{m}$ Projections from PCA')
        plt.xlabel('$ms$')
        plt.ylabel('$mV$')
        plt.legend(loc="upper left")
        plt.savefig(str('projections_weight_value_')+str(md)+'excitatory_analogsignals'+'.png');
        plt.close()
        print(data_projected,'data')
        print(pca.components_,'component direction vectors')
        print(pca.components_,'components_')

iter_distances = natsorted(glob.glob('pickles/qi*.p'))
mdfloop = {}
for k,i in enumerate(iter_distances):
    with open(i, 'rb') as f:
        from neo.core import analogsignal
        mdfloop[k] = pickle.load(f)

for k,mdf1 in mdfloop.items():
    print(mdf1,k)

titems = [ (k,mdf1) for k,mdf1 in mdfloop.items() ]
_ = list(map(iter_pca,titems));
