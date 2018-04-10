import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
#!pip install ipyvolume
from mpl_toolkits.mplot3d import Axes3D
#import ipyvolume.pylab as p3
import pickle
with open('pickles/qi3.p', 'rb') as f:
  mdf1 = pickle.load(f)
#print(mdf1)
ass = mdf1.analogsignals[0]
lens = np.shape(ass.as_array()[:,1])[0]
pca = PCA()
data = np.array(mdf1.analogsignals[0].as_array().T)
pca = PCA(n_components=lens).fit(data)
data_projected = np.dot(pca.components_,data.T).T


end_floor = np.floor(float(mdf1.t_stop))
dt = float(mdf1.t_stop) % end_floor
t_axis = np.arange(float(mdf1.t_start), float(mdf1.t_stop), dt)
the_last_trace = mdf1.analogsignals[0].as_array()[:,121]



print(data_projected,'data')
print(pca.components_,'component direction vectors')
def report_mean_var(data):
    for i in range(data.shape[1]):
        column = data[:,i]
        print("Dimension %d has mean %.2g and variance %.3g" % \
              (i+1,column.mean(),column.var()))
#name='word_complexity'
#summarize(data_rotated,name)
#report_mean_var(data_rotated)

def variance_explained(df,pca):
    #pca.fit(df.values)
    n_components = min(*df.shape)
    if pca.n_components:
        n_components = min(n_components,pca.n_components)
    for i in range(n_components):
        print("PC %d explains %.3g%% of the variance" % (i+1,100*pca.explained_variance_ratio_[i]))

variance_explained(pd.DataFrame(data_projected),pca)

plt.figure()
plt.clf()
for i,s in enumerate(data_projected):
    vm = s.as_array()[:,i]
    plt.plot([i for i in range(0,vm)],vm)
plt.savefig(str('rotated_')+'analogsignals'+'.png');
plt.close()


def annotate_scatter(ax,df_transformed,df):
    for i, txt in enumerate(df.index):
        x_loc = df_transformed['PC 1'].iloc[i]
        y_loc = df_transformed['PC 2'].iloc[i]
        ax.annotate(txt, (x_loc,y_loc), fontsize=9)
    #for i, text in enumerate(df.index):
    #    ax.text(df['PC 1'].iloc[i],df['PC 2'].iloc[i], text)

def plot_transformed_data(pca,df_transformed,df,figsize=None):
    plt.clf()
    #pca.fit(df.values)
    n_components = min(*df.shape)
    if pca.n_components:
        n_components = min(n_components,pca.n_components)
    pca_df = pd.DataFrame(pca.transform(df.values),
                      index=df.index,
                      columns=['PC %d' % (i+1) for i in range(n_components)])
    ax = pca_df.plot.scatter('PC 1','PC 2',figsize=figsize)
    #ax = mds_df.plot.scatter(x='PC 1',y='PC 2',figsize=(12,12))
    #annotate_scatter(ax,mds_df)
    annotate_scatter(ax,pca_df,df_transformed)
    plt.savefig('pca_wave_transformed.png')
    plt.close()

#plot_transformed_data(pca,pd.DataFrame(data),pd.DataFrame(data_rotated))

def plot3d(df):
    plt.clf()
    data = df.values
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*data[:,0:3].T)
    minn,maxx = data.min(),data.max()
    ax.set_xlim(minn,maxx)
    ax.set_ylim(minn,maxx)
    ax.set_zlim(minn,maxx)
    ax.set_xlabel(df.columns[0],labelpad=10)
    ax.set_ylabel(df.columns[1],labelpad=10)
    ax.set_zlabel(df.columns[2],labelpad=10)
    ax.dist = 12
    plt.tight_layout()
    plt.savefig('pca_scatter.png')

    plt.close()
#plot3d(pd.DataFrame(data_rotated))
#plt.close()

print(pca.components_,'components_')
