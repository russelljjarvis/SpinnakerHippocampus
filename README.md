# DataAnalysisComputationNeuroscience

Analysis and Visualization of spike time variance in a basic neuromorphic simulation of hippocampus cells.

Introduction: As a contribution to my ongoing research objectives, I want to be able to create parts of an analysis and visualization pipeline that will allow me to difference model predictions (characterizing population wide, and individual neuron simulated spike times), with some experimentally derived observations of spike time variation and spike time synchrony in the rodent hippocampus.  This might sound difficult, however there are already many off the shelf algorithms for computing spike time variation and spike train synchrony:
http://elephant.readthedocs.io/en/latest/reference/statistics.html
https://github.com/mariomulansky/PySpike/tree/master/pyspike.git

Much of this work may involve good visualize of multivariate spike train distance, and within neuron coefficient of variation of Inter Spike Intervals. I am also open to considering different and less elaborate spike time based statistics and visualizations that I can use to meaningfully compare simulated model predictions to experimental observations.

For this project I have created a dedicated github repository: https://github.com/russelljjarvis/DanalysisCNeuro.git

The idea is to run and record from neuronal network simulations run on a neuromorphic hardware substrate which I access via: 
https://collab.humanbrainproject.eu/
Note also, if you decided to join this group project and you were interested in the coding the model, I can provide instructions for requesting access, or I could run your code via my collaboration portal.

The simulation mentioned herein is a reduced (reduced in degree of cell numbers and electro-dynamic properties) hippocampus neuronal network model that is currently in development, however preliminary outputs should be ready before the project work starts in earnest (next week).

I have already talked to several people in class about collaborating in a final project, if you are one of these people please feel free to discuss either committing to this proposal, rejecting it and working with other people, or to suggesting changes and improvements.

The goal is to test if connectivity information combined with simple cell dynamic models can set up a situation where a
teaching relationship can be measured in the dynamic activity between grid (teacher) and place (student) populations.

## Goals 
## 1 
Get an experimentally informed connection matrix from from http://hippocampome.org/netlist
Or an excell spreadsheet of similar origin.
## 2 
Mutate the excell document into a adjacency matrix to pandas df
## 3 
Search inside the data frame, create a filtered df/matrix using only entities from the Medial Entorhinal Cortex (MEC)rt
Done

## 3.75 
Create histograms that summarize the target cells.
# 
Ascertain whether an entity is exhitatory or inhibitory and apropriately substitute in simplified versions of Fast Inhibitory, or Izhiketich excitatory.

Use PyNN instead of Neuromorphic hardware, as anything developed in PyNN can be run here.

## 4 
Use a transfer entropy toolbox to find out if grid cell behavior predicts place cell behavior, merely due to effective connectivity.

## 5
If grid cell's train place cells there should be higher directed mutual information from Grid -> Place. 

There should be lower directed mutual information from Place -> Grid.

# Don't Do:

## 4 
Search Allen Brain and NeuroElectro for physiological properties that can be modelled using the smaller list of cell type entities limited to the MEC. 
Only interested in models that can be implemented in PyNN, on SpiNNaker
## 5 
Programatically push a job to the Human Brain Collaboration Portal using:
## 6
Authenticate on the BBP using:
from bbp_client.oidc.client import BBPOIDCClient
client = BBPOIDCClient.file_auth('path_to_yaml')

## 6
https://developer.humanbrainproject.eu/docs/projects/bbp-client/0.4.4/task_service.html

## 7
https://collab.humanbrainproject.eu/#/collab/5458/nav/42545
