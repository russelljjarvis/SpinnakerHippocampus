# DataAnalysisComputationNeuroscience

# Assignment 3 Data Analysis via Pandas

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
