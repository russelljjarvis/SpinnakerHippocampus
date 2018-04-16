


# Data Analysis in Computation Neuroscience

Reproduction Steps:

We created a Dedicated Docker container, that had preinstalled `PyNN`, `NEURON`, `elephant`, `neo` and `R`.

Inside docker container we ran the files.

We programmatically downloaded a file `_hybrid_connectivity_matrix_20171103_092033.xlsx` from inside a file called `qi_ascoli.py` (named after the two authors, whose worked most informed our work).

The code for downloading those files (a snippet from qi_ascoli.py) is pasted below:
```
# Get some hippocampus connectivity data, based on a conversation with
# academic researchers on GH:
# https://github.com/Hippocampome-Org/GraphTheory/issues?q=is%3Aissue+is%3Aclosed
# scrape hippocamome connectivity data, that I intend to use to program neuromorphic hardware.
# conditionally get files if they don't exist.


path_xl = '_hybrid_connectivity_matrix_20171103_092033.xlsx'
if not os.path.exists(path_xl):
    os.system('wget https://github.com/Hippocampome-Org/GraphTheory/files/1657258/_hybrid_connectivity_matrix_20171103_092033.xlsx')
```

We estimated the rheobase current injection for the excitatory and inhibitory classes of cells using code from the _neuronunit_ model testing library which contains fast parallel solvers for extracting rheobase current injection values, by iteratively running an appropriate set of differential equations, that exhaustively search an appropriate range of current injection values.

This code was subequently commented out, but it persists in that form at the top of `qi_ascoli.py`. A snippet is below:
```
from neuronunit.models.reduced import ReducedModel
from neuronunit.optimization import get_neab
model = ReducedModel(get_neab.LEMS_MODEL_PATH,name=str('vanilla'),backend=('NEURON'))
attrs = {'a':0.02, 'b':0.2, 'c':-65+15*0.5, 'd':8-0.5**2 }
#from neuronunit.optimization.data_transport_container import DataTC
#dtc = DataTC()
from neuronunit.tests import fi
model.set_attrs(attrs)
from neuronunit.optimization import get_neab
rtp = get_neab.tests[0]
rheobase0 = rtp.generate_prediction(model)
rheobase = rheobase0
attrs2 = {'a':0.02+0.08*0.5, 'b':0.2-0.05*0.5, 'c':-65, 'd':2 }
#, i_offset=0)
#attrs = {'a':0.02, 'b':0.2, 'c':-65+15*0.5, 'd':8-0.5**2 }
#from neuronunit.optimization.data_transport_container import DataTC
#dtc = DataTC()
from neuronunit.tests import fi
model.set_attrs(attrs2)
from neuronunit.optimization import get_neab
rtp = get_neab.tests[0]
rheobase1 = rtp.generate_prediction(model)
print(rheobase0['value'],rheobase1['value'])
assert rheobase0['value'] != rheobase1['value']
```

Once rheobase current injections for the excitatory and inhibitory populations of cells where found.

We wired the cells, and constituent classes of sub-networks: Excitatory to Excitatory projections, Inhibitory to Excitatory, Excitatory to Inhibitory, and Inhibitory to Inhibitory, using the experimentally derived wiring rules implied by the spreadsheet: `_hybrid_connectivity_matrix_20171103_092033.xlsx`

PyNN code, for wiring the network, and recording the membrane potential was subsequently executed, inside the scope of a main method in qi_ascoli `sim_runner(wgf)`. Sim_runner takes a weight value as an argument, and it uniformly assigns this value to all synaptic weights in the simulated network model.

Another file, `forked.py` was used to invoke the python code in `qi_ascoli`, in an embarrassingly parallel manner (by using operating system level calls to the BASH command fork, making it possible to simulate very many networks of different weight values in a short amount of time.

Denise analysed the initial wiring map, in excel format (_hybrid_connectivity_matrix_20171103_092033.xlsx), and used it to extract indegree and outdegree distributions, for each cell in the network. Denise used excel to create vectors which describe indegree and outdegree of each neuron, as related to different anatomical regions in the hippocampus: 'DG', 'EC', 'CA1', 'CA2', and 'CA3'. She then used python to plot these region specific indegree and outdegree distributions per hippocampus sub-region. More information on Denises's workflow can be found in here jupyter notebook found at: https://github.com/russelljjarvis/DAnalysisCNeuro/blob/master/RichClub1-2.ipynb

Subsequently Daniel Petty's code for interactive network visualization using Shiny a module in R is launched. It acts on csv files defined in qi_ascoli. The initial wiring rules defined in `_hybrid_connectivity_matrix_20171103_092033.xlsx` used negative integers '-1', and '-2' to denote the presence of inhibitory synaptic connections, and '1' and '2' denoted excitatory connections.

We decomposed this connection matrix into 4 sub connection matrices that dealt with specific projections between the two populations, conventionally denoted by: 'EE', 'EI', 'II' and 'IE'. Such that '1' entry in the matrix denoted the presence of connection, and '0' denoted the absence of a connection. Values of '2', and '-2' in Ascoli's initial wiring map where used to represent putative connections, hypothesised connections, that have not been falsified yet. We upgraded putative connections to the status of confirmed connections, for the purposes of adding synaptic drive to neurons, that may otherwise suffer from sparse connectivity, making it easier to tip the neurons into a more realistic high conductance state.

We have tested the Dockerfile up to line 90, and we where able to confirm that this build is sufficient for launching both R, and python3 with PyNN, elephant and other dependencies, however we are unsure if running lines 91, and 92. Will flawlesly run the network visualization software. We are confident, that conceptually this approach to running all the software is correct.
https://github.com/russelljjarvis/DAnalysisCNeuro/blob/master/Dockerfile#L90-#L92
```
RUN R -e 'install.packages(c("rPython","shiny","igraph","visNetwork,"pracma,"stringr","chorddiag"))'
ENTRYPOINT R -e 'runApp()'
```
After `forked.py` runs parallel simulations that explore different synaptic weight values, the file `sa.py` (spike analysis) is called. `sa.py` is and analysis program which finds firing rates of cells, ISIs, Coefficients of Variation, Spike Distance Matrices, and many other network level feature analysis.

Spike analysis is called using `dask distributed's` parallel map function, to do a parallel execution of all of the pickle file data generated by running the simulation inside `forked.py`.

The graphs generated by `sa.py` (spike analysis) file where then loaded into a dedicated notebook `RichClubPresentation.ipynb` with code for interactively stepping through different values of synaptic weight, in a way that enabled us to interrogate the contribution of synaptic weight values on network dynamics.  
`










—

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
