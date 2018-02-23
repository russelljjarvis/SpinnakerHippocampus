
# coding: utf-8

# In[ ]:

import pyNN
#from pyNN.utility import get_simulator, init_logging, normalized_filename
sim = pyNN.neuron
class NCell(object):
    def __init__(self):
        #    self.cell_params = None
        # Cell parameters
        area     = 20000. # (µm²)
        tau_m    = 20.    # (ms)
        cm       = 1.     # (µF/cm²)
        g_leak   = 5e-5   # (S/cm²)
        benchmark = "CUBA"
        if benchmark == "COBA":
            E_leak   = -60.  # (mV)
        elif benchmark == "CUBA":
            E_leak   = -49.  # (mV)
        v_thresh = -50.   # (mV)
        v_reset  = -60.   # (mV)
        t_refrac = 5.     # (ms) (clamped at v_reset)
        v_mean   = -60.   # (mV) 'mean' membrane potential, for calculating CUBA weights
        tau_exc  = 5.     # (ms)
        tau_inh  = 10.    # (ms)

        # Synapse parameters
        if benchmark == "COBA":
            Gexc = 4.     # (nS)
            Ginh = 51.    # (nS)
        elif benchmark == "CUBA":
            Gexc = 0.27   # (nS) #Those weights should be similar to the COBA weights
            Ginh = 4.5    # (nS) # but the delpolarising drift should be taken into account
        Erev_exc = 0.     # (mV)
        Erev_inh = -80.   # (mV)

        ### what is the synaptic delay???

        # === Calculate derived parameters =============================================

        area  = area*1e-8                     # convert to cm²
        cm    = cm*area*1000                  # convert to nF
        Rm    = 1e-6/(g_leak*area)            # membrane resistance in MΩ
        assert tau_m == cm*Rm                 # just to check
        #n_exc = int(round((n*r_ei/(1+r_ei)))) # number of excitatory cells
        #n_inh = n - n_exc                     # number of inhibitory cells
        benchmark = "COBA"
        if benchmark == "COBA":
            celltype = sim.IF_cond_exp
            w_exc    = Gexc*2.5e-7              # We convert conductances to uS
            w_inh    = Ginh*1e-1
        elif benchmark == "CUBA":
            celltype = sim.IF_curr_exp
            w_exc = 2e-7*Gexc*(Erev_exc - v_mean)  # (nA) weight of excitatory synapses
            w_inh = 5e-1*Ginh*(Erev_inh - v_mean)  # (nA)
            assert w_exc > 0; assert w_inh < 0

        cell_params = {
            'tau_m'      : tau_m,    'tau_syn_E'  : tau_exc,  'tau_syn_I'  : tau_inh,
            'v_rest'     : E_leak,   'v_reset'    : v_reset,  'v_thresh'   : v_thresh,
            'cm'         : cm,       'tau_refrac' : t_refrac}

        if (benchmark == "COBA"):
            cell_params['e_rev_E'] = Erev_exc
            cell_params['e_rev_I'] = Erev_inh
        self.cell_params = cell_params
