'''
PlotLhoNb_logbinned.py

Copied from PlotLhoNb.py (authored by Georgia Mansell and Aaron Buikema)
Logbins the noisebudget to make it a reasonable size.

Craig Cahillane
Oct 1, 2019
'''

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
#plt.ion()
from scipy.io import loadmat
from plot_helpers import *
#from dataUtils import ReSamplingMatrixNonUniform

NB_LHO_contents = loadmat('LHO_NB_data_o3a.mat',squeeze_me=True, struct_as_record=False)
nb = NB_LHO_contents['NB']
#nb_darm_betterTime = loadmat('LHO_NB_data_darmOnly.mat',squeeze_me=True, struct_as_record=False)

mpl.rc('figure', figsize=(14, 9))

mpl.rcParams.update({'text.usetex': True,
                     'font.family': 'serif',
                     # 'font.serif': 'Georgia',
                     # 'mathtext.fontset': 'cm',
                     'lines.linewidth': 2.5,
                     'lines.markersize': 3,
                     'font.size': 22,
                     'xtick.labelsize': 'large',
                     'ytick.labelsize': 'large',
                     'legend.fancybox': True,
                     'legend.fontsize': 18,
                     'legend.framealpha': 0.9,
                     'legend.handletextpad': 0.5,
                     'legend.labelspacing': 0.2,
                     'legend.loc': 'best',
                     'legend.columnspacing': 2,
                     'savefig.dpi': 80,
                     'pdf.compression': 9})


#####   IFO Params   #####
arm_length = 3995 #dat's all u need to know

ff = nb.freq
#ff_O1 = spectra['o1'][:,0]
#ff_O2 = spectra['o2'][:,0]

num_points = 1500
fflog = np.logspace(np.log10(ff[0]), np.log10(ff[-1]), num_points)
#fflog_O1 = np.logspace(np.log10(ff_O1[0]), np.log10(ff_O1[-1]), num_points)
#fflog_O2 = np.logspace(np.log10(ff_O2[0]), np.log10(ff_O2[-1]), num_points)

plotDict = {}
labels = np.array([
'Measured noise (O3)',
'Sum of known noises',
'Quantum',
'Thermal',
'Seismic',
'Newtonian',
'Residual gas',
'Auxiliary length control',
'Alignment control',
'Beam jitter',
'Scattered light',
'Laser intensity',
'Laser frequency',
'Photodetector dark',
'Output mode cleaner length',
'Penultimate-mass actuator',
])

ASDs = np.array([
nb.DARM_reference,
nb.total,
nb.grouped_noises.quantum.total,
nb.grouped_noises.thermal.total,
nb.noises.seismic,
nb.noises.newtonian,
nb.noises.residual_gas,
nb.grouped_noises.lsc.total,
nb.grouped_noises.asc.total,
nb.grouped_noises.jitter.total,
nb.noises.scatter,
nb.noises.intensity,
nb.noises.frequency,
nb.noises.dark,
nb.noises.OMCLength,
nb.noises.DAC,
])

styles = np.array([
'C0-',
'k-',
'C4-',
'C3-',
'C2-',
'C9-',
'C8-',
'C1o',
'C3o',
'C0o',
'C4o',
'C2o',
'C6o',
'C7o',
'ko',
'C5o',
])

for label, ASD, style in zip(labels, ASDs, styles):
    # ASD_logbinned = np.dot(logbin_matrix, ASD)
    lin_log_ff, lin_log_ASD = linear_log_ASD(fflog, ff, ASD)
    if label == 'Measured noise (O3)':
        plt.loglog(lin_log_ff, lin_log_ASD, style, label=label, zorder=3)
    else:
        plt.loglog(lin_log_ff, lin_log_ASD, style, label=label)

#lin_log_ff_O1, lin_log_ASD_O1 = linear_log_ASD(fflog, ff_O1, arm_length*spectra['o1'][:,1])
#lin_log_ff_O2, lin_log_ASD_O2 = linear_log_ASD(fflog, ff_O2, arm_length*spectra['o2'][:,1])

#plt.loglog(lin_log_ff_O1, lin_log_ASD_O1, 'C1-', label='O1', alpha=0.5)
#plt.loglog(lin_log_ff_O2, lin_log_ASD_O2, 'C7-', label='O2', alpha=0.5)

plt.xlabel('Frequency [Hz]')
plt.ylabel(r'DARM [$\mathrm{m}/\sqrt{\mathrm{Hz}}$]')
plt.grid()
plt.grid(which='minor', ls='--', alpha=0.7)
plt.legend(ncol=2,markerscale=3,loc='upper right')

plt.xlim(10, 5e3)
plt.ylim(3e-21, 4e-17)

filename = 'lhoNb_logbinned.pdf'
plt.tight_layout()
print()
print('Writing plot to {}'.format(filename))
print()
plt.savefig(filename)
