import sys
import traceback
import numpy as np
import math
import matplotlib.pyplot as pl
import matplotlib.mlab as mlab
import matplotlib.colors
##from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate as interp
from beam import *
from lmtslr.grid.grid import *

class TsysView():
    def __init__(self,figure=1):
        self.figure = pl.figure(num=13, figsize=(6,6))

    def plot_tsys_levels(self,ICal):
        pl.clf()
        x = ICal.time-ICal.time[0]
        legend = []
        if False:
            for ipix in range(ICal.npix):
                legend.append('%2d %6.1f'%(ipix,ICal.tsys[ipix]))
                y = ICal.level[:,ipix]
                pl.plot(x,y,'.')
            pl.legend(legend,prop={'size': 10})
        else:
            if ICal.npix >= 16:
                ncols = 4
            else:
                ncols = 1
            nrows = int(ICal.npix/ncols)
            plot_scale = 0.0
            for ipix in range(ICal.npix):
                if ICal.tsys[ipix] < 500:
                    plot_scale = max(plot_scale, np.max(ICal.level[:,ipix]))
            #colors = pl.rcParams["axes.prop_cycle"]()
            colors = pl.rcParams['axes.prop_cycle']
            colors = [c['color'] for c in colors]
            plot_order = [1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16]
            for ipix in range(ICal.npix):
                if ncols == 1:
                    ipix1 = ipix+1
                else:
                    ipix1 = plot_order[(ipix%len(plot_order))]+int(ipix/len(plot_order))*len(plot_order) #ipix+1)
                ax = pl.subplot(nrows, ncols, ipix1)
                ax.tick_params(axis='both', which='major', labelsize=6)
                ax.tick_params(axis='both', which='minor', labelsize=6)
                label = '%2d %6.1f'%(ipix,ICal.tsys[ipix])
                legend.append(label)
                y = ICal.level[:,ipix]
                #color = next(colors)['color']
                color = colors[ipix%len(colors)]
                ax.plot(x,y,'.', color=color)
                plot_scale = np.mean(ICal.level[:,ipix])+np.min(ICal.level[:,ipix])
                ax.text(x[-1]/2, plot_scale/2, label, verticalalignment='center', horizontalalignment='center', zorder=10)
                if False and plot_scale != 0:
                    ax.set_ylim(0, plot_scale * 1.1)
        pl.suptitle("TSys %s ObsNum: %d"%(ICal.receiver,ICal.obsnum))

    def savefig(self, fname):
        pl.savefig(fname, bbox_inches='tight')

    def show(self):
        pl.show()
