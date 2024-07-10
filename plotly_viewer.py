import numpy as np
import math
import sys
import traceback

with_matplotlib = True

import scipy.interpolate as interp
from beam import *
from lmtslr.ifproc.ifproc import IFProc
from lmtslr.spec.spec import SpecBank
from lmtslr.grid.grid import Grid

row_map = 1
row_peak = 1

def config_plotly_viewer(w):
    global with_matplotlib
    with_matplotlib = w
    if with_matplotlib:
        global pl
        global mlab
        global mcolors
        import matplotlib.pyplot as pl
        import matplotlib.mlab as mlab
        import matplotlib.colors as mcolors
    else:
        global go
        global make_subplots
        global hex_to_rgb
        global px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from plotly.colors import hex_to_rgb
        import plotly.express as px
    
    
class PlotlyViewer():
    def __init__(self, figure=1):
        print('with_matplotlib', with_matplotlib)
        self.set_figure(figure)
        self.figures = {}

    def set_figure(self,figure):
        self.figure = figure

    def open_figure(self, nrows=1, ncols=1, specs=None):
        if with_matplotlib:
            pl.figure(self.figure)
            pl.clf()
        else:
            self.fig = make_subplots(
                rows=nrows, cols=ncols,
                shared_xaxes=False,
                specs=specs,
                subplot_titles=[" "]*ncols*nrows,
            )
            self.figures[self.figure] = self.fig

    def to_json(self):
        print(self.figures.keys())
        l = [self.figures[k].to_json() for k in self.figures.keys()]
        return l

    def write_json(self, fname):
        self.fig.write_json(fname)

    def write_html(self, fname):
        self.fig.write_html(fname)

    def savefig(self, fname):
        if with_matplotlib:
            pl.savefig(fname, bbox_inches='tight')
        else:
            self.fig.write_json(fname.replace('.png', '.json'))
            self.fig.write_html(fname.replace('.png', '.html'))

    def show(self):
        if with_matplotlib:
            pl.show()
        else:
            self.fig.show()

    def plot_spectrum(self, S, pixel, ispec, plot_axis, baseline_list, 
                      n_baseline_list):
        """
        Plots a specific spectrum in the time series for a specific 
        pixel.
        Args:
            S (object): SpecBank object
            pixel (int): pixel id for the spectra in the plot
            ispec (int): identifies the specific spectrum in the time 
                series to be plotted
            plot_axis (list): gives the desired axis for the plot in 
                the format [xlow, xhigh, ylow, yhigh]
            baseline_list (list): list of channels which will be 
                averaged to provide a constant baseline
            n_baseline_list (int): number of channels in the 
                baseline_list
        Returns:
            none
        """
        index = S.find_pixel_index(pixel)
        baseline = S.roach[index].baseline(ispec, baseline_list, 
                                           n_baseline_list)
        pl.plot((S.roach[pixel].reduced_spectra[ispec] - baseline))
        pl.axis(plot_axis)

    def find_peak_spectrum(self, S, pixel):
        """
        Returns the spectrum in time series which gives the maximum 
        value in map.
        Args:
            S (object): SpecBank object
            pixel (int): pixel id for the spectra in the plot
        Returns:
            int(ispec[0] (int)): position in spectrum where maximum is 
                located
        """
        map_index = S.find_map_pixel_index(pixel)
        mx = np.max(S.map_data[map_index])
        ispec = np.where(S.map_data[map_index] == mx)
        return(int(ispec[0]))
        
    def plot_peak_spectrum(self, S, pixel, plot_axis, baseline_list, 
                           n_baseline_list, plot_line_list=None, plot_baseline_list=None):
        """
        Plots the spectrum which gives maximum value in map.
        Args:
            S (object): SpecBank object
            pixel (int): pixel id for the spectra in the plot
            plot_axis (list): gives the desired axis for the plot in 
                the format [xlow, xhigh, ylow, yhigh]
            baseline_list (list): list of channels which will be 
                averaged to provide a constant baseline
            n_baseline_list (int): number of channels in the 
                baseline_list
        Returns:
            none
        """
        self.row = row_peak
        self.col = 1
        row = self.row
        col = self.col
        index = S.find_pixel_index(pixel)
        ispec = self.find_peak_spectrum(S, pixel)
        x = S.roach[index].reduced_spectra[ispec]
        baseline = np.sum(x[baseline_list]) / n_baseline_list
        v = S.create_velocity_scale()
        prange = np.where(np.logical_and(v >= plot_axis[0], v <= plot_axis[1]))
        plot_axis[2] = (x - baseline)[prange].min() * 1.1
        plot_axis[3] = (x - baseline)[prange].max() * 1.1
        mx = np.max((x-baseline)[prange])
        legend_label = 'Pixel %d\nPeak Spectrum[%d] = %0.2f'%(pixel, ispec, mx)
        if with_matplotlib:
            pl.plot(v, (x - baseline), label=legend_label)
        else:
            self.fig.add_trace(
                go.Scatter(
                    x=v, y=(x - baseline),
                    name='<br>'.join(legend_label.split('\n')),
                    showlegend=False),
                row=row, col=col)
        if plot_line_list is not None:
            for l in plot_line_list:
                if with_matplotlib:
                    pl.axvspan(l[0], l[1], alpha=0.1, color='b')
                else:
                    self.fig.add_vrect(
                        x0=l[0], x1=l[1], fillcolor='blue', opacity=0.1, row=row, col=col)
        if plot_baseline_list is not None:
            for l in plot_baseline_list:
                if with_matplotlib:
                    pl.axvspan(l[0], l[1], alpha=0.1, color='r')
                else:
                    self.fig.add_vrect(
                        x0=l[0], x1=l[1], fillcolor='red', opacity=0.1, row=row, col=col)
            
        #legend = pl.legend(fontsize='x-small')
        
        if with_matplotlib:
            pl.xlabel('Velocity (km/s)')
            pl.suptitle('ObsNum %d: %s %s %sGHz\n Pixel %d Peak Spectrum[%d] = %0.2f'%(
                S.obsnum, S.receiver, S.source, S.line_rest_frequency, pixel, ispec, mx))
            pl.axis(plot_axis)
        else:
            self.fig.update_xaxes(
                range=[plot_axis[0], plot_axis[1]],
                showgrid=True,
                title='Velocity (km/s)',
                row=row, col=col,
            )
            self.fig.update_yaxes(
                range=[plot_axis[2], plot_axis[3]],
                showgrid=True,
                row=row, col=col,
            )
            title='ObsNum %d: %s %s %sGHz<br>Pixel %d Peak Spectrum[%d] = %0.2f'%(
                S.obsnum, S.receiver, S.source, S.line_rest_frequency, pixel, ispec, mx)
            idx = 0
            font_size = 16
            self.fig.layout.annotations[idx].update(text=title, font_size=font_size)
            self.fig.update_layout(
                title_x=0.5,
                plot_bgcolor='white',
                legend=dict(x=0.7, y=0.9),
                font_size=16,
                height=800,
            )


    def plot_all_spectra(self, S, pixel, plot_axis, baseline_list, 
                         n_baseline_list, plot_line_list=None, plot_baseline_list=None):
        """
        Plots all spectra.
        Args:
            S (object): SpecBank object
            pixel (int): pixel id for the spectra in the plot
            plot_axis (list): gives the desired axis for the plot in 
                the format [xlow, xhigh, ylow, yhigh]
            baseline_list (list): list of channels which will be 
                averaged to provide a constant baseline
            n_baseline_list (int): number of channels in the 
                baseline_list
        Returns:
            none
        """
        pixel_index = S.find_pixel_index(pixel)
        v = S.create_velocity_scale()
        peak_index = np.argmax(S.map_data[pixel_index])
        prange = np.where(np.logical_and(v >= plot_axis[0], v <= plot_axis[1])
                         )
        plot_axis[2] = S.map_spectra[pixel_index][:, prange].min() * 1.1
        plot_axis[3] = S.map_spectra[pixel_index][:, prange].max() * 1.1
        plot_axis2 = np.zeros(4)
        plot_axis2[0] = plot_axis[0]
        plot_axis2[1] = plot_axis[1]
        plot_axis2[2] = 0#S.map_data.min()*1.1
        plot_axis2[3] = S.map_data.max() * 1.1
        plen = len(S.map_spectra[pixel_index])
        point_list = []
        xlen = ylen = int(math.sqrt(plen))

        # create an 2d list of array indices
        a = [[i + j * ylen for i in range(xlen)] for j in range(ylen)]
        # change into a numpy array to manipulate
        a = np.array(a)
        # flip the order of every other row
        a[1::2, :] = a[1::2, ::-1]
        # flip the whole array
        a = np.flipud(a)
        # flatten the array to get a 1d list
        a = a.flatten()

        if with_matplotlib:
            for index in range(plen):
                plot_index = a[index]
                ax = pl.subplot(xlen, ylen, plot_index+1)
                ax.tick_params(axis='both', which='major', labelsize=6)
                ax.tick_params(axis='both', which='minor', labelsize=6)
                ax.plot(v, (S.map_spectra[pixel_index][index] - np.sum(
                    S.map_spectra[pixel_index][index][S.blist]) / S.nb))
                if plot_line_list is not None:
                    for l in plot_line_list:
                        ax.axvspan(l[0], l[1], alpha=0.1, color='b')
                if plot_baseline_list is not None:
                    for l in plot_baseline_list:
                        ax.axvspan(l[0], l[1], alpha=0.1, color='r')
                ax.axis(plot_axis)
                ax.text(plot_axis[0] + 0.1 * (plot_axis[1] - plot_axis[0]), 
                        plot_axis[3] - 0.2 * (plot_axis[3] - plot_axis[2]), 
                        '%5.2f %5.2f %5.2f'%(S.map_x[pixel_index][index], 
                                             S.map_y[pixel_index][index], 
                                             S.map_data[pixel_index][index]), 
                                             size='6')
                ax2 = ax.twinx()
                ax2.tick_params(axis='both', which='major', labelsize=6)
                ax2.tick_params(axis='both', which='minor', labelsize=6)
                ax2.plot(0.5 * (plot_axis[0] + plot_axis[1]), 
                         S.map_data[pixel_index][index], 'or')
                ax2.axis(plot_axis2)
            pl.tight_layout(rect=[0, 0.03, 1, 0.9])
            pl.suptitle('ObsNum %d: %s %s %sGHz\n Pixel %d'%(S.obsnum, S.receiver,
                S.source, S.line_rest_frequency, pixel))
        else:
            nrows = ylen
            ncols = xlen
            self.open_figure(nrows=nrows, ncols=ncols,
                             specs=[[{"secondary_y": True}]*ncols]*nrows)
            for index in range(plen):
                plot_index = a[index]
                row = int(plot_index/nrows)+1
                col = (plot_index%ncols)+1
                x = v
                y = (S.map_spectra[pixel_index][index] - np.sum(S.map_spectra[pixel_index][index][S.blist]) / S.nb)
                self.fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        showlegend=False,
                    ),
                    row=row, col=col,
                )
                self.fig.add_trace(
                    go.Scatter(
                        x=[0.5 * (plot_axis[0] + plot_axis[1])], 
                        y=[S.map_data[pixel_index][index]],
                        mode="markers",
                        line=dict(color='black'),
                        showlegend=False,
                    ),
                    secondary_y=True,
                    row=row, col=col,
                )
                if plot_line_list is not None:
                    for l in plot_line_list:
                        self.fig.add_vrect(
                            x0=l[0], x1=l[1], fillcolor='blue', opacity=0.1,
                            row=row, col=col
                        )
                if plot_baseline_list is not None:
                    for l in plot_baseline_list:
                        self.fig.add_vrect(
                            x0=l[0], x1=l[1], fillcolor='red', opacity=0.1,
                            row=row, col=col
                        )
                self.fig.update_xaxes(
                    range=[plot_axis[0], plot_axis[1]],
                    showgrid=True,
                    row=row, col=col
                )
                self.fig.update_yaxes(
                    range=[plot_axis[2], plot_axis[3]],
                    showgrid=True,
                    row=row, col=col
                )
                self.fig.update_yaxes(
                    range=[plot_axis2[2], plot_axis2[3]],
                    secondary_y=True,
                    showgrid=False,
                    row=row, col=col
                )
                title = '%5.2f %5.2f %5.2f'%(S.map_x[pixel_index][index], 
                                             S.map_y[pixel_index][index], 
                                             S.map_data[pixel_index][index])
                idx = (row-1)*ncols+(col-1)
                font_size = 14
                self.fig.layout.annotations[idx].update(text=title, font_size=font_size)
            title = 'ObsNum %d: %s %s %sGHz\n Pixel %d'%(S.obsnum, S.receiver,
                                                         S.source, S.line_rest_frequency, pixel)
            font_size = 16
            width = max(1000, ncols*200)
            height = max(1000, nrows*200)
            self.fig.update_layout(title=title, title_x=0.5, font_size=font_size, width=width, height=height)
            self.fig.update_layout(margin=dict(t=200))


    def waterfall(self, S, pixel, window, plot_range, baseline_list, 
                  n_baseline_list):
        """
        Makes a waterfall plot of the reduced data with baseline 
        removed.
        Args:
            S (object): SpecBank to be viewed
            pixel (int): index of target pixel
            window (list): start and stop indices in the spectra time 
                series in format [start,stop]
            plot_range (list): range of the intensity scale in format 
                [ta_min, ta_max]
            baseline_list (list): list of channels which will be 
                averaged to provide a constant baseline
            n_baseline_list (int): number of channels in the 
                baseline_list
        Returns:
            none
        """
        index = S.find_pixel_index(pixel)
        ispec = self.find_peak_spectrum(S, pixel)
        start = ispec+window[0]
        stop = ispec+window[1]
        if start < 0:
            start = 0
        if stop > S.roach[index].nspec:
            stop = S.roach[index].nspec
        if stop > len(S.roach[index].reduced_spectra):
            stop = len(S.roach[index].reduced_spectra)
        r = range(start,stop)
        w = np.zeros((len(r),S.nchan))
        for i,ispec in enumerate(r):
            baseline,rms = S.roach[index].baseline(
                S.roach[index].reduced_spectra[ispec], baseline_list, 
                n_baseline_list)
            for j in range(S.nchan):
                w[i,j] = (S.roach[index].reduced_spectra[ispec][j] - 
                          baseline[j])
        pl.imshow(w, clim=plot_range, origin='lower', extent=[S.c2v(0), 
                  S.c2v(S.nchan - 1), start, stop])
        pl.suptitle('Obsnum: %d Pixel: %d Scans: %d to %d'%(S.obsnum, pixel, 
                                                            start, stop))
        pl.xlabel('Velocity (km/s)')
                

    def sanchez_map(self, S, map_region, grid_spacing, plot_range, 
                    pixel_list=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]):
        """
        Makes a "Sanchez Map" with all pixels displayed on sky in their
        grid positions. The map data must be created beforehand.
        Args:
            S (object): SpecBank to be viewed
            map_region (list): extent of the map in arcsec in format 
                [low left, low right, high left, high right]
            grid_spacing (float): size of the map cells in arcsec
            pixel_list (list): list of pixels to process (default is 
                all)
        Returns:
            none
        """
        if not with_matplotlib:
            idx = 0
            font_size = 16
            self.fig.layout.annotations[idx].update(text='sanchez_map Not Supported', font_size=font_size)
            return
        map_x = S.map_x
        map_y = S.map_y
        map_data = S.map_data
        if True or not map_region:
            map_region = [0, 0, 0, 0]
            map_region[0] = 1.1*(map_x[0]).min()
            map_region[1] = 1.1*(map_x[0]).max()
            map_region[2] = 1.1*(map_y[0]).min()
            map_region[3] = 1.1*(map_y[0]).max()
            #np.set_printoptions(threshold=sys.maxsize)
            #print(map_x, map_y)
        nx = int((map_region[1] - map_region[0]) / grid_spacing + 1)
        ny = int((map_region[3] - map_region[2]) / grid_spacing + 1)
        nx = max(nx, ny)
        ny = nx
        xi = np.linspace(map_region[0], map_region[1], nx)
        yi = np.linspace(map_region[2], map_region[3], ny)
        grid_x, grid_y = np.mgrid[map_region[0]:map_region[1]:complex(nx), map_region[2]:map_region[3]:complex(ny)]
        zi_sum = np.zeros((nx, ny))
        for pixel in pixel_list:
            index = S.find_map_pixel_index(pixel)
            try :
                zi = interp.griddata((map_x[index],map_y[index]),map_data[index],(grid_x,grid_y),method='linear').T
            except Exception as e:
                zi = mlab.griddata(map_x[index], map_y[index], 
                                   map_data[index], xi, yi, interp='linear')
            zi_sum = zi_sum + zi
        pl.imshow(zi_sum, clim=plot_range, interpolation='bicubic', 
                  cmap=pl.cm.jet, origin='lower', extent=map_region)
        pl.plot(map_x[index], map_y[index])
        pl.xlabel('dAz (")')
        pl.ylabel('dEl (")')
        pl.suptitle('Spectral Line Sanchez Map: %d'%(S.obsnum))
        
    def map(self, S, map_region, grid_spacing, plot_range, 
            pixel_list=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]):
        """
        Aligns the individual maps for each pixel according to a 
        nominal grid model. The map data must be created beforehand.
        Args:
            S (object): SpecBank to be viewed
            map_region (list): extent of the map in arcsec in format 
                [low left, low right, high left, high right]
            grid_spacing (float): size of the map cells in arcsec
            pixel_list (list): list of pixels to process (default is 
                all)
        Returns:
            none
        """
        g = Grid(S.receiver)
        gx, gy = g.azel(S.elev / 180 * np.pi, S.tracking_beam)
        print('azel', gx, gy)
        #gx, gy = g.radec(S.elev / 180 * np.pi, np.mean(S.map_p), S.tracking_beam)
        print('radec', gx, gy)
        map_x = S.map_x
        map_y = S.map_y
        map_data = S.map_data
        if True or not map_region:
            map_region = [0, 0, 0, 0]
            map_region[0] = 1.1*(map_x[0]).min()
            map_region[1] = 1.1*(map_x[0]).max()
            map_region[2] = 1.1*(map_y[0]).min()
            map_region[3] = 1.1*(map_y[0]).max()
            #np.set_printoptions(threshold=sys.maxsize)
            #print(map_x, map_y)
        nx = int((map_region[1] - map_region[0]) / grid_spacing + 1)
        ny = int((map_region[3] - map_region[2]) / grid_spacing + 1)
        nx = max(nx, ny)
        ny = nx
        xi = np.linspace(map_region[0], map_region[1], nx)
        yi = np.linspace(map_region[2], map_region[3], ny)
        grid_x, grid_y = np.mgrid[map_region[0]:map_region[1]:complex(nx), map_region[2]:map_region[3]:complex(ny)]
        zi_sum = np.zeros((nx, ny))
        wi_sum = np.zeros((nx, ny))
        for pixel in pixel_list:
            index = S.find_map_pixel_index(pixel)
            wdata = np.ones(len(map_data[index]))
            try :
                zi = interp.griddata((map_x[index]-gx[pixel],map_y[index]-gy[pixel]),map_data[index],(grid_x,grid_y),method='linear').T
                wi = interp.griddata((map_x[index]-gx[pixel],map_y[index]-gy[pixel]),wdata,(grid_x,grid_y),method='linear').T
            except:
                zi = mlab.griddata(map_x[index] - gx[pixel], 
                                   map_y[index] - gy[pixel], map_data[index], 
                                   xi, yi, interp='linear')
                wi = mlab.griddata(map_x[index] - gx[pixel], 
                                   map_y[index] - gy[pixel], wdata, xi, yi, 
                                   interp='linear')
            zi_sum = zi_sum + zi
            wi_sum = wi_sum + wi
        pl.imshow(zi_sum / wi_sum, clim=plot_range, interpolation='bicubic', 
                  cmap=pl.cm.jet, origin='lower', extent=map_region)
        pl.plot(map_x[index], map_y[index])
        pl.axis(map_region)
        pl.xlabel('dAz (")')
        pl.ylabel('dEl (")')
        pl.suptitle('Spectral Line Aligned Map: %d'%(S.obsnum))

    def pixel_map(self, S, pixel, map_region, grid_spacing, show_points=False
                 ):
        """
        Maps results for a single pixel. The map data must be created beforehand.
        Args:
            S (object): SpecBank to be viewed
            map_region (list): extent of the map in arcsec in format 
                [low left, low right, high left, high right]
            grid_spacing (float): size of the map cells in arcsec
            pixel_list (list): list of pixels to process (default is 
                all)
            show_points (bool): determines whether to plot the 
                locations of the spectra in the map (default is False)
        Returns:
            none
        """
        index = S.find_map_pixel_index(pixel)
        nx = int((map_region[1] - map_region[0]) / grid_spacing + 1)
        ny = int((map_region[3] - map_region[2]) / grid_spacing + 1)
        xi = np.linspace(map_region[0], map_region[1], nx)
        yi = np.linspace(map_region[2], map_region[3], ny)
        try :
            zi = interp.griddata((S.map_x[index],
                                  S.map_y[index]), 
                                 S.map_data[index], (xi, yi), method='linear')
        except:
            zi = mlab.griddata(S.map_x[index], 
                               S.map_y[index], S.map_data[index], 
                               xi, yi, interp='linear')
        pl.imshow(zi, interpolation='bicubic', cmap=pl.cm.jet, origin='lower',
                  extent=map_region)
        pl.xlabel('X (")')
        pl.ylabel('Y (")')
        pl.suptitle('Spectral Line Map: Obsnum: %d  Pixel: %d'%(
            S.ifproc.obsnum, pixel))
        pl.axis('equal')
        if show_points:
            pl.plot(S.map_x[index], S.map_y[index], 'w.')
            pl.axis(map_region)
        pl.colorbar()

    def plot_on(self, S):
        """
        Makes plots for pixels in SpecBank object S.
        Args:
            S (object): SpecBank to be viewed
        Returns:
            none
        """
        plot_order = [1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16]
        if with_matplotlib:
            for ipix in range(S.npix):
                pixel_id = S.roach_pixel_ids[ipix]
                ax = pl.subplot(4, 4, plot_order[pixel_id])
                ax.tick_params(axis='both', which='major', labelsize=6)
                ax.tick_params(axis='both', which='minor', labelsize=6)
                ax.plot(S.roach[ipix].on_spectrum)
                l = len(S.roach[ipix].on_spectrum)
                pl.xticks(np.arange(0, l+1, l/4))
            pl.suptitle('%s: ObsNum %d\n%s %s GHz'%(S.obspgm, S.obsnum, 
                S.receiver, S.line_rest_frequency))
        else:
            nrows = 4
            ncols = 4
            self.open_figure(nrows=nrows, ncols=ncols)
            for ipix in range(S.npix):
                pixel_id = S.roach_pixel_ids[ipix]
                col = int(pixel_id/ncols)+1
                row = (pixel_id%nrows)+1
                self.fig.add_trace(
                    go.Scatter(
                        y=S.roach[ipix].on_spectrum,
                        showlegend=False,
                    ),
                    row=row, col=col,
                )
            title = '%s: ObsNum %d\n%s %s GHz'%(S.obspgm, S.obsnum, 
                S.receiver, S.line_rest_frequency)
            font_size = 16
            width = max(1200, ncols*300)
            height = max(1200, nrows*300)
            self.fig.update_layout(title=title, title_x=0.5, font_size=font_size, width=width, height=height)
            self.fig.update_layout(margin=dict(t=200))

    def plot_ps(self, S, baseline_order, plot_axis=[-200, 200, -0.5, 2.0], 
                line_stats_all=[], plot_line_list=None, plot_baseline_list=None):
        """
        Makes position-switch plots for pixels in SpecBank object S.
        Args:
            S (object): SpecBank to be viewed
            baseline_order (int): not used
            plot_axis (list): list of axis limits in format [xmin, 
                xmax, ymin, ymax]
            line_stats_all (list): list of LineStatistics objects 
                holding line statistics data (default is empty list)
        Returns:
            none
        """
        plot_order = [1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16]
        line_stats = line_stats_all[0]
        prange = np.where(np.logical_and(line_stats.v >= plot_axis[0], 
                          line_stats.v <= plot_axis[1]))
        plot_axis[2] = line_stats.spectrum[prange].min() * 1.1
        plot_axis[3] = line_stats.spectrum[prange].max() * 1.4
        for ipix in range(S.npix):
            if ipix == 0: continue
            line_stats = line_stats_all[ipix]
            min_ps = line_stats.spectrum[prange].min() * 1.1
            max_ps = line_stats.spectrum[prange].max() * 1.4
            if min_ps < plot_axis[2]: plot_axis[2] = min_ps
            if max_ps > plot_axis[3]: plot_axis[3] = max_ps

        if with_matplotlib:
            for ipix in range(S.npix):
                pixel_id = S.roach_pixel_ids[ipix]
                ax = pl.subplot(4, 4, plot_order[pixel_id])
                ax.tick_params(axis='both', which='major', labelsize=6)
                ax.tick_params(axis='both', which='minor', labelsize=6)

                # for each line, fit baseline and compute line statistics
                line_stats = line_stats_all[ipix]
                ax.plot(line_stats.v, line_stats.spectrum)
                if plot_line_list is not None:
                    for l in plot_line_list:
                        ax.axvspan(l[0], l[1], alpha=0.1, color='b')
                if plot_baseline_list is not None:
                    for l in plot_baseline_list:
                        ax.axvspan(l[0], l[1], alpha=0.1, color='r')
                ax.axis(plot_axis)
                xtext = plot_axis[0] + 0.05 * (plot_axis[1] - plot_axis[0])
                ytext = plot_axis[3] - 0.05 * (plot_axis[3] - plot_axis[2])
                ax.text(xtext, ytext, '%2d I=%8.3f(%8.3f)'%(pixel_id, 
                    line_stats.yint, line_stats.yerr), horizontalalignment='left',
                    verticalalignment='top', fontsize=6)
                ytext = plot_axis[3] - 0.15 * (plot_axis[3] - plot_axis[2])
                pl.text(xtext, ytext, 'V=%8.3f RMS=%8.3f'%(line_stats.xmean, 
                                                           line_stats.rms), 
                        horizontalalignment='left', verticalalignment='top', 
                        fontsize=6)
            pl.suptitle('%s: ObsNum %d\n%s %s GHz'%(S.obspgm, S.obsnum, 
                S.receiver, S.line_rest_frequency))
        else:
            nrows = 4
            ncols = 4
            self.open_figure(nrows=nrows, ncols=ncols)
            for ipix in range(S.npix):
                pixel_id = S.roach_pixel_ids[ipix]
                col = int(pixel_id/ncols)+1
                row = (pixel_id%nrows)+1

                # for each line, fit baseline and compute line statistics
                line_stats = line_stats_all[pixel_id]
                self.fig.add_trace(
                    go.Scatter(
                        x=line_stats.v,
                        y=line_stats.spectrum,
                        showlegend=False,
                    ),
                    row=row, col=col,
                )
                if plot_line_list is not None:
                    for l in plot_line_list:
                        self.fig.add_vrect(
                            x0=l[0], x1=l[1], fillcolor='blue', opacity=0.1,
                            row=row, col=col
                        )
                if plot_baseline_list is not None:
                    for l in plot_baseline_list:
                        self.fig.add_vrect(
                            x0=l[0], x1=l[1], fillcolor='red', opacity=0.1,
                            row=row, col=col
                        )
                self.fig.update_xaxes(
                    range=[plot_axis[0], plot_axis[1]],
                    showgrid=True,
                    row=row, col=col
                )
                self.fig.update_yaxes(
                    range=[plot_axis[2], plot_axis[3]],
                    showgrid=True,
                    row=row, col=col
                )
                idx = (row-1)*ncols+(col-1)
                font_size = 16
                title = '%2d I=%8.3f(%8.3f)'%(pixel_id, 
                                              line_stats.yint, line_stats.yerr)
                title += '<br>'
                title += 'V=%8.3f RMS=%8.3f'%(line_stats.xmean, 
                                              line_stats.rms)
                title += '<br>'
                title += 'V=%8.3f RMS=%8.3f'%(line_stats.xmean, 
                                              line_stats.rms)
                self.fig.layout.annotations[idx].update(text=title, font_size=font_size)
            title = '%s: ObsNum %d\n%s %s GHz'%(S.obspgm, S.obsnum, S.receiver, S.line_rest_frequency)
            font_size = 16
            width = max(1200, ncols*300)
            height = max(1200, nrows*300)
            self.fig.update_layout(title=title, title_x=0.5, font_size=font_size, width=width, height=height)
            self.fig.update_layout(margin=dict(t=200))

    def plot_bs(self, S, baseline_order, plot_axis=[-200, 200, -0.5, 2.0], 
                line_stats=None, plot_line_list=None, plot_baseline_list=None):
        """
        Makes beam-switch plots for pixels in SpecBank object S.
        Args:
            S (object): SpecBank to be viewed
            baseline_order (int): not used
            plot_axis (list): list of axis limits in format [xmin, 
                xmax, ymin, ymax]
            line_stats (object): LineStatistics object holding line 
                statistics data (default is None)
        Returns:
            none
        """
        plot_order = [1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16];
        prange = np.where(np.logical_and(line_stats.v >= plot_axis[0], 
                          line_stats.v <= plot_axis[1]))
        plot_axis[2] = line_stats.spectrum[prange].min() * 1.1
        plot_axis[3] = line_stats.spectrum[prange].max() * 1.4
        if with_matplotlib:
            ax = pl.subplot(1,1,1)
            ax.tick_params(axis='both', which='major', labelsize=6)
            ax.tick_params(axis='both', which='minor', labelsize=6)

            ax.plot(line_stats.v, line_stats.spectrum)
            ax.axis(plot_axis)
            if plot_line_list is not None:
                for l in plot_line_list:
                    ax.axvspan(l[0], l[1], alpha=0.1, color='b')
            if plot_baseline_list is not None:
                for l in plot_baseline_list:
                    ax.axvspan(l[0], l[1], alpha=0.1, color='r')

            xtext = plot_axis[0] + 0.05 * (plot_axis[1] - plot_axis[0])
            ytext = plot_axis[3] - 0.05 * (plot_axis[3] - plot_axis[2])
            ax.text(xtext, ytext, '%2d/%2d I=%8.3f(%8.3f)'%(S.roach_pixel_ids[0], 
                S.roach_pixel_ids[1], line_stats.yint, line_stats.yerr), 
                horizontalalignment='left', verticalalignment='top', fontsize=10)
            ytext = plot_axis[3] - 0.1 * (plot_axis[3] - plot_axis[2])
            pl.text(xtext, ytext, 'V=%8.3f RMS=%8.3f'%(line_stats.xmean, 
                                                       line_stats.rms), 
                    horizontalalignment='left', verticalalignment='top', 
                    fontsize=10)
            pl.suptitle('%s: ObsNum %d\n%s %s GHz'%(S.obspgm, S.obsnum, 
                S.receiver, S.line_rest_frequency))
        else:
            self.fig.add_trace(
                go.Scatter(
                    x=line_stats.v,
                    y=line_stats.spectrum,
                ),
            )
            if plot_line_list is not None:
                for l in plot_line_list:
                    self.fig.add_vrect(
                        x0=l[0], x1=l[1], fillcolor='blue', opacity=0.1)
            if plot_baseline_list is not None:
                for l in plot_baseline_list:
                    self.fig.add_vrect(
                        x0=l[0], x1=l[1], fillcolor='red', opacity=0.1)
            self.fig.update_xaxes(
                range=[plot_axis[0], plot_axis[1]],
                showgrid=True,
            )
            self.fig.update_yaxes(
                range=[plot_axis[2], plot_axis[3]],
                showgrid=True,
            )
            idx = 0
            font_size = 16
            title = '%s: ObsNum %d\n%s %s GHz'%(S.obspgm, S.obsnum, 
                                                S.receiver, S.line_rest_frequency)
            title += '<br>'
            title += '%2d/%2d I=%8.3f(%8.3f)'%(S.roach_pixel_ids[0], 
                                               S.roach_pixel_ids[1], line_stats.yint, line_stats.yerr)
            title += '<br>'
            title += 'V=%8.3f RMS=%8.3f'%(line_stats.xmean, 
                                          line_stats.rms)
            self.fig.layout.annotations[idx].update(text=title, font_size=font_size)
            self.fig.update_layout(
                title_x=0.5,
                plot_bgcolor='white',
                legend=dict(x=0.7, y=0.9),
                font_size=16,
                height=800,
            )
            


    def plot_tsys(self, S):
        """
        Plots the tsys data from spec_cal object S.
        Args:
            S (object): spec_cal to be viewed
        Returns:
            none
        """
        plot_scale = 0.
        nscale = 0
        for ipix in range(S.npix):
            indx_fin = np.where(np.isfinite(S.roach[ipix].tsys_spectrum))
            indx_inf = np.where(np.isinf(S.roach[ipix].tsys_spectrum))
            indx_nan = np.where(np.isnan(S.roach[ipix].tsys_spectrum))
            print(ipix, 'fin------------', indx_fin[0])
            print(ipix, 'inf------------', indx_inf[0])
            print(ipix, 'nan------------', indx_nan[0])
            l_fin = len(indx_fin[0])
            if l_fin > 0 and S.roach[ipix].tsys > 0 and \
               S.roach[ipix].tsys < 500:
                plot_scale = plot_scale + S.roach[ipix].tsys
                nscale = nscale + 1
        if nscale > 0:
            plot_scale = plot_scale / nscale
        plot_order = [1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16];
        nrows = int(S.npix/4)
        # set nrows to 4 always for sequoia in case we're missing some roach files
        if S.receiver == 'Sequoia':
            nrows = 4
            ncols = 4
        elif S.receiver == 'Msip1mm':
            nrows = 4
            ncols = 1
        if nrows == 0: nrows = 1
        if not with_matplotlib:
            self.open_figure(nrows=nrows, ncols=ncols)
        for ipix in range(S.npix):
            pixel_id = S.roach_pixel_ids[ipix]
            if ncols == 1:
                ipix1 = ipix+1
                col = 1
                row = ipix1
            else:
                ipix1 = plot_order[pixel_id]
                #ipix1 = plot_order[(pixel_id%len(plot_order))]+int(ipix/len(plot_order))*len(plot_order)
                col = int(pixel_id/ncols)+1
                row = (pixel_id%nrows)+1
            print(ipix, pixel_id, ipix1, pixel_id, row, col)
            if with_matplotlib:
                ax = pl.subplot(nrows, ncols, ipix1)
                ax.tick_params(axis='both', which='major', labelsize=6)
                ax.tick_params(axis='both', which='minor', labelsize=6)
            indx_fin = np.where(np.isfinite(S.roach[ipix].tsys_spectrum))
            l_fin = len(indx_fin[0])
            if with_matplotlib:
                if l_fin > 0:
                    pl.plot(S.roach[ipix].tsys_spectrum[indx_fin])
                    if False:
                        pl.text(S.nchan / 2, 10, '%d %6.0fK'%(pixel_id+int(ipix/len(plot_order))*len(plot_order), 
                                                              S.roach[ipix].tsys), 
                                horizontalalignment='center')
                    else:
                        pl.text(l_fin / 2, 10, '%d %6.0fK'%(pixel_id+int(ipix/len(plot_order))*len(plot_order), 
                                                            S.roach[ipix].tsys), 
                                horizontalalignment='center')
                    if plot_scale != 0:
                        pl.axis([0, S.nchan, 0, plot_scale * 1.5])
                else:
                    pl.text(0.1, 0.5, '%d NaN'%(pixel_id))
            else:
                if l_fin > 0:
                    y = S.roach[ipix].tsys_spectrum[indx_fin]
                    label = '%d %6.0fK'%(pixel_id+int(ipix/len(plot_order))*len(plot_order), 
                                         S.roach[ipix].tsys)
                    self.fig.add_trace(
                        go.Scatter(
                            y=y,
                            name=label,
                            mode="markers",
                            showlegend=False),
                        row=row, col=col,
                        )
                    self.fig.update_xaxes(range=[0, S.nchan])
                    self.fig.update_yaxes(range=[0, plot_scale * 1.5])
                    idx = (row-1)*ncols+(col-1)
                    title = label
                    font_size = 14
                    self.fig.layout.annotations[idx].update(text=title, font_size=font_size)
        if with_matplotlib:
            pl.suptitle('TSys: ObsNum %d\n%s %s GHz, bank %d'%(S.obsnum, S.receiver, 
                                                               S.line_rest_frequency, S.bank))
        else:
            title = 'TSys: ObsNum %d\n%s %s GHz, bank %d'%(S.obsnum, S.receiver, 
                                                           S.line_rest_frequency, S.bank)
            font_size = 16
            width = max(400, ncols*300)
            height = max(200, nrows*200)
            self.fig.update_layout(title=title, title_x=0.5, font_size=font_size, width=width, height=height)


    def show_fit(self,B,ipix):
        """ plots the scan of data and the best fit model.
            B is the BeamMap with data and model
            ipix is the pixel id for plotting
        """
        index = B.BData.find_map_pixel_index(ipix)
        pix_list_index = B.find_pixel_index(ipix)
        model = np.array(B.model[pix_list_index])
        offset = B.peak_fit_params[pix_list_index,0]/2.0
        if with_matplotlib:
            pl.plot(B.BData.map_data[index],'k',label='Data')
            pl.plot(model,'r',label='Model')
            pl.plot(B.BData.map_data[index]-model-offset,'g',label='Resid')
            pl.plot(-offset*np.ones(len(B.BData.map_data[index])),'b')
            pl.xlabel('Sample')
            pl.legend()
            pl.suptitle('Model Fit for %d Pixel:%d'%(B.obsnum,ipix))
        else:
            self.fig.add_trace(
                go.Scatter(
                    y=B.BData.map_data[index],
                    name='Data',
                    mode="lines",
                    line=dict(color='black'),
                    showlegend=True),
            )
            self.fig.add_trace(
                go.Scatter(
                    y=model,
                    name='Model',
                    mode="lines",
                    line=dict(color='red'),
                    showlegend=True),
            )
            self.fig.add_trace(
                go.Scatter(
                    y=B.BData.map_data[index]-model-offset,
                    name='Resid',
                    mode="lines",
                    line=dict(color='green'),
                    showlegend=True),
            )
            self.fig.add_trace(
                go.Scatter(
                    y=-offset*np.ones(len(B.BData.map_data[index])),
                    name='Offset',
                    mode="lines",
                    line=dict(color='blue'),
                    showlegend=True),
            )
            self.fig.update_xaxes(
                showgrid=True,
                title='Sample',
            )
            self.fig.update_layout(
                title='Model Fit for %d Pixel:%d'%(B.obsnum,ipix),
                title_x=0.5,
                #plot_bgcolor='white',
                font_size=16,
                height=800,
            )

    def show_peaks(self,B,apply_grid_corrections=False,show_map_ids=True,show_map_points=-1):
        """ plots and identifies the peak positions in the BeamMap fits
            B is the BeamMap object with the data and fit results
            apply_grid_corrections=True will use the nominal grid to offset positions
        """
        row = self.row
        col = self.col
        g = Grid(B.BData.receiver)
        gx,gy = g.azel(B.BData.elev/180.*np.pi,B.BData.tracking_beam)
        
        if apply_grid_corrections:
            gxl = gx[B.pix_list]
            gyl = gy[B.pix_list]
        else:
            gxl = np.zeros(B.n_pix_list)
            gyl = np.zeros(B.n_pix_list)
        if with_matplotlib:
            pl.plot(B.peak_fit_params[:,1]-gxl,B.peak_fit_params[:,3]-gyl,'ko')
        else:
            self.fig.add_trace(go.Scatter(x=B.peak_fit_params[:,1]-gxl,
                                          y=B.peak_fit_params[:,3]-gyl,
                                          text=['%d'%B.pix_list[i] for i in range(B.n_pix_list)],
                                          mode="lines+text+markers",
                                          textposition='middle right',
                                          textfont_size=24,
                                          name='',
                                          marker=dict(color='black', size=20),
                                          showlegend=False,
                                          ),
                               row=row, col=col
                               )
        
        if show_map_ids:
            for i in range(B.n_pix_list):
                if with_matplotlib:
                    pl.text(B.peak_fit_params[i,1]-gxl[i],B.peak_fit_params[i,3]-gyl[i],'%d'%(B.pix_list[i]))

        if show_map_points >= 0:
            pixel_index = B.BData.find_map_pixel_index(show_map_points)
            plen = len(B.BData.map_x[pixel_index])
            xlen = ylen = int(math.sqrt(plen))
            for i in range(xlen):
                for j in range(ylen):
                    index = i+j*xlen
                    if with_matplotlib:
                        pl.plot(B.BData.map_x[pixel_index][index]-gxl[pixel_index],B.BData.map_y[pixel_index][index]-gyl[pixel_index],'.r')

    def sanchez_map(self,B,map_region,grid_spacing):
        """ makes a "Sanchez Plot" of the beams on the sky
            B is BeamMap object with data and fits
            map_region is the extent of the map: [low left, low right, high left, high right] (arcsec)
            grid_spacing is the size of the map cells (arcsec)
        """
        g = Grid(B.BData.receiver)
        gx,gy = g.azel(B.BData.elev/180.*np.pi,B.BData.tracking_beam)
        if not map_region:
            map_region = [0, 0, 0, 0]
            map_region[0] = 1.1*(B.BData.map_x[0]).min()
            map_region[1] = 1.1*(B.BData.map_x[0]).max()
            map_region[2] = 1.1*(B.BData.map_y[0]).min()
            map_region[3] = 1.1*(B.BData.map_y[0]).max()
            #np.set_printoptions(threshold=sys.maxsize)
            #print(map_x, map_y)
            print ('map_region', map_region)
        nx = int((map_region[1]-map_region[0])/grid_spacing)+1
        ny = int((map_region[3]-map_region[2])/grid_spacing)+1
        nx = ny = min(nx, ny)
        xi = np.linspace(map_region[0],map_region[1],nx)
        yi = np.linspace(map_region[2],map_region[3],ny)
        grid_x, grid_y = np.mgrid[map_region[0]:map_region[1]:complex(nx), map_region[2]:map_region[3]:complex(ny)]
        zi_sum = np.zeros((nx,ny))
        for i in range(B.n_pix_list):
            pixel = B.pix_list[i]
            index = B.BData.find_map_pixel_index(pixel)
            try:
                print('trying scipy.interpolate.griddata')
                zi = interp.griddata((B.BData.map_x[index],B.BData.map_y[index]),B.BData.map_data[index],(grid_x,grid_y),method='linear',fill_value=B.BData.map_data[index].min()).T
            except Exception as e:
                print(e)
                zi = mlab.griddata(B.BData.map_x[index],B.BData.map_y[index],B.BData.map_data[index],xi,yi,interp='linear')
            zi_sum = zi_sum + zi
        pl.imshow(zi_sum,interpolation='bicubic',cmap=pl.cm.jet,origin='lower',extent=map_region)
        pl.axis('equal')
        pl.grid()
        pl.xlabel('X (")')
        pl.ylabel('Y (")')
        isGood = np.zeros((B.n_pix_list))
        isGood = (B.peak_fit_status[:] != 5)
        az_map_offset = B.peak_fit_params[np.nonzero(isGood),1]
        el_map_offset = B.peak_fit_params[np.nonzero(isGood),3]
        textstr =           'Az Offset  %6.4f'%(az_map_offset.mean()-np.mean(gx[B.pix_list])) + '\n' 
        textstr = textstr + 'El Offset  %6.4f'%(el_map_offset.mean()-np.mean(gy[B.pix_list]))
        pl.suptitle('ObsNum %d: %s %s %sGHz\n %s'%(B.obsnum,B.BData.receiver,B.BData.source,B.BData.line_rest_frequency,textstr)) 
        try:
            pl.tight_layout(rect=[0, 0.03, 1, 0.9])
        except:
            pass
        pl.colorbar()

    def map(self,B,map_region,grid_spacing,apply_grid_corrections=False,display_coord=None):
        """ map aligns all maps on the sky using nominal grid and averages the maps
            B is BeamMap object with data and fits
            map_region is the extent of the map: [low left, low right, high left, high right] (arcsec)
            grid_spacing is the size of the map cells (arcsec)
        """
        self.row = row_map
        self.col = 1
        row = self.row
        col = self.col
        print(B.BData.receiver, B.pix_list, B.BData.map_coord)
        g = Grid(B.BData.receiver)
        invert_x = False
        if display_coord is None:
            pass
        elif display_coord == 0:
            try:
                map_x = B.BData.map_az
                map_y = B.BData.map_el
                label_x = 'Az'
                label_y = 'El'
                gx,gy = g.azel(B.BData.elev/180.*np.pi,B.BData.tracking_beam)
            except:
                display_coord = None
        elif display_coord == 1:
            try:
                map_x = B.BData.map_ra
                map_y = B.BData.map_dec
                label_x = 'Ra'
                label_y = 'Dec'
                gx,gy = g.radec(B.BData.elev/180.*np.pi,np.mean([np.mean(map_p) for map_p in B.BData.map_p]),B.BData.tracking_beam) # FIRST CUT
                invert_x = True
            except:
                display_coord = None
        elif display_coord == 2:
            try:
                map_x = B.BData.map_l
                map_y = B.BData.map_b
                label_x = 'L'
                label_y = 'B'
                gx,gy = g.latlon(B.BData.elev/180.*np.pi,np.mean([np.mean(map_p) for map_p in B.BData.map_p]),np.mean([np.mean(map_g) for map_g in B.BData.map_g]),B.BData.tracking_beam) # FIRST CUT
                invert_x = True
            except Exception as e:
                print(e)
                import traceback
                traceback.print_exc()
                display_coord = None
        elif display_coord == 11:
            try:
                map_x = B.BData.map_ra
                map_y = B.BData.map_dec
                label_x = 'Ra-interp'
                label_y = 'Dec-interp'
                gx,gy = g.radec(B.BData.elev/180.*np.pi,np.mean([np.mean(map_p) for map_p in B.BData.map_p]),B.BData.tracking_beam) # FIRST CUT
                invert_x = True
            except:
                display_coord = None
        elif display_coord == 21:
            try:
                map_x = B.BData.map_l
                map_y = B.BData.map_b
                label_x = 'Ra-astropy'
                label_y = 'Dec-astropy'
                gx,gy = g.radec(B.BData.elev/180.*np.pi,np.mean([np.mean(map_p) for map_p in B.BData.map_p]),B.BData.tracking_beam) # FIRST CUT
                invert_x = True
            except:
                display_coord = None
        else:
            display_coord = None

        if display_coord is None:
            map_x = B.BData.map_x
            map_y = B.BData.map_y
            label_x = 'Az'
            label_y = 'El'
            gx,gy = g.azel(B.BData.elev/180.*np.pi,B.BData.tracking_beam)

        map_data = B.BData.map_data
        if apply_grid_corrections:
            gxl = gx
            gyl = gy
        else:
            gxl = np.zeros(B.n_pix_list)
            gyl = np.zeros(B.n_pix_list)
            
        if not map_region:
            map_region = [0, 0, 0, 0]
            map_region[0] = 1.1*(map_x[0]).min()
            map_region[1] = 1.1*(map_x[0]).max()
            map_region[2] = 1.1*(map_y[0]).min()
            map_region[3] = 1.1*(map_y[0]).max()
            #np.set_printoptions(threshold=sys.maxsize)
            #print(map_x, map_y)
            print ('map_region', map_region)
        nx = int((map_region[1]-map_region[0])/grid_spacing+1)
        ny = int((map_region[3]-map_region[2])/grid_spacing+1)
        nx = ny = min(nx, ny)
        xi = np.linspace(map_region[0],map_region[1],nx)
        yi = np.linspace(map_region[2],map_region[3],ny)
        grid_x, grid_y = np.mgrid[map_region[0]:map_region[1]:complex(nx), map_region[2]:map_region[3]:complex(ny)]
        zi_sum = np.zeros((nx,ny))
        wi_sum = np.zeros((nx,ny))
        for i in range(B.n_pix_list):
            pixel = B.pix_list[i]
            if len(map_data) == 1:
                index = i
            else:
                index = B.BData.find_map_pixel_index(pixel)
            wdata = np.ones(len(map_data[index]))
            try: 
                print('trying scipy.interpolate.griddata')
                zi = interp.griddata((map_x[index]-gxl[pixel],map_y[index]-gyl[pixel]),map_data[index],(grid_x,grid_y),method='linear').T #,fill_value=map_data[index].min()).T
                wi = interp.griddata((map_x[index]-gxl[pixel],map_y[index]-gyl[pixel]),wdata,(grid_x, grid_y),method='linear',fill_value=wdata.min()).T
            except Exception as e:
                print(e)
                try:
                    zi = mlab.griddata(map_x[index]-gxl[pixel],map_y[index]-gyl[pixel],map_data[index],xi,yi,interp='linear')
                    wi = mlab.griddata(map_x[index]-gxl[pixel],map_y[index]-gyl[pixel],wdata,xi,yi,interp='linear')
                except:
                    zi = mlab.griddata(map_x[index]-gxl[pixel],map_y[index]-gyl[pixel],map_data[index],xi,yi,interp='nn')
                    wi = mlab.griddata(map_x[index]-gxl[pixel],map_y[index]-gyl[pixel],wdata,xi,yi,interp='nn')
            zi_sum = zi_sum + zi
            wi_sum = wi_sum + wi
        if with_matplotlib:
            pl.imshow(zi_sum/wi_sum,interpolation='bicubic',cmap=pl.cm.jet,origin='lower',extent=map_region)
            pl.plot(map_x[index],map_y[index])
            pl.axis('equal')
            pl.grid()
            pl.xlabel('%s (")'%label_x)
            pl.ylabel('%s (")'%label_y)
            if invert_x:
                pl.gca().invert_xaxis()
        isGood = np.zeros((B.n_pix_list))
        isGood = (B.peak_fit_status[:] != 5)
        az_map_offset = B.peak_fit_params[np.nonzero(isGood),1]
        az_map_hpbw = B.peak_fit_params[np.nonzero(isGood),2]
        el_map_offset = B.peak_fit_params[np.nonzero(isGood),3]
        el_map_hpbw = B.peak_fit_params[np.nonzero(isGood),4]
        textstr =           'Az Offset  %6.4f   HPBW  %6.4f'%(az_map_offset.mean()-np.mean(gx[B.pix_list]),az_map_hpbw.mean()) + '\n' 
        textstr = textstr + 'El Offset  %6.4f   HPBW  %6.4f'%(el_map_offset.mean()-np.mean(gy[B.pix_list]),el_map_hpbw.mean())
        map_coord = {0: 'Az-El', 1: 'Ra-Dec', 2: 'L-B'}
        textstr = textstr +'\n Map Coord %s'%(map_coord.get(B.BData.map_coord, 'Err'))
        try:
            if B.BData.xoffset != 0 or B.BData.yoffset != 0:
                textstr = textstr +', Offsets %0.2f %0.2f'%(B.BData.xoffset*B.BData.xlength, B.BData.yoffset*B.BData.ylength)
        except:
            pass
        textstr = textstr +('\n Tracking Beam %d'%B.BData.tracking_beam if B.BData.tracking_beam >=0 else '\nTracking Center')
        peak_amplitude = B.peak_fit_params[np.nonzero(isGood),0]
        peak_error = B.peak_fit_errors[np.nonzero(isGood),0]
        textstr = textstr +('   Peak Amplitude %0.2f (%0.2f)'%(peak_amplitude,peak_error))
        if with_matplotlib:
            pl.suptitle('ObsNum %d: %s %s %sGHz\n %s'%(B.obsnum,B.BData.receiver,B.BData.source,B.BData.line_rest_frequency,textstr)) 
            try:
                pl.tight_layout(rect=[0, 0.03, 1, 0.9])
            except:
                pass
            pl.colorbar()
        else:
            pxim = px.imshow(zi_sum/wi_sum, x=xi,y=yi, origin='lower', color_continuous_scale='jet')
            z = pxim.data[0]
            z['colorscale'] = 'jet'
            z['coloraxis'] = None
            #z['colorbar_len'] = 0.5
            #z['colorbar_y'] = 0.2
            self.fig.add_trace(z, row=row, col=col)
            self.fig.add_trace(go.Scatter(x=map_x[index],y=map_y[index],
                                          name='',
                                          showlegend=False),
                               row=row, col=col)
            self.fig.update_xaxes(
                range=[map_region[0], map_region[1]],
                showgrid=True,
                title='%s (")'%label_x,
                row=row, col=col,
            )
            self.fig.update_yaxes(
                # setting range causes plot to not be square
                #range=[map_region[2], map_region[3]],
                showgrid=True,
                title='%s (")'%label_y,
                scaleanchor='x',
                scaleratio=1,
                row=row, col=col,
            )
            title = 'ObsNum %d: %s %s %sGHz\n %s'%(B.obsnum,B.BData.receiver,B.BData.source,B.BData.line_rest_frequency,textstr)
            title='<br>'.join(title.split('\n'))
            idx = 0
            font_size = 16
            self.fig.layout.annotations[idx].update(text=title, font_size=font_size)
            self.fig.update_layout(
                margin=dict(t=200),
                title_x=0.5,
                #plot_bgcolor='white',
                font_size=16,
                height=800,
            )


    def map3d(self,B,map_region,grid_spacing,apply_grid_corrections=False):
        """ map aligns all maps on the sky using nominal grid and averages the maps
            B is BeamMap object with data and fits
            map_region is the extent of the map: [low left, low right, high left, high right] (arcsec)
            grid_spacing is the size of the map cells (arcsec)
        """
        print(B.BData.receiver)
        g = Grid(B.BData.receiver)
        if B.BData.map_coord == 0:
            gx,gy = g.azel(B.BData.elev/180.*np.pi,B.BData.tracking_beam)
        elif B.BData.map_coord == 1:
            gx,gy = g.radec(B.BData.elev/180.*np.pi,np.mean([np.mean(map_p) for map_p in B.BData.map_p]),B.BData.tracking_beam) # FIRST CUT
        elif B.BData.map_coord == 2:
            gx,gy = g.latlon(B.BData.elev/180.*np.pi,np.mean([np.mean(map_p) for map_p in B.BData.map_p]),np.mean([np.mean(map_g) for map_g in B.BData.map_g]),B.BData.tracking_beam) # FIRST CUT
        else:
            gx,gy = g.azel(B.BData.elev/180.*np.pi,B.BData.tracking_beam)

        if apply_grid_corrections:
            gxl = gx
            gyl = gy
        else:
            gxl = np.zeros(B.n_pix_list)
            gyl = np.zeros(B.n_pix_list)
        if not map_region:
            map_region = [0, 0, 0, 0]
            map_region[0] = 1.1*(B.BData.map_x[0]).min()
            map_region[1] = 1.1*(B.BData.map_x[0]).max()
            map_region[2] = 1.1*(B.BData.map_y[0]).min()
            map_region[3] = 1.1*(B.BData.map_y[0]).max()
            print (map_region)
        nx = int((map_region[1]-map_region[0])/grid_spacing+1)
        ny = int((map_region[3]-map_region[2])/grid_spacing+1)
        nx = ny = min(nx, ny)
        xi = np.linspace(map_region[0],map_region[1],nx)
        yi = np.linspace(map_region[2],map_region[3],ny)
        grid_x, grid_y = np.mgrid[map_region[0]:map_region[1]:complex(nx), map_region[2]:map_region[3]:complex(ny)]
        zi_sum = np.zeros((nx,ny))
        wi_sum = np.zeros((nx,ny))
        with_fill_value = True
        for i in range(B.n_pix_list):
            pixel = B.pix_list[i]
            if len(B.BData.map_data) == 1:
                index = i
            else:
                index = B.BData.find_map_pixel_index(pixel)
            wdata = np.ones(len(B.BData.map_data[index]))
            try:
                print('trying scipy.interpolate.griddata')
                if with_fill_value:
                    zi = interp.griddata((B.BData.map_x[index]-gxl[pixel],B.BData.map_y[index]-gyl[pixel]),B.BData.map_data[index],(grid_x,grid_y),method='linear',fill_value=B.BData.map_data[index].min()).T
                    wi = interp.griddata((B.BData.map_x[index]-gxl[pixel],B.BData.map_y[index]-gyl[pixel]),wdata,(grid_x, grid_y),method='linear',fill_value=wdata.min()).T
                else:
                    zi = interp.griddata((B.BData.map_x[index]-gxl[pixel],B.BData.map_y[index]-gyl[pixel]),B.BData.map_data[index],(grid_x,grid_y),method='linear').T
                    wi = interp.griddata((B.BData.map_x[index]-gxl[pixel],B.BData.map_y[index]-gyl[pixel]),wdata,(grid_x, grid_y),method='linear').T
                    
            except Exception as e:
                print(e)
                try:
                    zi = mlab.griddata(B.BData.map_x[index]-gxl[pixel],B.BData.map_y[index]-gyl[pixel],B.BData.map_data[index],xi,yi,interp='linear')
                    wi = mlab.griddata(B.BData.map_x[index]-gxl[pixel],B.BData.map_y[index]-gyl[pixel],wdata,xi,yi,interp='linear')
                except:
                    zi = mlab.griddata(B.BData.map_x[index]-gxl[pixel],B.BData.map_y[index]-gyl[pixel],B.BData.map_data[index],xi,yi,interp='nn')
                    wi = mlab.griddata(B.BData.map_x[index]-gxl[pixel],B.BData.map_y[index]-gyl[pixel],wdata,xi,yi,interp='nn')
            zi_sum = zi_sum + zi
            wi_sum = wi_sum + wi

        zi = zi_sum/wi_sum
        fig = pl.figure()
        ax = fig.gca(projection='3d')
        xm,ym = np.meshgrid(xi, yi)
        # this breaks when grid has nan
        with_norm_colormap = False
        if with_norm_colormap:
            norm =  mcolors.Normalize(vmin=np.min(zi), vmax=np.max(zi))
            my_col = pl.cm.jet(norm(zi))
            surf = ax.plot_surface(xm, ym, zi, rstride=1, cstride=1, facecolors=my_col, linewidth=1, antialiased=True)
            m = pl.cm.ScalarMappable(cmap=pl.cm.jet, norm=norm)
            m.set_array([])
            pl.colorbar(m)
        else:
            surf = ax.plot_surface(xm, ym, zi, rstride=1, cstride=1, cmap=pl.cm.jet, linewidth=1, antialiased=True)
            pl.colorbar(surf)
        pl.xlabel('Azimuth (")')
        pl.ylabel('Elevation (")')
        isGood = np.zeros((B.n_pix_list))
        isGood = (B.peak_fit_status[:] != 5)
        az_map_offset = B.peak_fit_params[np.nonzero(isGood),1]
        el_map_offset = B.peak_fit_params[np.nonzero(isGood),3]
        textstr =           'Az Offset  %6.4f'%(az_map_offset.mean()-np.mean(gx[B.pix_list])) + '\n' 
        textstr = textstr + 'El Offset  %6.4f'%(el_map_offset.mean()-np.mean(gy[B.pix_list]))
        pl.suptitle('ObsNum %d: %s %s %sGHz\n %s'%(B.obsnum,B.BData.receiver,B.BData.source,B.BData.line_rest_frequency,textstr)) 
        try:
            pl.tight_layout(rect=[0, 0.03, 1, 0.9])
        except:
            pass


    def align_plot(self,B,show_id=True):
        """ align_plot shows the residuals of peak positions after nominal grid offsets removed
            B is BeamMap object with data and fits
            show_id=True will print the pixel id on the map next to the peak position
        """
        g = Grid(B.BData.receiver)
        gx,gy = g.azel(B.BData.elev/180.*np.pi,B.BData.tracking_beam)
        ang = g.rotation - B.elev/180.*np.pi
        x = np.zeros(B.n_pix_list)
        y = np.zeros(B.n_pix_list)
        for i in range(B.n_pix_list):
            ipix = B.pix_list[i]
            index = B.BData.find_map_pixel_index(ipix)
            x[index] = B.peak_fit_params[index,1] - gx[ipix]
            y[index] = B.peak_fit_params[index,3] - gy[ipix]
        mx = np.mean(x)
        my = np.mean(y)
        pl.plot(x-mx,y-my,'k*',label='%d'%(B.obsnum))
        if show_id:
            for i in range(B.n_pix_list):
                ipix = B.pix_list[i]
                index = B.BData.find_map_pixel_index(ipix)
                pl.text(x[index]-mx,y[index]-my,'%d'%(B.pix_list[i]))
        pl.xlabel('Azimuth Offset')
        pl.ylabel('Elevation Offset')
        pl.suptitle('Aligned Beam Position %d'%(B.obsnum))
        pl.axis('equal')
        pl.axis([-4,4,-4,4])
        pl.grid('on')


    def print_pixel_fits(self,B):
        """ print_pixel_fits makes a printout of all the individual fits to pixels in the pix_list
            B is BeamMap object with data and fits
        """
        print('-------------------------------------')
        print('Fits to Individual Beams: %d El=%6.2f'%(B.obsnum,B.elev))
        print('-- ------        ------        ------        ------        ------        ------')
        print('PX   Peak         AzOFF        AzHPBW         ElOFF        ElHPBW        Success')
        if B.BData.cal_flag == True:
            print('      (K)          (")           (")           (")           (")')
        else:
            print('      (V)          (")           (")           (")           (")')
            
        print('-- ------        ------        ------        ------        ------        ------')
        for i in range(B.n_pix_list):
            ipix = B.pix_list[i]
            print('%02d %6.2f %4.2f   %6.2f %4.2f   %6.2f %4.2f   %6.2f %4.2f   %6.2f %4.2f  %d'%(B.pix_list[i],
                                                                                                  B.peak_fit_params[i,0],B.peak_fit_errors[i,0],
                                                                                                  B.peak_fit_params[i,1],B.peak_fit_errors[i,1],
                                                                                                  B.peak_fit_params[i,2],B.peak_fit_errors[i,2],
                                                                                                  B.peak_fit_params[i,3],B.peak_fit_errors[i,3],
                                                                                                  B.peak_fit_params[i,4],B.peak_fit_errors[i,4],
                  B.peak_fit_status[i]))
        print('-- ------        ------        ------        ------        ------        ------')

    def print_grid_fit(self,B):
        """ print_grid_fit prints out the best fit to the grid of pixel positions
            B is BeamMap object with data and fits
        """
        print('-----------------------------------------')
        print('Array Geometry: ObsNum=%d Elev=%6.2f'%(B.BData.obsnum,B.BData.elev))
        print('-----------------------------------------')
        print('AZ Offset = %6.1f (%4.1f) arcsec'%(B.grid_param[0],B.grid_param_error[0]))
        print('EL Offset = %6.1f (%4.1f) arcsec'%(B.grid_param[1],B.grid_param_error[1]))
        print('SPACING   = %6.2f (%4.2f) arcsec'%(B.grid_param[2],B.grid_param_error[2]))
        print('GRID THETA= %6.2f (%4.2f) deg'%(B.grid_param[3],B.grid_param_error[3]))
        print('GRID ROT  = %6.1f (%4.2f) deg'%(B.grid_param[3]+B.elev,B.grid_param_error[3]))
        print('FIT RMS   = %6.2f arcsec'%(B.grid_rms))
        print('-----------------------------------------')

    def plot_tsys_levels(self,ICal):
        if with_matplotlib:
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
            if with_matplotlib:
                #colors = pl.rcParams["axes.prop_cycle"]()
                colors = pl.rcParams['axes.prop_cycle']
                colors = [c['color'] for c in colors]
            else:
                self.set_figure(figure=1)
                self.open_figure(nrows=nrows, ncols=ncols)
            plot_order = [1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16]
            for ipix in range(ICal.npix):
                if ncols == 1:
                    ipix1 = ipix+1
                    row = ipix1
                    col = 1
                else:
                    ipix1 = plot_order[(ipix%len(plot_order))]+int(ipix/len(plot_order))*len(plot_order) #ipix+1)
                    pixel_id = ipix
                    row = (pixel_id%ncols)+1+int(pixel_id/16)*4
                    col = int((pixel_id%16)/ncols)+1
                    #print(ipix, ipix1, pixel_id, row, col)
                if with_matplotlib:
                    ax = pl.subplot(nrows, ncols, ipix1)
                    ax.tick_params(axis='both', which='major', labelsize=6)
                    ax.tick_params(axis='both', which='minor', labelsize=6)
                label = '%2d %6.1f'%(ipix,ICal.tsys[ipix])
                legend.append(label)
                y = ICal.level[:,ipix]
                #color = next(colors)['color']
                plot_scale = np.mean(ICal.level[:,ipix])+np.min(ICal.level[:,ipix])
                if with_matplotlib:
                    color = colors[ipix%len(colors)]
                    ax.plot(x,y,'.', color=color)
                    ax.text(x[-1]/2, plot_scale/2, label, verticalalignment='center', horizontalalignment='center', zorder=10)
                    if False and plot_scale != 0:
                        ax.set_ylim(0, plot_scale * 1.1)
                else:
                    self.fig.add_trace(
                        go.Scatter(
                            x=x, y=y,
                            name=label,
                            mode="markers",
                            showlegend=False),
                        row=row, col=col,
                        )
                    if False:
                        self.fig.update_xaxes(showticklabels=False,
                                              row=row, col=col)
                        self.fig.update_yaxes(showticklabels=False,
                                              row=row, col=col)
                    idx = (row-1)*ncols+(col-1)
                    title = label
                    font_size = 14
                    self.fig.layout.annotations[idx].update(text=title, font_size=font_size)
            if with_matplotlib:
                pl.suptitle("TSys %s ObsNum: %d"%(ICal.receiver,ICal.obsnum))
            else:
                title = "TSys %s ObsNum: %d"%(ICal.receiver,ICal.obsnum)
                font_size = 16
                width = max(400, ncols*300)
                height = max(200, nrows*200)
                self.fig.update_layout(title=title, title_x=0.5, font_size=font_size, width=width, height=height)
        if True:
            return
    
class SpecViewer(PlotlyViewer):
    def __init__(self):
        PlotlyViewer.__init__(self)
    
class SpecCalViewer(PlotlyViewer):
    def __init__(self):
        PlotlyViewer.__init__(self)
    
class BeamMapView(PlotlyViewer):
    def __init__(self):
        PlotlyViewer.__init__(self)

class TsysView(PlotlyViewer):
    def __init__(self):
        PlotlyViewer.__init__(self)

class SpecBankViewer(PlotlyViewer):
    def __init__(self):
        PlotlyViewer.__init__(self)

def merge_png(image_files, newfile):
    if with_matplotlib:
        from merge_png import merge_png
        merge_png(image_files, newfile)
    else:
        if True: return
        import plotly.io as pio
        rows = len(image_files)
        rows = 2
        cols = 1
        print('rows =', rows, 'cols = ', cols)
        fig = make_subplots(
            shared_xaxes = False,
            rows=rows,
            cols=cols,
            subplot_titles=[str(x) for x in range(rows)],
        )
        for i, f in enumerate(image_files):
            row = i+1
            col = 1
            fig1 = pio.read_json(f.replace('.png', '.json'))
            print('row =', row)
            for d in fig1.data:
                fig.add_trace(d,
                              row=row,
                              col=col)
            print(fig1.layout)
            idx = (row-1)*cols+(col-1)
            for annot in fig1.layout['annotations']:
                annot['y'] = 1-i*.625
                fig.layout.annotations[idx].update(annot)
            xa = fig1.layout['xaxis']
            if i == 0:
                xa['range'] = [-131, 129]
            ya = fig1.layout['yaxis']
            #fig.update_xaxes(xa, selector={'text': str(row)}, row=row, col=col)
            #fig.update_yaxes(ya, selector={'text': str(row)}, row=row, col=col)
            if i == 0:
                xa['anchor'] = 'y'
                ya['anchor'] = 'x'
                ya['domain'] = [.625, 1]
                fig.update_layout(xaxis=xa, yaxis=ya)
            elif i == 2:
                xa['anchor'] = 'y2'
                #xa['range'] = [-131, 129]
                ya['anchor'] = 'x2'
                ya['domain'] = [0, 1-.625]
                fig.update_layout(xaxis2=xa, yaxis2=ya)
            legend = fig1.layout['legend']
            if legend['x'] != None:
                fig.update_layout(legend=legend)
            shapes = fig1.layout['shapes']
            fig.update_layout(shapes=shapes)
        fig.write_html(newfile.replace('.png', '.html'))
        fig.show()

        print('combined')
        print(fig.layout)

        try:
            print('orig')
            figs = pio.read_json('pv_111695.json')
            print(figs.layout)
        except:
            pass
