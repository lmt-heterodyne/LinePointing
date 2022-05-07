import numpy
import math
import matplotlib.pyplot as pl
try:
  import matplotlib.gridspec as gridspec
except Exception as e:
  print('m2fit_viewer', e)


class m2fit_viewer():
    """ base class for viewing fit data
    """
    def __init__(self,figure=1):
        self.figure = figure

    def set_figure(self,figure):
        self.figure = figure

    def open_figure(self):
        """ opens the figure window
        """
        pl.figure(self.figure)
        pl.clf()

    def close_figure(self):
        """ closes the figure window
        """
        pl.close(self.figure)

    def ion(self):
        """ opens the figure window
        """
        pl.ion()

    def plot_fits(self,paramfit,obsNumArg=False,line_stats_all=[],plot_axis=[-200,200,-5,15]):
        """Plots graphs of all data and fits.

        paramfit is the input param fit instance with the results.
        figno specifies the figure to receive the plot.
        """
        if paramfit.m2pos == 0:
            self.tlabel = 'M2.Z Offset'
            self.xlabel = 'M2.Z Offset (mm)'
            prange = numpy.arange(-7,7.1,.1)
        elif paramfit.m2pos == 1:
            self.tlabel = 'M2.Y Offset'
            self.xlabel = 'M2.Y Offset (mm)'
            prange = numpy.arange(-36,36.1,.1)
        elif paramfit.m2pos == 2:
            self.tlabel = 'M2.X Offset'
            self.xlabel = 'M2.X offset (mm)'
            prange = numpy.arange(-36,36.1,.1)
        elif paramfit.m2pos == 3:
            self.tlabel = 'M1.ZernikeC0'
            self.xlabel = 'M1.ZernikeC0 (um)'
            prange = numpy.arange(-1000,1000,10)
        else:
            self.tlabel = 'Error: Nothing is changing'
            self.xlabel = 'Offset'
            prange = numpy.arange(-1,1.1,.1)
        pl.xlabel(self.xlabel)
        pl.ylabel('Intensity')
        prange = numpy.arange(min(paramfit.m2_position)-.1,max(paramfit.m2_position)+.1,.1)
        prange = numpy.arange(min(paramfit.scans_xpos)-.1,max(paramfit.scans_xpos)+.1,.1)
        
        plot_order = [1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16];

        # outer grid
        if paramfit.n == 1:
            nrows = ncols = 1
        else:
            nrows = ncols = 4

        # inner grid
        if len(line_stats_all) != 0 and len(line_stats_all[0]) != 0:
            num_sub_cols = len(line_stats_all)
            num_sub_rows = num_sub_cols+1
            # find axis for spectra plots
            line_stats = line_stats_all[0][0]
            vrange = numpy.where(numpy.logical_and(line_stats.v >= plot_axis[0], line_stats.v <= plot_axis[1]))
            plot_axis[2] = line_stats.spectrum[vrange].min()*1.1
            plot_axis[3] = line_stats.spectrum[vrange].max()*1.1
            for index in range(paramfit.n):
                for i in range(num_sub_cols):
                    if index == 0 and i == 0: continue
                    line_stats = line_stats_all[i][index]
                    min_ps = line_stats.spectrum[vrange].min()*1.1
                    max_ps = line_stats.spectrum[vrange].max()*1.1
                    if min_ps < plot_axis[2]: plot_axis[2] = min_ps
                    if max_ps > plot_axis[3]: plot_axis[3] = max_ps
        else:
            num_sub_rows = num_sub_cols = 0

        if num_sub_rows > 0 and num_sub_cols > 0:
          outer_grid = gridspec.GridSpec(nrows, ncols)

        for index in range(paramfit.n):
            if paramfit.n == 1:
                pixel_index = 0
                hspace = None
            else:
                pixel_index = plot_order[pixel_index]-1
                hspace = 1.0
            if(math.isnan(paramfit.result_relative[index])):
                continue
            model = (paramfit.parameters[index,0]
                     + paramfit.parameters[index,1]*prange
                     + paramfit.parameters[index,2]*prange*prange
                     )
            if num_sub_rows > 0 and num_sub_cols > 0:
                inner_grid = gridspec.GridSpecFromSubplotSpec(num_sub_rows, num_sub_cols, subplot_spec=outer_grid[pixel_index], hspace=hspace)
                ax = pl.subplot(inner_grid[:-1,:])
            else:
                ax = pl.subplot(111)
            
            if paramfit.m2pos >= 0:
                ax.plot(paramfit.m2_position,paramfit.data[:,index],'o')
                ax.plot(prange,model,'r')
                pl.axhline(y=.5*numpy.max(paramfit.data[:,index]), color='b')
                if paramfit.status < 0:
                  props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                  pl.text(numpy.min(paramfit.m2_position)+0.1*(numpy.max(paramfit.m2_position)-numpy.min(paramfit.m2_position)), 0.5*numpy.max(paramfit.data[:,index]), paramfit.msg, bbox=props, color='red')
                try:
                  pl.tick_params(axis='both',which='major',labelsize=6)
                except:
                  pass
                for i in range(num_sub_cols):
                    line_stats = line_stats_all[i][index]
                    ax = pl.subplot(inner_grid[-1,i])
                    pl.gca().get_xaxis().set_visible(False);
                    pl.gca().get_yaxis().set_visible(False);
                    pl.plot(line_stats.v[vrange],line_stats.spectrum[vrange])
                    ax.axis(plot_axis)
            try:
              ax.tick_params(axis='both',which='major',labelsize=6)
            except:
              pass
            try:
              pl.tight_layout(rect=[0, 0.03, 1, 0.95])
            except:
              pass
            if obsNumArg == False:
                titleObsNum = paramfit.obsnum
            else:
                titleObsNum = obsNumArg
            pl.suptitle('ObsNum: %s %s %s %s\n%s'%(titleObsNum,paramfit.obspgm,paramfit.receiver.strip(),paramfit.source.strip(),self.tlabel))

    def plot_focus_model_fit(self,paramfit,obsNumArg=False):
        """Plots data and focus model fit."""
        
        result_relative = paramfit.result_relative
        brange = numpy.arange(-0.5*(paramfit.n-1),0.5*(paramfit.n-1)+1,1)
        brange = numpy.arange(0,paramfit.n,1)
        the_model = paramfit.relative_focus_fit+(paramfit.focus_slope)*brange
        pl.plot(brange,result_relative,'o')
        pl.plot(brange,the_model,'r')
        if len(brange) == 1:
            xpos = brange[0]+0.01
            ypos = result_relative*1.01
        else:
            xpos = brange[0]+.5
            ypos = result_relative.max()-0.2*(result_relative.max()-result_relative.min())
        if paramfit.m2pos == 0:
            self.tlabel = 'M2.Z Offset'
            self.ylabel = 'M2.Z Offset (mm)'
            prange = numpy.arange(-7,7.1,.1)
        elif paramfit.m2pos == 1:
            self.tlabel = 'M2.Y Offset'
            self.ylabel = 'M2.Y Offset (mm)'
            prange = numpy.arange(-36,36.1,.1)
        elif paramfit.m2pos == 2:
            self.tlabel = 'M2.X Offset'
            self.ylabel = 'M2.X Offset (mm)'
            prange = numpy.arange(-36,36.1,.1)
        elif paramfit.m2pos == 3:
            self.tlabel = 'M1.ZernikeC0'
            self.ylabel = 'M1.ZernikeC0 (um)'
            prange = numpy.arange(-1000,1000,10)
        else:
            self.tlabel = 'Error: Nothing is changing'
            self.ylabel = 'Offset'
            prange = numpy.arange(-36,36.1,.1)
        pl.ylabel(self.ylabel)
        if obsNumArg == False:
            titleObsNum = paramfit.obsnum
        else:
            titleObsNum = obsNumArg
        pl.suptitle('ObsNum: %s %s %s %s\n%s'%(titleObsNum,paramfit.obspgm,paramfit.receiver.strip(),paramfit.source.strip(),self.tlabel))
        if paramfit.m2pos == 2:
            fitype = 'M2.X'
        elif paramfit.m2pos == 1:
            fitype = 'M2.Y'
        elif paramfit.m2pos == 0:
            fitype = 'M2.Z'
        elif paramfit.m2pos == 3:
            fitype = 'M1.ZernikeC0'
        else:
            fitype = 'Error'
        textstr =           'Relative '+fitype+':   ' +str(round(paramfit.relative_focus_fit,4)) + '\n' 
        textstr = textstr + fitype+' Error:         ' +str(round(paramfit.focus_error,4)) + '\n' 
        textstr = textstr + fitype+' Slope:       ' +str(round(paramfit.focus_slope,4)) + '\n' 
        textstr = textstr + 'Absolute '+fitype+':  ' +str(round(paramfit.absolute_focus_fit,4)) + '\n' 
        textstr = textstr + 'Fit RMS:                ' +str(round(paramfit.fit_rms,4))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        pl.text(xpos, ypos, textstr, bbox=props, color='red')
        try:
          pl.tick_params(axis='both',which='major',labelsize=6)
        except:
          pass
        try:
          pl.tight_layout(rect=[0, 0.03, 1, 0.95])
        except:
          pass
