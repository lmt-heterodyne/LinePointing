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

    def close_figure(self, figure=None):
        """ closes the figure window
        """
        if figure is not None:
          pl.close(figure)

    def save_figure(self, fname):
        pl.savefig(fname)#, bbox_inches='tight')

    def show(self):
        pl.show()

    def ion(self):
        """ opens the figure window
        """
        pl.ion()

    def plot_fits(self,paramfit,obsNumArg=False,line_stats_all=[],plot_axis=[-200,200,-5,15],use_gaus=False,row_id=None,col_id=None,names=None):
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
            self.tlabel = paramfit.msg #'Error: Nothing is changing'
            self.xlabel = 'Offset'
            prange = numpy.arange(-1,1.1,.1)
        #pl.xlabel(self.xlabel)
        #pl.ylabel('Intensity')
        prange = numpy.arange(min(paramfit.m2_position)-.1,max(paramfit.m2_position)+.1,.1)
        if len(paramfit.scans_xpos) > 0:
          prange = numpy.arange(min(paramfit.scans_xpos)-.1,max(paramfit.scans_xpos)+.1,.1)
        else:
          prange = numpy.arange(0)
        
        plot_order = [1,5,9,13,2,6,10,14,3,7,11,15,4,8,12,16];

        # outer grid
        if paramfit.n == 1:
            nrows = ncols = 1
            row0 = 0
        else:
            nrows = 1
            ncols = paramfit.n
            row0 = 0
            if paramfit.receiver == 'RedshiftReceiver':
                nrows = len(set(row_id[0]))
                ncols = len(set(col_id[0]))
                row0 = int(min(set(row_id[0])))
                ncols = 6
            elif paramfit.receiver == 'Toltec':
                nrows = 3
                ncols = 1
            elif paramfit.receiver == 'Sequoia':
                nrows = 4
                ncols = 4
        print('rows/cols =' , row_id, col_id, nrows, ncols, row0, paramfit.n, paramfit.receiver)

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

        if False and nrows > 0 and ncols > 0:
          outer_grid = gridspec.GridSpec(nrows, ncols)

        for index in range(paramfit.n):
            if paramfit.n == 1:
                pixel_index = 0
                hspace = None
            else:
              if paramfit.receiver == 'Sequoia':
                pixel_index = plot_order[index]-1
              elif paramfit.receiver == 'RedshiftReceiver':
                pixel_index = (int(row_id[0][index])-row0)*6+int(col_id[0][index])
              else:
                pixel_index = index
                hspace = 1.0
            if(math.isnan(paramfit.result_relative[index])):
                continue
            model = (paramfit.parameters[index,0]
                     + paramfit.parameters[index,1]*prange
                     + paramfit.parameters[index,2]*prange*prange
                     )
            if use_gaus == True:
                def gaus(x,a,x0,sigma):
                    return a*numpy.exp(-(x-x0)**2/(2*sigma**2))
                model = gaus(prange, *paramfit.parameters[index])
            if num_sub_rows > 0 and num_sub_cols > 0:
                inner_grid = gridspec.GridSpecFromSubplotSpec(num_sub_rows, num_sub_cols, subplot_spec=outer_grid[pixel_index], hspace=hspace)
                ax = pl.subplot(inner_grid[:-1,:])
            else:
                ax = pl.subplot(nrows, ncols, pixel_index+1)#outer_grid[pixel_index])
            
            if paramfit.m2pos >= 0:
                ax.plot(paramfit.m2_position,paramfit.data[:,index],'o')
                ax.plot(prange,model,'r')
                for i in range(len(paramfit.m2_position)):
                  ax.text(paramfit.m2_position[i],paramfit.data[i,index],
                          "%d"%paramfit.obsnums[i])
                if use_gaus == False:
                  ax.axhline(y=.5*numpy.max(paramfit.data[:,index]), color='b')
                if len(paramfit.status) == paramfit.n and paramfit.status[index] < 0:
                  props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                  ax.text(numpy.min(paramfit.m2_position)+0.1*(numpy.max(paramfit.m2_position)-numpy.min(paramfit.m2_position)), 0.5*numpy.max(paramfit.data[:,index]), paramfit.msg, bbox=props, color='red')
                if type(names) == list and type(names[0]) == list:
                  props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                  ax.text(numpy.min(paramfit.m2_position)+0.1*(numpy.max(paramfit.m2_position)-numpy.min(paramfit.m2_position)), 0.3*numpy.max(paramfit.data[:,index]), names[0][index], bbox=props, color='red')
                try:
                  ax.tick_params(axis='both',which='major',labelsize=6)
                except:
                  pass
                try:
                  ax.tick_params(axis='x', labelrotation=90)
                except:
                  print('cant rotate x-ticks')
                  pass
                
                for i in range(num_sub_cols):
                    if len(line_stats_all) == 0:
                      break
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
              ax.tick_params(axis='x', labelrotation=90)
            except:
              print('cant rotate x-ticks')
              pass
            try:
              _ = None
              #pl.tight_layout(rect=[0, 0.03, 1, 0.95])
            except:
              pass
        if obsNumArg == False:
          titleObsNum = paramfit.obsnum
        else:
          titleObsNum = obsNumArg
        pl.suptitle('ObsNum: %s %s %s %s\n%s'%(titleObsNum,paramfit.obspgm,paramfit.receiver.strip(),paramfit.source.strip(),self.tlabel))

    def plot_focus_model_fit(self,paramfit,obsNumArg=False,row_id=None,col_id=None,names=None):
        """Plots data and focus model fit."""
        
        if paramfit.receiver == 'RedshiftReceiver':
            M = paramfit
            band_order = [0,2,1,3,5,4]
            freq = [75.703906,82.003906,89.396094,94.599906,100.899906,108.292094]
            freq_0 = (freq[0]+freq[5])/2.
            d_freq = (freq[5]-freq_0)
            brange = numpy.arange(-1.2,1.3,.1)
            f = freq_0+brange*d_freq
            band_freq = []
            result_relative = []
            for index in range(M.n):
                if(math.isnan(M.result_relative[index])):
                    continue
                band_freq.append(freq[band_order[int(col_id[0][index])]])
                result_relative.append(M.result_relative[index])
            band_freq = numpy.array(band_freq)
            result_relative = numpy.array(result_relative)
            pl.plot(band_freq,result_relative,'o')
            the_model = M.relative_focus_fit+M.focus_slope*brange
            pl.plot(f,the_model,'r')
            pl.xlabel('Frequency (GHz)')
            xpos = 93
            ypos = result_relative.max()-0.3*(result_relative.max()-result_relative.min())
        else:
            result_relative = paramfit.result_relative
            brange = numpy.arange(-0.5*(paramfit.n-1),0.5*(paramfit.n-1)+1,1)
            #brange = numpy.arange(0,paramfit.n,1)
            the_model = paramfit.relative_focus_fit+(paramfit.focus_slope)*brange
            pl.plot(brange,result_relative,'o')
            pl.plot(brange,the_model,'r')
            if type(names) == list and type(names[0]) == list:
              pl.xticks(brange, names[0])
            if len(brange) == 1:
                xpos = brange[0]+0.01
                ypos = result_relative*1.01
            else:
                xpos = brange[0]+.25
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
            self.tlabel = paramfit.msg #'Error: Nothing is changing'
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
        if paramfit.receiver == 'Toltec':
          textstr =           'Relative '+fitype+':   ' +str(['%.4f'%round(x,4) for x in result_relative]).replace("'",  "") + '\n'
        else:
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
          _ = None
          #pl.tight_layout(rect=[0, 0.03, 1, 0.95])
          xlim = pl.gca().get_xlim()
          ylim = pl.gca().get_ylim()
          dx = xlim[1]-xlim[0]
          dy = ylim[1]-ylim[0]
          xlim = [xlim[0]-0.1*dx, xlim[1]+0.1*dx]
          ylim = [ylim[0]-0.1*dy, ylim[1]+0.1*dy]
          pl.gca().set_xlim(xlim)
          pl.gca().set_ylim(ylim)
        except Exception as e:
          print(e)
          pass
