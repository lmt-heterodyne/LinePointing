""" Module for viewing BeamMap's

classes: BeamMapView
uses: BeamMap, Grid
author: FPS, KS
date: May 2018
changes: 
 
"""
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

class BeamMapView():
    def __init__(self,figure=1):
        """ BeamMapView contains methods to display BeamMap results and Map Data
            figure is the number of the figure for drawing plots
            Notes: we do plotting in interactive mode
                   map units are arcsec
        """
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

    def show_fit(self,B,ipix):
        """ plots the scan of data and the best fit model.
            B is the BeamMap with data and model
            ipix is the pixel id for plotting
        """
        index = B.BData.find_map_pixel_index(ipix)
        pix_list_index = B.find_pixel_index(ipix)
        model = np.array(B.model[pix_list_index])
        pl.plot(B.BData.map_data[index],'k',label='Data')
        pl.plot(model,'r',label='Model')
        offset = B.peak_fit_params[pix_list_index,0]/2.0
        pl.plot(B.BData.map_data[index]-model-offset,'g',label='Resid')
        pl.plot(-offset*np.ones(len(B.BData.map_data[index])),'b')
        pl.xlabel('Sample')
        pl.legend()
        pl.suptitle('Model Fit for %d Pixel:%d'%(B.obsnum,ipix))

    def show_peaks(self,B,apply_grid_corrections=False,show_map_ids=True,show_map_points=-1):
        """ plots and identifies the peak positions in the BeamMap fits
            B is the BeamMap object with the data and fit results
            apply_grid_corrections=True will use the nominal grid to offset positions
        """
        g = Grid(B.BData.receiver)
        gx,gy = g.azel(B.BData.elev/180.*np.pi,B.BData.tracking_beam)
        
        if apply_grid_corrections:
            gxl = gx[B.pix_list]
            gyl = gy[B.pix_list]
        else:
            gxl = np.zeros(B.n_pix_list)
            gyl = np.zeros(B.n_pix_list)
        pl.plot(B.peak_fit_params[:,1]-gxl,B.peak_fit_params[:,3]-gyl,'ko')
        if show_map_ids:
            for i in range(B.n_pix_list):
                pl.text(B.peak_fit_params[i,1]-gxl[i],B.peak_fit_params[i,3]-gyl[i],'%d'%(B.pix_list[i]))

        if show_map_points >= 0:
            pixel_index = B.BData.find_map_pixel_index(show_map_points)
            plen = len(B.BData.map_x[pixel_index])
            xlen = ylen = int(math.sqrt(plen))
            for i in range(xlen):
                for j in range(ylen):
                    index = i+j*xlen
                    pl.plot(B.BData.map_x[pixel_index][index]-gxl[pixel_index],B.BData.map_y[pixel_index][index]-gyl[pixel_index],'.r')

    def sanchez_map(self,B,map_region,grid_spacing):
        """ makes a "Sanchez Plot" of the beams on the sky
            B is BeamMap object with data and fits
            map_region is the extent of the map: [low left, low right, high left, high right] (arcsec)
            grid_spacing is the size of the map cells (arcsec)
        """
        g = Grid(B.BData.receiver)
        gx,gy = g.azel(B.BData.elev/180.*np.pi,B.BData.tracking_beam)
        nx = int((map_region[1]-map_region[0])/grid_spacing)+1
        ny = int((map_region[3]-map_region[2])/grid_spacing)+1
        xi = np.linspace(map_region[0],map_region[1],nx)
        yi = np.linspace(map_region[2],map_region[3],ny)
        grid_x, grid_y = np.mgrid[map_region[0]:map_region[1]:complex(nx), map_region[2]:map_region[3]:complex(ny)]
        zi_sum = np.zeros((nx,ny))
        for i in range(B.n_pix_list):
            pixel = B.pix_list[i]
            index = B.BData.find_map_pixel_index(pixel)
            try:
                print('trying scipy.interpolate.griddata')
                zi = interp.griddata((B.BData.map_x[index],B.BData.map_y[index]),B.BData.map_data[index],(grid_x,grid_y),method='linear').T
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
        print(B.BData.receiver, B.pix_list, B.BData.map_coord)
        g = Grid(B.BData.receiver)
        invert_x = False
        if display_coord is None:
            map_x = B.BData.map_x
            map_y = B.BData.map_y
            label_x = 'Az'
            label_y = 'El'
            gx,gy = g.azel(B.BData.elev/180.*np.pi,B.BData.tracking_beam)
        elif display_coord == 0:
            map_x = B.BData.map_az
            map_y = B.BData.map_el
            label_x = 'Az'
            label_y = 'El'
            gx,gy = g.azel(B.BData.elev/180.*np.pi,B.BData.tracking_beam)
        elif display_coord == 1:
            map_x = B.BData.map_ra
            map_y = B.BData.map_dec
            label_x = 'Ra'
            label_y = 'Dec'
            gx,gy = g.radec(B.BData.elev/180.*np.pi,np.mean([np.mean(map_p) for map_p in B.BData.map_p]),B.BData.tracking_beam) # FIRST CUT
            invert_x = True
        elif display_coord == 2:
            map_x = B.BData.map_l
            map_y = B.BData.map_b
            label_x = 'L'
            label_y = 'B'
            gx,gy = g.latlon(B.BData.elev/180.*np.pi,np.mean([np.mean(map_p) for map_p in B.BData.map_p]),np.mean([np.mean(map_g) for map_g in B.BData.map_g]),B.BData.tracking_beam) # FIRST CUT
            invert_x = True
        elif display_coord == 11:
            map_x = B.BData.map_ra
            map_y = B.BData.map_dec
            label_x = 'Ra-interp'
            label_y = 'Dec-interp'
            gx,gy = g.radec(B.BData.elev/180.*np.pi,np.mean([np.mean(map_p) for map_p in B.BData.map_p]),B.BData.tracking_beam) # FIRST CUT
            invert_x = True
        elif display_coord == 21:
            map_x = B.BData.map_l
            map_y = B.BData.map_b
            label_x = 'Ra-astropy'
            label_y = 'Dec-astropy'
            gx,gy = g.radec(B.BData.elev/180.*np.pi,np.mean([np.mean(map_p) for map_p in B.BData.map_p]),B.BData.tracking_beam) # FIRST CUT
            invert_x = True
        else:
            map_x = B.BData.map_x
            map_y = B.BData.map_y
            label_x = 'Az'
            label_y = 'El'
            gx,gy = g.azel(B.BData.elev/180.*np.pi,B.BData.tracking_beam)

        if apply_grid_corrections:
            if True or len(B.BData.map_data) == 1:
                gxl = gx[B.pix_list]
                gyl = gy[B.pix_list]
            else:
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
            if B.n_pix_list == 1:
                index = i
            else:
                index = B.BData.find_map_pixel_index(pixel)
            wdata = np.ones(len(B.BData.map_data[index]))
            try: 
                print('trying scipy.interpolate.griddata')
                zi = interp.griddata((map_x[index]-gxl[index],map_y[index]-gyl[index]),B.BData.map_data[index],(grid_x,grid_y),method='linear').T
                wi = interp.griddata((map_x[index]-gxl[index],map_y[index]-gyl[index]),wdata,(grid_x, grid_y),method='linear').T
            except Exception as e:
                print(e)
                try:
                    zi = mlab.griddata(map_x[index]-gxl[index],map_y[index]-gyl[index],B.BData.map_data[index],xi,yi,interp='linear')
                    wi = mlab.griddata(map_x[index]-gxl[index],map_y[index]-gyl[index],wdata,xi,yi,interp='linear')
                except:
                    zi = mlab.griddata(map_x[index]-gxl[index],map_y[index]-gyl[index],B.BData.map_data[index],xi,yi,interp='nn')
                    wi = mlab.griddata(map_x[index]-gxl[index],map_y[index]-gyl[index],wdata,xi,yi,interp='nn')
            zi_sum = zi_sum + zi
            wi_sum = wi_sum + wi
        pl.imshow(zi_sum/wi_sum,interpolation='bicubic',cmap=pl.cm.jet,origin='lower',extent=map_region)
        #pl.plot(map_x[index],map_y[index])
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
        pl.suptitle('ObsNum %d: %s %s %sGHz\n %s'%(B.obsnum,B.BData.receiver,B.BData.source,B.BData.line_rest_frequency,textstr)) 
        try:
            pl.tight_layout(rect=[0, 0.03, 1, 0.9])
        except:
            pass
        pl.colorbar()


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
            if len(B.BData.map_data) == 1:
                gxl = gx[B.pix_list]
                gyl = gy[B.pix_list]
            else:
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
        for i in range(B.n_pix_list):
            pixel = B.pix_list[i]
            index = B.BData.find_map_pixel_index(pixel)
            wdata = np.ones(len(B.BData.map_data[index]))
            try:
                print('trying scipy.interpolate.griddata')
                zi = interp.griddata((B.BData.map_x[index]-gxl[index],B.BData.map_y[index]-gyl[index]),B.BData.map_data[index],(grid_x,grid_y),method='linear').T
                wi = interp.griddata((B.BData.map_x[index]-gxl[index],B.BData.map_y[index]-gyl[index]),wdata,(grid_x, grid_y),method='linear').T
            except Exception as e:
                print(e)
                try:
                    zi = mlab.griddata(B.BData.map_x[index]-gxl[index],B.BData.map_y[index]-gyl[index],B.BData.map_data[index],xi,yi,interp='linear')
                    wi = mlab.griddata(B.BData.map_x[index]-gxl[index],B.BData.map_y[index]-gyl[index],wdata,xi,yi,interp='linear')
                except:
                    zi = mlab.griddata(B.BData.map_x[index]-gxl[index],B.BData.map_y[index]-gyl[index],B.BData.map_data[index],xi,yi,interp='nn')
                    wi = mlab.griddata(B.BData.map_x[index]-gxl[index],B.BData.map_y[index]-gyl[index],wdata,xi,yi,interp='nn')
            zi_sum = zi_sum + zi
            wi_sum = wi_sum + wi

        zi = zi_sum/wi_sum
        fig = pl.figure()
        ax = fig.gca(projection='3d')
        xm,ym = np.meshgrid(xi, yi)
        norm =  matplotlib.colors.Normalize(vmin=np.min(zi), vmax=np.max(zi))
        my_col = pl.cm.jet(norm(zi))
        surf = ax.plot_surface(xm, ym, zi, rstride=1, cstride=1, facecolors=my_col, linewidth=1, antialiased=False)
        #surf = ax.plot_surface(xm, ym, zi, rstride=1, cstride=1, cmap=pl.cm.jet, linewidth=1, antialiased=True)
        #fig.colorbar(surf)
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
        m = pl.cm.ScalarMappable(cmap=pl.cm.jet, norm=norm)
        m.set_array([])
        pl.colorbar(m)


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

