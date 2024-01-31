import numpy as np
from lmtslr.spec.spec import SpecBankData, SpecBankCal
from lmtslr.viewer.spec_viewer import SpecBankViewer, SpecCalViewer
from lmtslr.ifproc.ifproc import lookup_ifproc_file, IFProcQuick, IFProcData, IFProcCal
from lmtslr.utils.roach_file_utils import find_roach_from_pixel, lookup_roach_files, create_roach_list
import sys
from beam import BeamMap
from beam_viewer import BeamMapView
from merge_png import merge_png
import time
import os
import matplotlib.pyplot as pl
from msg_image import mkMsgImage

def extend_ifproc(ifproc):
    self = ifproc
    import netCDF4
    import datetime
    from scipy import interpolate
    if os.path.isfile(self.filename):
        self.nc = netCDF4.Dataset(self.filename, 'a')

    self.tel_utc = 180/15/np.pi*self.nc.variables['Data.TelescopeBackend.TelUtc'][:][:]
    utdate = self.utdate
    utdate = datetime.datetime.utcfromtimestamp(self.time[0]).date()
    print(utdate, self.tel_utc[0])
    import dateutil.parser as dparser
    print([dparser.parse(str(utdate)+' '+str(datetime.timedelta(hours=self.tel_utc[0]))+' UTC', fuzzy=True) for i in range(1)])
    self.sky_time = np.array([dparser.parse(str(utdate)+' '+str(datetime.timedelta(hours=self.tel_utc[i]))+' UTC', fuzzy=True).timestamp() for i in range(len(self.tel_utc))])

    self.ut = self.nc.variables['Data.TelescopeBackend.TelUtc'][:] / 2 / np.pi * 24
    self.lst = self.nc.variables['Data.TelescopeBackend.TelLst'][:] / 2 / np.pi * 24
    self.ramap_file = (self.nc.variables['Data.TelescopeBackend.SourceRaAct'][:] - self.source_RA) * np.cos(self.source_Dec) * 206264.8
    self.decmap_file = (self.nc.variables['Data.TelescopeBackend.SourceDecAct'][:] - self.source_Dec) * 206264.8

    # interpolate ra/dec based on tel time
    if False:
        sl = slice(0, len(self.time), 1)
        ra_file = self.nc.variables['Data.TelescopeBackend.SourceRaAct'][:]
        dec_file = self.nc.variables['Data.TelescopeBackend.SourceDecAct'][:]
        self.ra_interpolation_function = interpolate.interp1d(self.sky_time[sl],
                                                              ra_file[sl],
                                                              bounds_error=False,
                                                              kind='previous',
                                                              fill_value='extrapolate')
        self.dec_interpolation_function = interpolate.interp1d(self.sky_time[sl],
                                                               dec_file[sl],
                                                               bounds_error=False,
                                                               kind='previous',
                                                               fill_value='extrapolate')
        ra_interp = self.ra_interpolation_function(
            np.ma.getdata(self.time, subok=False))
        dec_interp = self.dec_interpolation_function(
            np.ma.getdata(self.time, subok=False))
        self.ramap_interp = (ra_interp - self.source_RA) * np.cos(self.source_Dec) * 206264.8
        self.decmap_interp = (dec_interp - self.source_Dec) * 206264.8
        if False:
            print('sky_time-time', self.sky_time-self.time)
            print('ramap_file', self.ramap_file)
            print('sky_time', self.sky_time)
            print('time', self.time)
            print('ra_interp', ra_interp)
            print('ramap_interp', self.ramap_interp)

    # compute ra/dec from astropy
    if False:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        from astropy.time import Time
        from astroplan import Observer
        from astropy import coordinates as coord
        from pytz import timezone

        _lmt_info = {
            'instru': 'lmt',
            'name': 'LMT',
            'name_long': "Large Millimeter Telescope",
            'location': {
                'lon': '-97d18m52.5s',
                'lat': '+18d59m9.6s',
                'height': 4640 << u.m,
                },
            'timezone_local': 'America/Mexico_City'
        }

        lmt_location = coord.EarthLocation.from_geodetic(**_lmt_info['location'])
        """The local of LMT."""

        lmt_timezone_local = timezone(_lmt_info['timezone_local'])
        """The local time zone of LMT."""

        lmt_observer = Observer(
            name=_lmt_info['name_long'],
            location=lmt_location,
            timezone=lmt_timezone_local,
        )

        observer = lmt_observer

        tel_time = Time(self.nc['Data.TelescopeBackend.TelTime'][:], format='unix', scale='utc', location=lmt_observer.location)
        tel_az = self.nc['Data.TelescopeBackend.TelAzAct'][:] << u.rad
        tel_alt = self.nc['Data.TelescopeBackend.TelElAct'][:] << u.rad
        tel_az_cor = self.nc['Data.TelescopeBackend.TelAzCor'][:] << u.rad
        tel_alt_cor = self.nc['Data.TelescopeBackend.TelElCor'][:] << u.rad
        tel_az_tot = tel_az - (tel_az_cor) / np.cos(tel_alt)
        tel_alt_tot = tel_alt - (tel_alt_cor)
        altaz_frame = observer.altaz(time=tel_time)
        tel_icrs_astropy = SkyCoord(tel_az_tot, tel_alt_tot, frame=altaz_frame).transform_to('icrs')
        # update variables and save
        parang = observer.parallactic_angle(time=tel_time, target=tel_icrs_astropy)
        self.parang_astropy = parang.radian
        self.ramap_astropy = (tel_icrs_astropy.ra.radian - self.source_RA) * np.cos(self.source_Dec) * 206264.8
        self.decmap_astropy = (tel_icrs_astropy.dec.radian - self.source_Dec) * 206264.8

        az_ = tel_az_tot.value
        el_ = tel_alt_tot.value
        ra_ = tel_icrs_astropy.ra.radian
        dec_ = tel_icrs_astropy.dec.radian
        ut_ = tel_time.ut1.value
        lst_ = tel_time.sidereal_time('apparent').value

        if False:
            self.nc.variables['Data.TelescopeBackend.SourceRaAct'][:] = tel_icrs_astropy.ra.radian
            self.nc.variables['Data.TelescopeBackend.SourceDecAct'][:] = tel_icrs_astropy.dec.radian

            self.nc.close()

        if False:
            import matplotlib.pyplot as pl
            if False:
                pl.plot(self.ramap,self.decmap, 'r')
                pl.plot(self.ramap_astropy,self.decmap_astropy, 'b')
            else:
                pl.plot(self.ramap-self.ramap_astropy, 'r')
                ax2 = pl.twinx()
                ax2.plot(self.decmap-self.decmap_astropy, 'b')

            pl.show()

            self.ramap = self.ramap_astropy
            self.decmap = self.decmap_astropy

    # set the ra/dec map
    #self.ramap = self.ramap_interp
    #self.decmap = self.decmap_interp

    if False:
        def stat_change(d, d_orig, unit, name):
            #dd = (d - d_orig).to_value(unit)
            dd = (d - d_orig)
            dd = dd[np.isfinite(dd)]
            #print(f"{name} changed with diff ({unit}): min={dd.max()} max={dd.min()} mean={dd.mean()} std={np.std(dd)}")
        stat_change(self.parang_astropy, self.parang, u.deg, 'ActParAng') 
        stat_change(self.ramap_file, self.ramap_interp, u.arcsec, 'file-interp') 
        stat_change(self.decmap_file, self.decmap_interp, u.arcsec, 'file-interp')
        stat_change(self.ramap_file, self.ramap_astropy, u.arcsec, 'file-astropy') 
        stat_change(self.decmap_file, self.decmap_astropy, u.arcsec, 'file-astropy')
        stat_change(self.ramap_interp, self.ramap_astropy, u.arcsec, 'interp-astropy') 
        stat_change(self.decmap_interp, self.decmap_astropy, u.arcsec, 'interp-astropy')

    if False:
        import matplotlib.pyplot as pl
        sl = np.where(self.bufpos == 0)
        # traces
        ax = pl.subplot()
        ax.plot(self.time[sl],self.ramap_file[sl], 'r', label='file')
        ax.plot(self.time[sl],self.ramap_interp[sl], 'm', label='interp')
        ax.plot(self.time[sl],self.ramap_astropy[sl], 'y', label='astropy')
        pl.legend()
        pl.show()
        ax = pl.subplot()
        ax.plot(self.time[sl],self.ramap_file[sl]-self.ramap_interp[sl], 'r', label='file-interp')
        ax.plot(self.time[sl],self.ramap_file[sl]-self.ramap_astropy[sl], 'b', label='file-astropy')
        pl.legend()
        pl.show()
        ax = pl.subplot()
        ax.plot(self.time[sl],self.ut[sl]-ut_[sl], 'm', label='UT')
        ax.plot(self.time[sl],self.lst[sl]-lst_[sl], 'g', label='LST')
        pl.legend()
        pl.show()
        import sys
        sys.exit(0)

    self.close_nc()

def create_map_data_main(ifproc):
    self = ifproc
    idx = np.where(self.bufpos == 0)[0]
    self.map_x = self.map_x[:,idx]
    self.map_y = self.map_y[:,idx]
    self.map_az = self.map_az[:,idx]
    self.map_el = self.map_el[:,idx]
    self.map_ra = self.map_ra[:,idx]
    self.map_dec = self.map_dec[:,idx]
    self.map_l = self.map_l[:,idx]
    self.map_b = self.map_b[:,idx]
    self.map_p = self.map_p[:,idx]
    self.map_g = self.map_g[:,idx]
    #self.map_n = self.map_n[:,idx]
    self.map_data = self.map_data[:,idx]

def linepoint(args_dict, view_opt=0):

    obsnum = (args_dict.get('ObsNum', None))
    spec_cont = (args_dict.get('SpecOrCont', 'Spec')).lower()
    line_list = (args_dict.get('LineList', None))
    baseline_list = (args_dict.get('BaselineList', None))
    baseline_fit_order = (args_dict.get('BaselineFitOrder', 0))
    tsys = (args_dict.get('TSys', None))
    tracking_beam = (args_dict.get('TrackingBeam', None))
    opt = (args_dict.get('Opt', 0))
    bank = (args_dict.get('Bank', 0)) 

    print('args = ', obsnum, spec_cont, line_list, baseline_list, baseline_fit_order, tsys, tracking_beam, opt, bank)
    
    # define time stamp
    file_ts = '%d_%d_%d'%(obsnum, int(time.time()*1000), os.getpid())

    # define roach_pixels_all
    roach_pixels_all = [[i+j*4 for i in range(4)] for j in range(4)]

    # read the ifproc file to get the data and the tracking beam
    ifproc_file = lookup_ifproc_file(obsnum)
    if not ifproc_file:
        txt = 'No ifproc files found for %d'%obsnum
        print (txt)
        mkMsgImage(pl, obsnum, txt=txt, im='lmtlp_%s.png'%file_ts, label='Error', color='r')
        return {'plot_file': 'lmtlp_%s.png'%file_ts}

    if view_opt == 0x1234:
        IData = IFProcData(ifproc_file)
        pl.clf()
        pl.plot(IData.azmap, IData.elmap)
        pl.xlabel('AzMap (")')
        pl.ylabel('ElMap (")')
        pl.title('%d %s el=%lf'%(obsnum,IData.date_obs,IData.elev))
        pl.savefig('%s_azelmap.png'%obsnum, bbox_inches='tight')
        pl.show()
        return {'plot_file': '%s_azelmap.png'%obsnum}

    # probe the ifproc file for obspgm
    ifproc_file_quick = IFProcQuick(ifproc_file)

    # get obspgm
    obspgm = ifproc_file_quick.obspgm

    print ('obsnum', obsnum)
    print ('receiver', ifproc_file_quick.receiver)
    print ('obspgm', obspgm)
    print ('tracking_beam', tracking_beam)

    # init params to None so we can return if no params
    params = None

    if obspgm == 'Cal':
        ICal = IFProcCal(ifproc_file)
        ICal.compute_calcons()
        ICal.compute_tsys()
        for ipix in range(ICal.npix):
            print ('IFPROC Tsys[%2d] = %6.1f'%(ipix,ICal.tsys[ipix]))
        if ICal.receiver == 'Msip1mm':
            msip1mm_pixel_description = {0: 'P0_USB',
                                         1: 'P0_LSB',
                                         2: 'P1_LSB',
                                         3: 'P1_USB'}
            dirname = '/var/www/vlbi1mm/'
            filename = 'vlbi1mm_tsys.html'
            if os.path.exists(dirname):
                filename = dirname + filename
            with open(filename, 'w') as fp:
                for ipix in range(ICal.npix):
                    desc =  msip1mm_pixel_description.get(ipix)
                    val = ICal.tsys[ipix]
                    fp.write("%s %3.1f\n" %(desc, val))
                fp.write("ObsNum %d\n" %(obsnum))
                fp.write("Time %3.1f\n" %(ICal.time[0]))

        pl.figure(num=3,figsize=(6,6))
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
        pl.savefig('lmtlp_2_%s.png'%file_ts, bbox_inches='tight')

        if spec_cont == 'cont':
            merge_png(['lmtlp_2_%s.png'%file_ts], 'lmtlp_%s.png'%file_ts)
        else:
            files,nfiles = lookup_roach_files(obsnum)
            if ICal.receiver == 'Msip1mm':
                bank_files = [files, files]
                bank_pixel_list = [[0, 2], [1, 3]]
            else:
                bank_files = [files[i:i+4] for i in range(0, len(files), 4)] 
                bank_pixel_list = 2*[list(range(16))]
            fnames = []
            for bank in range(len(bank_files)):
                SCal = SpecBankCal(bank_files[bank],ICal,bank=bank,pixel_list=bank_pixel_list[bank])
                # create viewer
                SV = SpecCalViewer()
                SV.set_figure(figure=1+bank)
                SV.open_figure()
                SV.plot_tsys(SCal)
                fnames += ['lmtlp_%s_%d.png'%(file_ts, bank)]
                pl.savefig(fnames[-1], bbox_inches='tight')

            # merge plots
            print(fnames)
            merge_png(fnames+['lmtlp_2_%s.png'%file_ts], 'lmtlp_%s.png'%file_ts)
        
        params = np.zeros((1,1))
        params[0,0] = np.mean(ICal.tsys)
        return {'plot_file': 'lmtlp_%s.png'%file_ts, 'params' : params, 'ifproc_data': ICal}

    # not a Cal
    # not a Cal
    # not a Cal
    # not a Cal
    
    # open data file
    IData = IFProcData(ifproc_file)
    extend_ifproc(IData)
    IData.xmap = IData.azmap
    IData.ymap = IData.elmap
    
    bs_beams = IData.bs_beams
    map_motion = IData.map_motion
    line_velocity = IData.velocity
    reduce_type = 2

    # specify beam for fitting and plotting
    selected_beam = 10

    # check if tracking a specific pixel to modify the roach list and the selected_beam
    if 'Msip1mm' not in IData.receiver:
        tracking_beam = None
    if tracking_beam == None:
        tracking_beam = int(IData.tracking_beam)
    if False and tracking_beam == -1:
        tracking_beam = selected_beam
    if tracking_beam != -1:
        selected_beam = tracking_beam
        if bs_beams != []:
            roach_list = []
            for beam in bs_beams:
                roach_beam = find_roach_from_pixel(beam)
                if roach_beam[0] not in roach_list:
                    roach_list += roach_beam
        else:
            roach_list = find_roach_from_pixel(tracking_beam)
    else:
        roach_list = list(range(4))

    if 'Msip1mm' not in IData.receiver:
        roach_list = [i+(bank*4) for i in roach_list]
    else:
        roach_list = [0]

    # build the roach directory list
    roach_dir_list = [ 'roach%d'%i for i in roach_list]

    # build the pixel list
    if bs_beams != []:
        pixel_list = bs_beams
    elif tracking_beam == -1 or obspgm == 'On':
        pixel_list = sum([roach_pixels_all[roach_index] for roach_index in range(len(roach_pixels_all))], [])
    else:
        pixel_list = [tracking_beam]

    print ('tracking_beam', tracking_beam)
    print ('selected_beam', selected_beam)
    print ('roach_list', roach_list)
    print ('roach_dir_list', roach_dir_list)
    print ('pixel_list', pixel_list)
    print ('bs_beams', bs_beams)
    print ('map_motion', map_motion)
    print ('line_velocity', line_velocity)
    print ('line_freq', IData.line_rest_frequency)

    if spec_cont == 'cont':
        print ('continuum reduction')
        pass
    else:
        print ('spectral line reduction')
        # find the roach files
        files,nfiles = lookup_roach_files(obsnum,roach_dir_list)
        if not files:
            txt = 'No roach files found for %d in %s' % (obsnum,str(roach_dir_list))
            print (txt)
            mkMsgImage(pl, obsnum, txt=txt, im='lmtlp_%s.png'%file_ts, label='Error', color='r')
            return {'plot_file': 'lmtlp_%s.png'%file_ts}

        if IData.receiver == 'Msip1mm':
            bank_files = [files, files]
            bank_pixel_list = [[0, 2], [1, 3]]
            if selected_beam in bank_pixel_list[0]:
                bank = 0
            else:
                bank = 1
        else:
            bank_files = [[],[]]
            bank_files[bank] = files
        print('bank', bank, 'bank_files', bank_files)

    # build reduction parameters
    #line_list = [[-27.5,-25.5]]

    # get line_list from arg then from ifproc data file then from default
    if not line_list or (all(isinstance(x, list) for x in line_list) and not [item for sublist in line_list for item in sublist]):
        line_list = IData.line_list
        print ('line_list in file', line_list)
        if not line_list:
            line_list = [[-30+line_velocity,30+line_velocity]]
            print ('line_list from default', line_list)
        else:
            print ('line_list from ifproc', line_list)
    else:
        if not any(isinstance(el, list) for el in line_list):
            print ('modify line_list to be a list of lists')
            line_list = [line_list]
        print ('line_list from arg', line_list)
        
    # get baseline_list from arg then from ifproc data file then from default
    if not baseline_list or not [item for sublist in baseline_list for item in sublist]:
        baseline_list = IData.baseline_list
        print ('baseline_list in file', baseline_list)
        if baseline_list:
            flat_list = [item for sublist in baseline_list for item in sublist]
        else:
            flat_list = None
        if not flat_list:
            baseline_list = [[-100-30+line_velocity,-30+line_velocity],[30+line_velocity,100+30+line_velocity]]
            print ('baseline_list from default', baseline_list)
        else:
            print ('baseline_list from ifproc', baseline_list)
    else:
        if not any(isinstance(el, list) for el in baseline_list):
            print ('modify baseline_list to be a list of lists')
            baseline_list = [baseline_list]
        print ('baseline_list from arg', baseline_list)

    if not tsys:
        tsys = 200

    if baseline_fit_order < 0:
        baseline_fit_order = 0
    elif baseline_fit_order > 3:
        baseline_fit_order = 3
        
    print ('line_list', line_list)
    print ('baseline_list', baseline_list)
    print ('baseline_fit_order', baseline_fit_order)
    print ('tsys', tsys)
    plot_axis = [-100-30+line_velocity, 100+30+line_velocity,-5,15] # modify y according to peak spectrum
    fit_circle = 30

    # create a list of cal spectra
    tsys_spectra = []

    if spec_cont == 'cont':
        pass
    else:
        # create the spec_bank object. This reads all the roaches in the list "files"
        print(files, bank_files[bank], pixel_list)
        SData = SpecBankData(bank_files[bank],IData,pixel_list=pixel_list,bank=bank)

        # set the pixel list to the pixels from the files we could find
        print ('unmodified pixel_list = ', pixel_list)
        pixel_list = SData.roach_pixel_ids
        print ('modified pixel_list = ', pixel_list)

    # check whether to use calibration and open necessary file
    use_calibration = True
    if use_calibration == True:
        calobsnum = IData.calobsnum
        print ('cal obs num = ', calobsnum)
        ifproc_cal_file = ''
        if calobsnum > 0:
            cal_files,ncalfiles = lookup_roach_files(calobsnum,roach_dir_list)
            ifproc_cal_file = lookup_ifproc_file(calobsnum)
        if ifproc_cal_file == '':
            use_calibration = False
        else:
            ICal = IFProcCal(ifproc_cal_file)
            if spec_cont == 'cont':
                ICal.compute_calcons()
                ICal.compute_tsys()
                for ipix in range(ICal.npix):
                    print ('IFPROC Tsys[%2d] = %6.1f'%(ipix,ICal.tsys[ipix]))
            else:
                if IData.receiver == 'Msip1mm':
                    bank_cal_files = [cal_files, cal_files]
                else:
                    bank_cal_files = [cal_files[i:i+4] for i in range(0, len(cal_files), 4)] 
                SCal = SpecBankCal(bank_cal_files[bank],ICal,pixel_list=pixel_list,bank=bank)
                check_cal = SCal.test_cal(SData)
                for ipix in range(SData.npix):
                    tsys_spectra.append(SCal.roach[ipix].tsys_spectrum)
                    print ('SPEC Tsys[%2d] = %6.1f'%(ipix,SCal.roach[ipix].tsys))
                if check_cal > 0:
                    print ('WARNING: CAL MAY NOT BE CORRECT')

    # line statistics
    line_stats_all = []

    if spec_cont == 'cont':
        if use_calibration == False:
            IData.dont_calibrate_data()
        else:
            IData.calibrate_data(ICal)
        IData.create_map_data()

    else:
        if use_calibration == False:
            print ('WARNING: NOT USING CAL, USING TSYS',tsys)
            for ipix in range(SData.npix):
                tsys_spectra.append(0)

        SData.cal_flag = use_calibration
        # reduce all spectra
        if obspgm == 'Map' or obspgm == 'Lissajous':
            for ipix in range(SData.npix):
                SData.roach[ipix].reduce_spectra(stype=2,calibrate=use_calibration,tsys_spectrum=tsys_spectra[ipix],tsys_no_cal=tsys)
        elif obspgm == 'Bs':
            dict_pix = {}
            dict_pix[SData.roach[0].pixel] = 0
            dict_pix[SData.roach[1].pixel] = 1
            SData.roach[dict_pix[bs_beams[0]]].reduce_ps_spectrum(stype=2,normal_ps=True,calibrate=use_calibration,tsys_spectrum=tsys_spectra[dict_pix[bs_beams[0]]],tsys_no_cal=tsys)
            SData.roach[dict_pix[bs_beams[1]]].reduce_ps_spectrum(stype=2,normal_ps=False,calibrate=use_calibration,tsys_spectrum=tsys_spectra[dict_pix[bs_beams[1]]],tsys_no_cal=tsys)
        elif obspgm == 'Ps':
            for ipix in range(SData.npix):
                SData.roach[ipix].reduce_ps_spectrum(stype=2,normal_ps=True,calibrate=use_calibration,tsys_spectrum=tsys_spectra[ipix],tsys_no_cal=tsys)
        elif obspgm == 'On':
            for ipix in range(SData.npix):
                SData.roach[ipix].reduce_on_spectrum(calibrate=False,tsys_spectrum=tsys_spectra[ipix],tsys_no_cal=tsys)

        # set the baseline channels from velocities
        SData.make_velocity_list(baseline_list,'baseline')
        # set the line integration channels from velocities
        SData.make_velocity_list(line_list,'line')

        # create viewer
        SV = SpecBankViewer()

        # bs
        if obspgm == 'Bs':
            theSpectrum = (SData.roach[0].ps_spectrum+SData.roach[1].ps_spectrum)/2.
            v = SData.create_velocity_scale()
            line_stats = SData.roach[0].LineStatistics(SData.roach[0],v,theSpectrum,SData.clist,SData.nc,SData.blist,SData.nb,baseline_fit_order,SData.roach_pixel_ids[0],SData.roach_pixel_ids[1],obspgm)
            line_stats_all.append(line_stats)
            print ('line_stats =', line_stats.to_string())
            params = np.zeros((1,2))
            params[0,0] = line_stats.yint
            params[0,1] = pixel_list[0]
            SV.set_figure(figure=1)
            SV.open_figure()
            SV.plot_bs(SData,baseline_fit_order,plot_axis,line_stats,line_list,baseline_list)

        # on
        elif obspgm == 'On':
            v = SData.create_velocity_scale()
            params = np.zeros((SData.npix,2))
            for ipix in range(SData.npix):
                theSpectrum = SData.roach[ipix].on_spectrum
                line_stats = SData.roach[ipix].LineStatistics(SData.roach[ipix],v,theSpectrum,SData.clist,SData.nc,SData.blist,SData.nb,baseline_fit_order,SData.roach_pixel_ids[ipix],SData.roach_pixel_ids[ipix],obspgm)
                line_stats_all.append(line_stats)
                print ('line_stats =', line_stats.to_string())
                params[ipix,0] = line_stats.yint
                params[ipix,1] = pixel_list[ipix]
                prange = np.where(np.logical_and(v >= plot_axis[0], v <= plot_axis[1]))

            SV.set_figure(figure=1)
            SV.open_figure()
            SV.plot_on(SData)

        # ps
        elif obspgm == 'Ps':
            v = SData.create_velocity_scale()
            params = np.zeros((SData.npix,2))
            for ipix in range(SData.npix):
                theSpectrum = SData.roach[ipix].ps_spectrum
                line_stats = SData.roach[ipix].LineStatistics(SData.roach[ipix],v,theSpectrum,SData.clist,SData.nc,SData.blist,SData.nb,baseline_fit_order,SData.roach_pixel_ids[ipix],SData.roach_pixel_ids[ipix],obspgm)
                line_stats_all.append(line_stats)
                print ('line_stats =', line_stats.to_string())
                params[ipix,0] = line_stats.yint
                params[ipix,1] = pixel_list[ipix]
                prange = np.where(np.logical_and(v >= plot_axis[0], v <= plot_axis[1]))

            SV.set_figure(figure=1)
            SV.open_figure()
            SV.plot_ps(SData,baseline_fit_order,plot_axis,line_stats_all,line_list,baseline_list)

        # grid map
        elif map_motion == 'Discrete':
            # this does a baseline and integration for the pixels in the list
            SData.create_map_grid_data(SData.clist,SData.nc,SData.blist,SData.nb,baseline_fit_order,pixel_list=pixel_list)

            # plot the spectra
            SV.set_figure(figure=1)
            SV.open_figure()
            SV.plot_all_spectra(SData,selected_beam,plot_axis,SData.blist,SData.nb,line_list,baseline_list)

        # raster map
        elif map_motion == 'Continuous' or obspgm == 'Lissajous':
            # this does a baseline and integration for the pixels in the list
            SData.create_map_data(SData.clist,SData.nc,SData.blist,SData.nb,baseline_fit_order,pixel_list=pixel_list)

            # show the peak spectrum
            SV.set_figure(figure=1)
            SV.open_figure()
            SV.plot_peak_spectrum(SData,selected_beam,plot_axis,SData.blist,SData.nb,line_list,baseline_list)

        # save spectra plot
        pl.savefig('lmtlp_%s.png'%file_ts, bbox_inches='tight')

        # write a combined ifproc/roach file
        #SW = spec_bank_writer()
        #spec_files = SW.write(ifproc_file,SData,roach_list=roach_list,pixel_list=pixel_list,tracking_beam=tracking_beam)

    # set grid spacing
    grid_spacing = 1
    # now use the BeamMap class to solve for peaks
    B = None
    if obspgm == 'Map' or obspgm == 'Lissajous':
        #pixel_list = [i for i in range(16)]
        print ('beam map pix list', pixel_list)
        if IData.receiver == "Msip1mm":
            fit_circle = 10
        else:
            fit_circle = 30
        if spec_cont == 'cont':
            create_map_data_main(IData)
            B = BeamMap(IData,pix_list=pixel_list)
            B.fit_peaks_in_list(fit_circle=fit_circle)
        else:
            B = BeamMap(SData,pix_list=pixel_list)
            B.fit_peaks_in_list(fit_circle=fit_circle)
        if tracking_beam != -1:
            params = np.zeros((1,4))
            if len(pixel_list) == 1:
                pix_id = 0
            else:
                pix_id = B.BData.find_map_pixel_index(tracking_beam)
            params[0,0] = B.peak_fit_params[pix_id,0]
            params[0,1] = B.peak_fit_params[pix_id,1]
            params[0,2] = B.peak_fit_params[pix_id,3]
            params[0,3] = pixel_list[pix_id]
        else:
            params = np.zeros((len(pixel_list), 4))
            for ipix in range(len(pixel_list)):
                params[ipix,0] = B.peak_fit_params[ipix,0]
                params[ipix,1] = B.peak_fit_params[ipix,1]
                params[ipix,2] = B.peak_fit_params[ipix,3]
                params[ipix,3] = pixel_list[ipix]
        BV = BeamMapView()
        BV.print_pixel_fits(B)

        if map_motion == 'Discrete':
            # show a map of the data with peak position overlain
            BV.set_figure(figure=2)
            BV.open_figure()
            #BV.add_subplot(SV.fig)
            BV.map(B,[],grid_spacing,apply_grid_corrections=True)
            if spec_cont != 'cont':
                BV.show_peaks(B,apply_grid_corrections=True,show_map_points=selected_beam)
            pl.savefig('lmtlp_2_%s.png'%file_ts, bbox_inches='tight')

            if spec_cont == 'cont':
                BV.set_figure(figure=10)
                BV.open_figure()
                BV.show_fit(B,selected_beam)
                pl.savefig('lmtlp_%s.png'%file_ts, bbox_inches='tight')
            # merge plots
            merge_png(['lmtlp_%s.png'%file_ts, 'lmtlp_2_%s.png'%file_ts], 'lmtlp_%s.png'%file_ts)

        else:
            # show a map of the data with peak position overlain
            BV.set_figure(figure=2)
            BV.open_figure()
            #BV.add_subplot(SV.fig)
            BV.map(B,[],grid_spacing,apply_grid_corrections=True)
            BV.show_peaks(B,apply_grid_corrections=True)
            pl.savefig('lmtlp_2_%s.png'%file_ts, bbox_inches='tight')

            if view_opt & 0x4:
                BV.set_figure(figure=3)
                BV.open_figure()
                BV.map(B,[],grid_spacing,apply_grid_corrections=True,display_coord=0)
                BV.set_figure(figure=4)
                BV.open_figure()
                BV.map(B,[],grid_spacing,apply_grid_corrections=True,display_coord=1)
                BV.set_figure(figure=5)
                BV.open_figure()
                BV.map(B,[],grid_spacing,apply_grid_corrections=True,display_coord=2)
            if False:
                BV.set_figure(figure=6)
                BV.open_figure()
                BV.map(B,[],grid_spacing,apply_grid_corrections=True,display_coord=11)
                BV.set_figure(figure=7)
                BV.open_figure()
                BV.map(B,[],grid_spacing,apply_grid_corrections=True,display_coord=21)
            if spec_cont == 'cont':
                BV.set_figure(figure=10)
                BV.open_figure()
                BV.show_fit(B,selected_beam)
                pl.savefig('lmtlp_%s.png'%file_ts, bbox_inches='tight')
            # merge plots
            merge_png(['lmtlp_%s.png'%file_ts, 'lmtlp_2_%s.png'%file_ts], 'lmtlp_%s.png'%file_ts)
    
    if view_opt & 0x2:
        if obspgm == 'Map' or obspgm == 'Lissajous':
            # show the fit to the data
            BV.set_figure(figure=10)
            BV.open_figure()
            BV.show_fit(B,selected_beam)

        if map_motion == 'Continuous':
            if spec_cont == 'cont':
                BV.set_figure(figure=11)
                BV.open_figure()
                BV.sanchez_map(B,[],grid_spacing)
                BV.show_peaks(B,apply_grid_corrections=False)
            else:
                # the "sanchez map" shows the array grid on the sky
                SV.set_figure(figure=11)
                SV.open_figure()
                SV.sanchez_map(SData,[],grid_spacing,None,pixel_list=pixel_list)

                # the "map" uses the array grid model to align the pixels
                SV.set_figure(figure=12)
                SV.open_figure()
                SV.map(SData,[],grid_spacing,None,pixel_list=pixel_list)

                # show the waterfall plot near the peak spectrum
                SV.set_figure(figure=13)
                SV.open_figure()
                SV.waterfall(SData,selected_beam,[-1500,1500],[-1,10],SData.blist,SData. nb)

    if view_opt & 0x8:
        BV.map3d(B,[],grid_spacing,apply_grid_corrections=True)


    print ('plot_file', 'lmtlp_%s.png'%file_ts)
    print ('params', params)
    return {'plot_file': 'lmtlp_%s.png'%file_ts,
            'params': params,
            'ifproc_data': IData,
            'line_stats': line_stats_all,
            'ifproc_file': ifproc_file,
            'peak_fit_params': B.peak_fit_params if B is not None else None,
            'peak_fit_errors': B.peak_fit_errors if B is not None else None,
            'peak_fit_snr': B.peak_fit_snr if B is not None else None,
            'clipped': B.clipped if B is not None else None,
            'pixel_list': pixel_list
    }

if __name__ == '__main__':
    # define options, 0 = write spec file only, 1 = plot, 2 = generate grid
    opt = 0

    if len(sys.argv) > 2:
        try:
            opt = int(sys.argv[-2], 0)
            print ('opt =', opt)
        except:
            pass

    # Chi_Cyg cal
    obsNum = 78003
    # Chi_Cyg grid
    obsNum = 78004
    # IRC otf
    obsNum = 76406
    # MARS otf
    obsNum = 77358
    # BS
    obsNum = 78060
    # PS
    obsNum = 76085
    # R-Cas Ra Map
    obsNum = 94052

    try:
        arg = sys.argv[-1]
        if arg.startswith('c'):
            obsNum = 78003
        elif arg.startswith('g'):
            obsNum = 78004
        elif arg.startswith('m'):
            obsNum = 76406
        elif arg.startswith('p'):
            obsNum = 76085
        elif arg.startswith('b'):
            obsNum = 78075
        elif arg == ('9pt'):
            obsNum = 78091
        else:
            obsNum = int(arg)
    except:
        pass

    if obsNum > 0:
        args_dict = dict()
        args_dict['ObsNum'] = obsNum
        args_dict['SpecOrCont'] = 'Cont' if opt & 0x1000 else 'Spec'
        args_dict['LineList'] = None
        args_dict['BaselineList'] = None
        args_dict['BaselineFitOrder'] = 0
        args_dict['TSys'] = 0
        args_dict['TrackingBeam'] = None
        args_dict['Opt'] = opt
        args_dict['Bank'] = 0

        for arg in sys.argv:
            if '=' in arg:
                x = arg.split('=')
                if 'p' in x[0]:
                    args_dict['TrackingBeam'] = int(x[1])
                    print('TrackingBeam', args_dict['TrackingBeam'])
        
        lp_dict = linepoint(args_dict, view_opt=opt)
        print(lp_dict)

        # show plot
        if opt & 0x1:
            pl.show()
    else:
        print("obsnum %d not valid" % obsNum)
            

