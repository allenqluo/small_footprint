import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib
from datetime import datetime
import simplekml
from itertools import product
import matplotlib.dates as mdates
from IceFiles import IceSatFiles, FilterGroup, NewIceParam
from textwrap import wrap


#GLOBAL PLOTTING OPTIONS AND VARIOUS USEFUL KEYS FOR ICESAT LANDCOVER VALUES
months_lookup = {1: 'Jan',
                    2: 'Feb',
                    3: 'Mar',
                    4: 'Apr',
                    5: 'May',
                    6: 'Jun',
                    7: 'Jul',
                    8: 'Aug',
                    9: 'Sep',
                    10: 'Oct',
                    11: 'Nov',
                    12: 'Dec'}

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

#matplotlib.use('Agg')
matplotlib.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
matplotlib.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
matplotlib.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
matplotlib.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
matplotlib.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

landcover_lookup = {
111: 'Closed forest, evergreen needle leaf',
113: 'Closed forest, deciduous needle leaf',
112: 'Closed forest, evergreen broad leaf',
114: 'Closed forest, deciduous broad leaf',
115: 'Closed forest, mixed',
116: 'Closed forest, unknown',
121: 'Open forest, evergreen needle leaf',
123: 'Open forest, deciduous needle leaf',
122: 'Open forest, evergreen broad leaf',
124: 'Open forest, deciduous broad leaf',
125: 'Open forest, mixed',
126: 'Open forest, unknown',
20: 'Shrubs'
}



closed_forest = [111,112,113,114,115,116]
open_forest = [121,122,123,124,125,126]
shrubs = [20]

landcover_groups = {
    'Closed Forest': closed_forest,
    'Open Forest': open_forest,
    'Shrubs': shrubs
}

landcover_group_colors = {
    'Closed Forest': '#67000d',
    'Open Forest': '#00441b',
    'Shrubs': '#6baed6'
}

landcover_colors = {
        111: '#67000d',
        113: '#a50f15',
        112: '#cb181d',
        114: '#ef3b2c',
        115: '#fb6a4a',
        116: '#fc9272',
        121: '#00441b',
        123: '#006d2c',
        122: '#238b45',
        124: '#41ab5d',
        125: '#74c476',
        126: '#a1d99b',
        20: '#6baed6'
    }

landcover_colors_by_name = {
'Closed forest, evergreen needle leaf':'#67000d',  
'Closed forest, deciduous needle leaf':'#a50f15', 
'Closed forest, evergreen broad leaf': '#cb181d', 
'Closed forest, deciduous broad leaf': '#ef3b2c', 
'Closed forest, mixed':                '#fb6a4a', 
'Closed forest, unknown':              '#fc9272', 
'Open forest, evergreen needle leaf':  '#00441b', 
'Open forest, deciduous needle leaf':  '#006d2c', 
'Open forest, evergreen broad leaf':   '#238b45', 
'Open forest, deciduous broad leaf':   '#41ab5d', 
'Open forest, mixed':                  '#74c476', 
'Open forest, unknown':                '#a1d99b', 
'Closed forest, mixed/unknown':        '#fb6a4a', 
'Open forest, mixed/unknown':          '#74c476',
'Shrubs':                               '#6baed6'
}
date_colors = {
    1: '#006837',
    2: '#1a9850',
    3: '#66bd63',
    4: '#a6d96a',
    5: '#d9ef8b',
    6: '#ffffbf',
    7: '#fee08b',
    8: '#fdae61',
    9: '#f46d43',
    10: '#d73027',
    11: '#a50026',
    12: '#690018'
}

###HELPER FUNCTIONS




#Merges together segments from the same date 
def manage_segment_merge(segment_holder, segment, merge_keys):
    segment_date = segment['date']
    if segment_date in segment_holder:
        cur_segment = segment_holder[segment_date]
        for k in merge_keys:
            to_merge = segment[k], cur_segment[k]
            cur_segment[k] = np.append(*to_merge)
    else:
        segment_holder[segment_date] = segment
    return segment_holder    

#For each segment or merged segment, call a plotting function


###PLOTTING FUNCTIONS
#Get armstron rhov/rhog function for a matplotlib axis
def armstron_per_ax(canopy_arr, terrain_arr, ax, reverse=False):
    if reverse:
        canopy_copy = canopy_arr.copy()
        canopy_arr = terrain_arr
        terrain_arr = canopy_copy
    fitted = LinearRegression().fit(canopy_arr.reshape(-1, 1), terrain_arr)

    beta_1 = fitted.coef_[0]
    y_int = fitted.intercept_
    ax.scatter(canopy_arr, terrain_arr)
    line = np.linspace(canopy_arr.min(), canopy_arr.max(), 100)
    ax.plot(line, line*beta_1 + y_int, c='red', ls='--')
    # ax.xaxis.set_label_position('top')
    # ax.set_xlabel(f'y = {beta_1:.4f}x+{y_int:.4f}')
    #ax.set_title(f'y = {beta_1:.4f}x+{y_int:.4f}')

    plt_title = r'Regression $\rho_v/\rho_g$: ' + f'{beta_1*-1:.4f}'
    ax.set_title(plt_title)
    return ax

#Put together all plots for rvg_per_segment
def create_rv_rg_plots(segment, save_dir, options):
    fig, ax = plt.subplots(2,3, figsize=(15, 8))
    ax = np.ravel(ax)

    ax[2].hist(segment['rv_rg_arr'])
    ax[2].set_title('Histograms\n')
    ax[5].hist(segment['rv_rg_arr_norm'])

    armstron_per_ax(segment['canopy'], segment['terrain'], ax[0])
    ax[0].set_title('Ground/Canopy')
    ax[0].set_ylabel('Photon Counts')
    armstron_per_ax(segment['canopy'], segment['terrain'], ax[1], reverse=True)
    ax[1].set_title('Canopy/Ground')

    armstron_per_ax(segment['canopy_norm'], segment['terrain_norm'], ax[3])
    ax[3].set_ylabel('Photon Rates')
    armstron_per_ax(segment['canopy_norm'], segment['terrain_norm'], ax[4], reverse=True)

    
    merged_text = f" - {segment['start']} - {segment['end']} - {segment['forest_type']}" if not options['merged'] else ''
    rv_rg_text = r'Photon Count: Mean $\rho_v/\rho_g$: ' + f"{np.mean(segment['rv_rg_arr']):.4f} | " + r"Median $\rho_v/\rho_g$: " + f"{np.median(segment['rv_rg_arr']):.4f}\n"
    rv_rg_text = rv_rg_text + r'Normalized: Mean $\rho_v/\rho_g$: ' + f"{np.mean(segment['rv_rg_arr_norm']):.4f} | " + r"Median $\rho_v/\rho_g$: " + f"{np.median(segment['rv_rg_arr_norm']):.4f}"
    segment_text = ' - 100m segments' if not options['subsegments'] else ' - 20m segments'
    title = segment['date'] + segment_text + merged_text +'\n' + rv_rg_text

    save_name = f"{segment['date']}{segment_text}{merged_text}.png".replace('/', '_')
    plt.suptitle(title)
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, save_name))
    plt.show()




    return None

def create_rv_rg_output(segment_holder, save_dir=None, options={}, fn=create_rv_rg_plots):

    for k, v in segment_holder.items():
        fn(v, save_dir, options)

    return None

def create_rv_rg_plot_with_pic(segment, save_dir, options):
    fig, axd = plt.subplot_mosaic((['upper_left', 'right'],
                                    ['lower_left', 'right']), figsize=(12,9))

    armstron_per_ax(segment['canopy_norm'], segment['terrain_norm'], axd['upper_left'], reverse=True)
    #axd['upper_left'].set_title('Canopy/Ground')
    axd['upper_left'].set_ylabel('Canopy Photon Rate')
    axd['upper_left'].set_xlabel('Ground Photon Rate')

    axd['lower_left'].hist(segment['rv_rg_arr_norm'])
    hist_title = r'Median $\rho_v/\rho_g$: ' + f'{np.median(segment["rv_rg_arr_norm"]):.4f}'
    axd['lower_left'].axvline(np.median(segment["rv_rg_arr_norm"]), c='red', ls='--')
    axd['lower_left'].set_title(hist_title)
    axd['lower_left'].set_ylabel('Count')
    axd['lower_left'].set_xlabel(r'$\rho_v/\rho_g$')

    axd['right'].get_xaxis().set_visible(False)
    axd['right'].get_yaxis().set_visible(False)
    axd['right'].set_box_aspect(0.74074074074)
    axd['right'].set_title('Phenocam Image')

    segment_text = ' - 100m segments' if not options['subsegments'] else '20m subsegments: '
    merged_text = f"{segment['start']} - {segment['end']}\n{segment['forest_type']}" if not options['merged'] else ' - Merged Landtypes'
    title = options['site'] + ' ' + segment['local_time'] + ' (AST)\n' + segment_text + merged_text

    plt.suptitle(title)


    plt.tight_layout()
    save_name = f"{segment['local_time']} {segment['start']} - {segment['end']}.png".replace('/', '_').replace(':','_')
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, save_name))
    plt.show()

def create_rv_rg_super_plot(segments: dict, save_dir: str, options: dict):
    matplotlib.use('Agg')
    num_segments = len(segments.items())

    fig = plt.figure(constrained_layout=True, figsize=(11, num_segments*3.5))
    subfigs = fig.subfigures(nrows=num_segments, ncols=1)
    seg_counter = 0
    pheno_loc = os.path.join(options['pheno_imgs'], options['site'])
    for segment in segments.values():
        segment_text = ' - 100m segments' if not options['subsegments'] else '20m subsegments: '
        merged_text = f"{segment['forest_type']}"
        title = options['site'] + ' ' + segment['local_time'] + ' (AST)\n' + segment_text + merged_text
        subfigs[seg_counter].suptitle(title)

        ax = subfigs[seg_counter].subplots(nrows=1, ncols=3)
        cur_ax = ax[0]
        armstron_per_ax(segment['canopy_norm'], segment['terrain_norm'], cur_ax, reverse=True)
        cur_ax.set_ylabel('Canopy Photon Rate')
        cur_ax.set_xlabel('Ground Photon Rate')

        cur_ax = ax[1]
        cur_ax.hist(segment['rv_rg_arr_norm'])
        hist_title = r'Median $\rho_v/\rho_g$: ' + f'{np.median(segment["rv_rg_arr_norm"]):.4f}'
        cur_ax.axvline(np.median(segment["rv_rg_arr_norm"]), c='red', ls='--')
        cur_ax.set_title(hist_title)
        cur_ax.set_ylabel('Count')
        cur_ax.set_xlabel(r'$\rho_v/\rho_g$')

        cur_ax = ax[2]
        p_img = plt.imread(os.path.join(pheno_loc, segment['orig_date'] + '.jpg'))
        cur_ax.imshow(p_img)

        cur_ax.get_xaxis().set_visible(False)
        cur_ax.get_yaxis().set_visible(False)

        cur_ax.set_title('Phenocam Image')

        seg_counter += 1

    #plt.tight_layout()
    save_name = f"{options['site']}.png"
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, save_name))
    #plt.show()





    pass


###ANALYSIS FUNCTIONS
#This is mega slow due to np.isin. Update or avoid using.
def split_ground_and_canopy(seg_number, 
                            photons, 
                            photon_segments, 
                            height_threshold, 
                            delta_time=None, 
                            seg_step=5):
    sub_segs = np.arange(seg_number, seg_number+seg_step, step=1)
    id_filter = np.isin(photon_segments, sub_segs)

    select_photons = photons[id_filter]
    
    ground_select = select_photons < height_threshold

    #Return canopy count, ground count
    if delta_time is None:
        return len(select_photons[~ground_select]), len(select_photons[ground_select])
    else:
        num_shots = np.unique(delta_time[id_filter]).shape[0]
        if num_shots == 0:
            return 0, 0
        return len(select_photons[~ground_select])/num_shots, len(select_photons[ground_select])/num_shots


def split_ground_and_canopy_preproc(data_dict,
                            param_keys,
                            height_threshold=2,
                            step_size = 1):

    out_dict = {pk:dict() for pk in param_keys}
    jday=data_dict['jday']
    #TODO: assumes keys. fix?
    for j in data_dict['ph_h'].keys():
        ph_segment_id = data_dict['ph_segment_id'][j]
        ph_h = data_dict['ph_h'][j]
        delta_time = data_dict['delta_time'][j]
        beams = data_dict['ph_h_beam_id'][j]
        if step_size == 1:
            all_segments, seg_idxs = np.unique(ph_segment_id, return_index=True)
        else:
            first_seg = ph_segment_id[0]
            # Change origin to first segment id for easy math
            zeroed = ph_segment_id - first_seg
            # Get difference above desired step size
            modded = zeroed % step_size
            # Subtract difference
            final = ph_segment_id - modded

            #If we are doing step_size 5 that probably means we want things to match up with 100m segments, so lets make sure everything is perfect
            #The aggregation technique can re-introduce previously removed segments if there were subsegments remaining
            #TODO: this could likely be worked into the logic of __reconcile__
            if step_size == 5 and 'segment_id_beg' in data_dict.keys():
                matches = np.isin(final, data_dict['segment_id_beg'][j])
                if matches.sum() != final.shape[0]:
                    final = final[matches]
                    ph_h = ph_h[matches]
                    delta_time = delta_time[matches]
                    beams = beams[matches]

            all_segments, seg_idxs = np.unique(final,  return_index=True)
        #Ive run out of clever ways to solve problems. We will just nix this file if it doesn't want to cooperate.
        if len(all_segments != 0):


            ground = np.zeros(all_segments.shape, dtype=np.float32)
            canopy = np.zeros_like(ground)

            new_beams = np.ones(all_segments.shape, dtype=np.uint8)
            new_beams = new_beams * beams[0]
            out_dict['gc_beam_ids'][j] = new_beams

            jdays = np.ones(all_segments.shape, dtype = np.uint64)
            jdays = jdays * jday
            out_dict['gc_jday'][j] = jdays

            for ix in range(len(all_segments)):
                start_idx = seg_idxs[ix]
                end_idx = seg_idxs[ix+1] if ix + 1 < len(all_segments) else len(ph_h)

                seg_heights = ph_h[start_idx:end_idx]
                num_shots = len(np.unique(delta_time[start_idx:end_idx]))
                if num_shots != 0:
                    ground_select = seg_heights < height_threshold
                    ground[ix] = np.sum(ground_select)/num_shots
                    canopy[ix] = np.sum(~ground_select)/num_shots

            out_dict['ground'][j] = ground
            out_dict['canopy'][j] = canopy
            out_dict['gc_seg_ids'][j] = all_segments
        else:
            return None
        
    return out_dict



def robust_rvrg(data, **kwargs):
    #data = kwargs['data']

    if 'ground' in data and 'canopy' in data:
        return robust_rvrg_with_pre(data, **kwargs)
    else:
        return robust_rvrg_post(data, **kwargs)

def robust_rvrg_with_pre(data,
                    seg_ids = None,
                    beam = None, 
                    seg_step = 5,
                    **kwargs):
    ground = data['ground']
    canopy = data['canopy']
    gc_segs = data['gc_seg_ids']
    gc_beams = data['gc_beam_ids']

    if beam is not None:
        beam_filter = gc_beams == beam
        ground = ground[beam_filter]
        canopy = canopy[beam_filter]
        gc_segs = gc_segs[beam_filter]
    
    if seg_ids is not None:
        seg_filter = np.isin(gc_segs, seg_ids)
        gc_segs = gc_segs[seg_filter]
        ground = ground[seg_filter]
        canopy = canopy[seg_filter]


    rvrg_out = []

    for ix, seg_number in np.ndenumerate(gc_segs):
        rv_0 = canopy[ix]
        rg_0 = ground[ix]

        next_index = ix[0] + 1
        #Check to see if:
            #We are not leaving the array, the next segment in the array is contiguous, and we have not switched beams
        if next_index<len(gc_segs) and gc_segs[next_index] == seg_number+seg_step and gc_beams[next_index] == gc_beams[ix]:
            rv_1 = canopy[next_index]
            rg_1 = ground[next_index]

            denom = rg_1-rg_0
            if denom != 0:
                to_append = (rv_0-rv_1)/denom
                rvrg_out.append(to_append)
    
    return {'rvrg': np.array(rvrg_out),
            'canopy': canopy,
            'terrain': ground}


def robust_rvrg_post(data: dict, 
                height_threshold: int = 2, 
                height_min: int = -2, 
                height_max: int = 75, 
                seg_ids: np.ndarray = None, 
                norm: bool = False, 
                beam: int = None,
                seg_step: int = 5):

    ph_h = data['ph_h']
    height_filter = (ph_h > height_min) & (ph_h<height_max)
    if beam is not None:
        beam_filter = data['ph_h_beam_id'] == beam
        photon_filter = np.logical_and(beam_filter, height_filter)
    else:
        photon_filter = height_filter
    ph_h = ph_h[photon_filter]
    ph_segments = data['ph_segment_id']
    ph_segments = ph_segments[photon_filter]

    if seg_ids is None:
        segment_ids = data['segment_id_beg']
    else:
        segment_ids = seg_ids

    rvrg_out = []
    canopy_out = np.zeros(segment_ids.shape, dtype=np.float32)
    terrain_out = np.zeros(segment_ids.shape, dtype=np.float32)

    for ix, seg_number in np.ndenumerate(segment_ids):
        #num_shots = 
        if not norm:
            rv_0, rg_0 = split_ground_and_canopy(seg_number, ph_h, ph_segments, height_threshold, seg_step=seg_step)
        else:
            rv_0, rg_0 = split_ground_and_canopy(seg_number, ph_h, ph_segments, height_threshold, delta_time=data['delta_time'][photon_filter], seg_step=seg_step)
        canopy_out[ix]=rv_0
        terrain_out[ix] = rg_0
        if ix[0]+1 < len(segment_ids):
            next_seg_number = segment_ids[ix[0]+1]
            if (seg_number+seg_step) == next_seg_number:
                if not norm:
                    rv_1, rg_1 = split_ground_and_canopy(next_seg_number, ph_h, ph_segments, height_threshold, seg_step=seg_step)
                else:
                    rv_1, rg_1 = split_ground_and_canopy(next_seg_number, ph_h, ph_segments, height_threshold, delta_time=data['delta_time'][photon_filter], seg_step=seg_step)
                denom = rg_1-rg_0
                if denom != 0:
                    to_append = (rv_0-rv_1)/denom
                    rvrg_out.append(to_append)

    return {'rvrg': np.array(rvrg_out),
            'canopy': canopy_out,
            'terrain': terrain_out
            }


#MAIN FUNCTIONS
#Make a kml from given Icesat files
def make_kml(base_dir,
            out_file,
            save_dir,
            cam_coords=None,
            key=True):

    ice_files = IceSatFiles(
        base_dir=base_dir,
        groups = [
            ['land_segments', 'segment_landcover'],
            ['land_segments', 'segment_id_beg'],
            ['land_segments', 'latitude'],
            ['land_segments', 'longitude']
        ],
        filters = [
            #(['land_segments', 'msw_flag'],'<=', 0),
            (['land_segments', 'segment_landcover'], 'isin', list(landcover_lookup.keys())),
        ]
    )

    data_holder = ice_files.get_all_files()

    #Set up cam style
    cam_style = simplekml.Style(iconstyle=simplekml.IconStyle(icon=simplekml.Icon(href='http://maps.google.com/mapfiles/kml/shapes/camera.png')))

    #Tree styles
    tree_styles = {
        111: simplekml.Style(iconstyle=simplekml.IconStyle(icon=simplekml.Icon(href='http://maps.google.com/mapfiles/kml/paddle/red-circle.png'))),
        113: simplekml.Style(iconstyle=simplekml.IconStyle(icon=simplekml.Icon(href='http://maps.google.com/mapfiles/kml/paddle/red-diamond.png'))),
        112: simplekml.Style(iconstyle=simplekml.IconStyle(icon=simplekml.Icon(href='http://maps.google.com/mapfiles/kml/paddle/red-square.png'))),
        114: simplekml.Style(iconstyle=simplekml.IconStyle(icon=simplekml.Icon(href='http://maps.google.com/mapfiles/kml/paddle/red-stars.png'))),
        115: simplekml.Style(iconstyle=simplekml.IconStyle(icon=simplekml.Icon(href='http://maps.google.com/mapfiles/kml/paddle/red-blank.png'))),
        116: simplekml.Style(iconstyle=simplekml.IconStyle(icon=simplekml.Icon(href='http://maps.google.com/mapfiles/kml/paddle/red-blank.png'))),
        121: simplekml.Style(iconstyle=simplekml.IconStyle(icon=simplekml.Icon(href='http://maps.google.com/mapfiles/kml/paddle/grn-circle.png'))),
        123: simplekml.Style(iconstyle=simplekml.IconStyle(icon=simplekml.Icon(href='http://maps.google.com/mapfiles/kml/paddle/grn-diamond.png'))),
        122: simplekml.Style(iconstyle=simplekml.IconStyle(icon=simplekml.Icon(href='http://maps.google.com/mapfiles/kml/paddle/grn-square.png'))),
        124: simplekml.Style(iconstyle=simplekml.IconStyle(icon=simplekml.Icon(href='http://maps.google.com/mapfiles/kml/paddle/grn-stars.png'))),
        125: simplekml.Style(iconstyle=simplekml.IconStyle(icon=simplekml.Icon(href='http://maps.google.com/mapfiles/kml/paddle/grn-blank.png'))),
        126: simplekml.Style(iconstyle=simplekml.IconStyle(icon=simplekml.Icon(href='http://maps.google.com/mapfiles/kml/paddle/grn-blank.png'))),
        20: simplekml.Style(iconstyle=simplekml.IconStyle(icon=simplekml.Icon(href='http://maps.google.com/mapfiles/kml/paddle/blu-blank.png'))),
    }


    kml = simplekml.Kml()
    if cam_coords is not None:
        pnt = kml.newpoint(coords=cam_coords, name='Phenocam', description='Phenocam')
        pnt.style = cam_style
    
    if key:
        screen = kml.newscreenoverlay(name='Key')
        screen.icon.href = 'https://antalb.com/assets/kml_key.png'
        screen.overlayxy = simplekml.OverlayXY(x=0,y=1,xunits=simplekml.Units.fraction,
                                       yunits=simplekml.Units.fraction)
        screen.screenxy = simplekml.ScreenXY(x=15,y=15,xunits=simplekml.Units.pixels,
                                            yunits=simplekml.Units.insetpixels)
        screen.size.x = -1
        screen.size.y = -1
        screen.size.xunits = simplekml.Units.fraction
        screen.size.yunits = simplekml.Units.fraction


    for k, v in data_holder.items():
        lats = v['latitude']
        lons = v['longitude']
        day_of = k
        seg_ids = v['segment_id_beg']
        seg_cover = v['segment_landcover']
        beams= v['latitude_beam_id']
        for i in range(len(lats)):
            pnt = kml.newpoint(
                coords=[(lons[i], lats[i])],
                name= f'{seg_ids[i]}-{beams[i]}',
                description = f'{day_of} - {landcover_lookup[seg_cover[i]]}'
            )
            pnt.style = tree_styles[seg_cover[i]]
    kml.save(os.path.join(save_dir, out_file))

#Make an rhov/rhog plot given Icesat segments

def rvg_per_segment(
    base_dir,
    segments,
    height_threshold,
    use_subsegments = False,
    height_min = -2,
    height_max = 75,
    save_dir = None,
    merge_same_day=False,
    plt_fn = create_rv_rg_plots,
    one_plot_per_segment = True,
    sitename='',
    pheno_imgs = None
):
    ice_files = IceSatFiles(
        base_dir=base_dir,
        groups = [
            ['signal_photons', 'ph_h'],
            ['signal_photons', 'ph_segment_id'],
            ['land_segments', 'segment_id_beg'],
            ['signal_photons', 'delta_time']
        ],
        filters = [
            (['land_segments', 'msw_flag'],'<=', 0),
            #(['land_segments', 'night_flag'], '==', 1),
            (['land_segments', 'segment_landcover'], 'isin', list(landcover_lookup.keys())),
            (['signal_photons', 'd_flag'], '==', 1),
            (['signal_photons', 'ph_h'], '>', height_min),
            (['signal_photons', 'ph_h'], '<', height_max)
        ],
        new_params=[NewIceParam(['ground', 'canopy', 'gc_seg_ids', 'gc_beam_ids', 'gc_jday'], split_ground_and_canopy_preproc, {"height_threshold":2})]
    )

    data_holder = ice_files.get_all_files()
    
    segment_holder = dict()

    for segment in segments:
        #I am a hack
        segment['orig_date'] = segment['date']
        
        segment['date'] = datetime.strptime(segment['date'], '%Y%m%d')
        cur_data = data_holder[segment['date']]
        segment['local_time'] = cur_data['local_time']
        beam = segment['beam']

        step_size = 1 if use_subsegments else 5
            
        segment_ids = np.arange(segment['start'], segment['end']+5, step=step_size)
        rvrg_norm = robust_rvrg(cur_data, seg_ids=segment_ids, height_min=height_min, height_max=height_max, height_threshold=height_threshold, norm=True, beam=beam, seg_step=step_size)
        rvrg = robust_rvrg(cur_data, seg_ids=segment_ids, height_min=height_min, height_max=height_max, height_threshold=height_threshold, norm=False, beam=beam, seg_step=step_size)


        segment['terrain'] = rvrg['terrain']
        segment['canopy'] = rvrg['canopy']
        segment['canopy_norm'] = rvrg_norm['canopy']
        segment['terrain_norm'] = rvrg_norm['terrain']
        segment['rv_rg_arr'] = rvrg['rvrg']
        segment['rv_rg_arr_norm'] = rvrg_norm['rvrg']
        segment['segment_ids'] = segment_ids

        if merge_same_day:
            merge_keys = ['terrain', 'canopy', 'canopy_norm', 'terrain_norm', 'rv_rg_arr', 'rv_rg_arr_norm', 'segment_ids']
            segment_holder = manage_segment_merge(segment_holder, segment, merge_keys)
        else:
            segment_holder[f'{segment["date"]}_{segment["beam"]}_{segment["start"]}'] = segment

    if one_plot_per_segment:
        create_rv_rg_output(segment_holder, save_dir, {'merged': merge_same_day, 'subsegments': use_subsegments, 'site': sitename}, fn=plt_fn)
    else:
        plt_fn(segment_holder, save_dir, {'merged': merge_same_day, 'subsegments': use_subsegments, 'site': sitename, 'pheno_imgs': pheno_imgs})

def rvg_per_month(
    base_dir,
    save_dir = None,
    title = "",
    landcover_split = [
    FilterGroup('Closed forest, evergreen needle leaf', ['land_segments', 'segment_landcover'], 'isin',  [111]),
    FilterGroup('Closed forest, mixed/unknown',  ['land_segments', 'segment_landcover'], 'isin', [115, 116]),
    FilterGroup('Open forest, mixed/unknown', ['land_segments', 'segment_landcover'], 'isin', [125, 126]),
    FilterGroup('Shrubs', ['land_segments', 'segment_landcover'], 'isin', [20])
    ],
    height_min = -2,
    height_max = 75,
    seg_step = 5
    ):

    ice_files = IceSatFiles(
        base_dir=base_dir,
        #Data we want
        groups = [
            ['signal_photons', 'ph_h'],
            ['signal_photons', 'ph_segment_id'],
            ['land_segments', 'segment_id_beg'],
            ['signal_photons', 'delta_time'],
            ['land_segments', 'segment_landcover'],
            ['land_segments', 'canopy', 'segment_cover']
        ],
        #Filters to use on data
        filters = [
            (['land_segments', 'msw_flag'],'<=', 0),
            #(['land_segments', 'night_flag'], '==', 1),
            (['land_segments', 'segment_landcover'], 'isin', list(landcover_lookup.keys())),
            (['signal_photons', 'd_flag'], '==', 1),
            (['signal_photons', 'ph_h'], '>', height_min),
            (['signal_photons', 'ph_h'], '<', height_max)
        ],
        new_params=[NewIceParam(['ground', 'canopy', 'gc_seg_ids', 'gc_beam_ids', 'gc_jday'], split_ground_and_canopy_preproc, {"height_threshold":2, "step_size": seg_step})],
        filter_groups = landcover_split,
        minimum_segments=5

    )

    all_months = ice_files.get_all_files(by='month')

    median_keeper = {lc.filter_key:{'value':[],'month':[]} for lc in landcover_split}
    fig, axs = plt.subplots(3, 4, figsize=(11, 7.5), sharex=True, sharey=True)
    axs = axs.flat
    handles = []
    labels = []

    for k, v in all_months.items():
        cur_month = k
        ax = axs[cur_month - 1]
        ax.set_xlim((0, 100))
        ax.set_ylim((0, 1))
        ax.set_title(months_lookup[cur_month])
        #ax.set_aspect(1.0)

        if landcover_split is not None:
            for lc in landcover_split[::-1]:
                key = lc.filter_key
                cur_data = v[key]
                c = landcover_colors_by_name[key]

                rv_rg_results = robust_rvrg(cur_data, seg_step=seg_step)
                canopy = rv_rg_results['canopy']
                terrain = rv_rg_results['terrain']
                median_rv_rg = np.median(rv_rg_results['rvrg'])
                median_keeper[key]['value'].append(median_rv_rg)
                median_keeper[key]['month'].append(cur_month)

                y = canopy/((median_rv_rg*terrain) + canopy)

                ax.scatter(cur_data['segment_cover'], y, marker='.', label='\n'.join(wrap(key, 20)), c=[c]*len(canopy), alpha=0.8)
                

                cur_handles, cur_labels = ax.get_legend_handles_labels()
                for h, l in zip(cur_handles, cur_labels):
                    if l not in labels:
                        labels.append(l)
                        handles.append(h)
        
    fig.supxlabel('Copernicus Vegetation Cover', ha='center', 
    #y=.125
    )
    fig.supylabel('ICESat-2 Vegetation Cover')
    fig.legend(handles, labels, 
    loc='right', 
    #bbox_to_anchor=(1, 0.5),
    markerscale=2,
    frameon=False,
    title='Copernicus\nLandcover Type'
    )
    fig.suptitle(title, linespacing=1.5)
    plt.subplots_adjust(right=0.8)
    plt.subplots_adjust(top=0.85)
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, title.replace("\n", " - ")))
    #plt.tight_layout()
    #plt.show()

def rvg_per_date(
    base_dir,
    plt_title = None,
    out_dir = None,
    landcover_split: list[FilterGroup] = [
    FilterGroup('Closed forest, evergreen needle leaf', ['land_segments', 'segment_landcover'], 'isin',  [111]),
    FilterGroup('Closed forest, mixed/unknown',  ['land_segments', 'segment_landcover'], 'isin', [115, 116]),
    FilterGroup('Open forest, mixed/unknown', ['land_segments', 'segment_landcover'], 'isin', [125, 126]),
    FilterGroup('Shrubs', ['land_segments', 'segment_landcover'], 'isin', [20])
    ],
    norm = True,
    height_min = -2,
    height_max = 75,
    seg_step = 1,
    ):

    ice_files = IceSatFiles(
        base_dir=base_dir,
        #Data we want
        groups = [
            ['signal_photons', 'ph_h'],
            ['signal_photons', 'ph_segment_id'],
            ['land_segments', 'segment_id_beg'],
            ['signal_photons', 'delta_time'],
            ['land_segments', 'segment_landcover'],
            ['land_segments', 'canopy', 'segment_cover']
        ],
        #Filters to use on data
        filters = [
            (['land_segments', 'msw_flag'],'<=', 0),
            #(['land_segments', 'night_flag'], '==', 1),
            (['land_segments', 'segment_landcover'], 'isin', list(landcover_lookup.keys())),
            (['signal_photons', 'd_flag'], '==', 1),
            (['signal_photons', 'ph_h'], '>', height_min),
            (['signal_photons', 'ph_h'], '<', height_max)
        ],
        #New groups created with filters
        filter_groups = landcover_split,
        new_params=[NewIceParam(['ground', 'canopy', 'gc_seg_ids', 'gc_beam_ids', 'gc_jday'], split_ground_and_canopy_preproc, {"height_threshold":2, "step_size": seg_step})],
        minimum_segments=5

    )

    all_dates = ice_files.get_all_files(by='date')
    unmerged = ice_files.get_all_files(by='date', custom_groups=False)

    group_holders = {g.filter_key:[] for g in landcover_split}
    date_holder = []
    overall_holder = []

    for k, v in all_dates.items():
        #cur_date = k
        date_holder.append(k)

        overall_rvrg = robust_rvrg(unmerged[k], norm=norm, seg_step=seg_step)
        overall_holder.append(np.median(overall_rvrg['rvrg']))

        for fg in landcover_split:
            fg_k = fg.filter_key
            fg_rvrg = robust_rvrg(v[fg_k], norm=norm, seg_step=seg_step)
            group_holders[fg_k].append(np.median(fg_rvrg['rvrg']))
    
    fig, ax = plt.subplots(5,1, figsize=(9, 11))
    ax = ax.flatten()
    date_holder = np.array(date_holder)
    overall_holder = np.array(overall_holder)

    get_f = lambda x: np.logical_and(x>0.0, x<1.0)

    f = get_f(overall_holder)

    yticks = [0.0, 0.25, 0.5, 0.75, 1.0]

    ax[0].plot(date_holder[f], overall_holder[f],'.-')
    ax[0].set_title('All Landtypes')
    ax[0].set_ylim(0,1)
    ax[0].set_xlim(datetime(2018, 10, 1), datetime(2022, 11, 1))
    ax[0].set_yticks(yticks)
    i = 1
    for k, v in group_holders.items():
        v = np.array(v)
        f= get_f(v)
        ax[i].plot(date_holder[f], v[f],'.--', c=landcover_colors_by_name[k])
        ax[i].set_title(k)
        ax[i].set_ylim(0,1)
        ax[i].set_yticks(yticks)
        ax[i].set_xlim(datetime(2018, 10, 1), datetime(2022, 11, 1))
        ax[i].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,4,7,10)))
        ax[i].xaxis.set_minor_locator(mdates.MonthLocator())
        ax[i].xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax[i].xaxis.get_major_locator()))
        i += 1


    #ax[0].get_shared_x_axes().join(*ax)
    ax[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1,4,7,10)))
    ax[0].xaxis.set_minor_locator(mdates.MonthLocator())
    ax[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax[0].xaxis.get_major_locator()))
    #fig.legend()
    #ax.set_ylim(-1,1)
    fig.text(0.00, 0.5, r"Median $\rho_v/\rho_g$", va='center', rotation='vertical')
    if plt_title is not None:
        fig.suptitle(plt_title)
    plt.tight_layout()
    if out_dir is not None:
        plt.savefig(os.path.join(out_dir, plt_title.replace("\n", " - ")))
    #plt.show()
    plt.close()
        
    




if __name__ == '__main__':


    folders = {'6': '_cam',
               '20': '_10km',
               '100': '_50km',
               '200': '_100km'}
    #sizes = ['6', '20', '100', '200']
    sizes = ['100', '200']

    names = ['BONA', 'DEJU']
    steps = [1, 5]

    # all_combos = list(product(sizes, names))
    # for c in all_combos:
    #     rvg_per_month(os.path.join('/data/shared/rsdata/lidar/icesat/', c[1] + folders[c[0]]),
    #             save_dir = '/data/anthony/icesat/rv_rg_per_month',
    #             title = f"ICESat-2 Vegetation Cover vs Copernicus Vegetation Cover\n{c[1]} - {c[0]}km by {c[0]}km Bounding Box Centered Around Phenocam",
    #             )
    all_combos = list(product(sizes, names, steps))

    for c in all_combos:
        rvg_per_date(
            os.path.join('/data/shared/rsdata/lidar/icesat/', c[1] + folders[c[0]]),
            plt_title = f"{c[1]} - {c[0]}km by {c[0]}km Bounding Box Centered Around Phenocam\n{c[2]*20}m Segments",
            out_dir='/data/anthony/icesat/rv_rg_per_date',
            norm=True,
            seg_step= c[2]
        )
        

    


    BONA_segments = [
            {'date': '20200922', 'beam': 2, 'start': 639379, 'end': 639424, 'forest_type':'Closed Forest, unknown/mixed'},
            {'date': '20200922', 'beam': 2, 'start': 639459, 'end': 639509, 'forest_type':'Closed Forest, evergreen needle leaf'},
            {'date': '20210208', 'beam': 1, 'start': 362367, 'end': 362377, 'forest_type':'Closed Forest, unknown/mixed'},
            {'date': '20210208', 'beam': 1, 'start': 362262, 'end': 362292, 'forest_type':'Closed Forest, evergreen needle leaf'},
            {'date': '20210208', 'beam': 1, 'start': 362202, 'end': 362257, 'forest_type':'Closed Forest, unknown/mixed'},
            {'date': '20210322', 'beam': 2, 'start': 639375, 'end': 639425, 'forest_type':'Closed Forest, unknown/mixed'},
            {'date': '20210322', 'beam': 2, 'start': 639460, 'end': 639485, 'forest_type':'Closed Forest, evergreen needle leaf'},
            {'date': '20220321', 'beam': 2, 'start': 639381, 'end': 639406, 'forest_type':'Closed Forest, unknown/mixed'},
            {'date': '20220321', 'beam': 2, 'start': 639461, 'end': 639476, 'forest_type':'Closed Forest, evergreen needle leaf'},     
            {'date': '20220606', 'beam': 1, 'start': 362471, 'end': 362556, 'forest_type':'Closed Forest, unknown/mixed'},
        ]

    DEJU_segments = [
        {'date': '20181105', 'beam': 1, 'start': 355320, 'end': 355380, 'forest_type':'Open Forest, unknown/mixed'},
        {'date': '20190516', 'beam': 2, 'start': 646624, 'end': 646704, 'forest_type':'Closed Forest, evergreen needle leaf'},
        {'date': '20200503', 'beam': 2, 'start': 355183, 'end': 355228, 'forest_type':'Closed Forest, unknown/mixed'},
        {'date': '20210502', 'beam': 2, 'start': 355138, 'end': 355193, 'forest_type':'Closed Forest, evergreen needle leaf'},
        {'date': '20210502', 'beam': 2, 'start': 355298, 'end': 355323, 'forest_type':'Closed Forest, evergreen needle leaf'},
        {'date': '20210801', 'beam': 1, 'start': 355235, 'end': 355320, 'forest_type':'Closed Forest, evergreen needle leaf'},        
        {'date': '20210801', 'beam': 1, 'start': 355175, 'end': 355230, 'forest_type':'Closed Forest, unknown/mixed'},
        {'date': '20220430', 'beam': 3, 'start': 355220, 'end': 355265, 'forest_type':'Closed Forest, unknown/mixed'},
        {'date': '20220430', 'beam': 2, 'start': 355192, 'end': 355307, 'forest_type':'Closed Forest, evergreen needle leaf'},
        {'date': '20220430', 'beam': 2, 'start': 355102, 'end': 355152, 'forest_type':'Closed Forest, evergreen needle leaf'},
        ]

# rvg_per_segment(
#         base_dir = '/data/shared/rsdata/lidar/icesat/DEJU_cam',
#         segments = DEJU_segments,
#         height_threshold=2,
#         use_subsegments=True,
#         save_dir = '/data/anthony/icesat/',
#         merge_same_day=False,
#         plt_fn=create_rv_rg_super_plot,
#         one_plot_per_segment = False,
#         sitename='DEJU',
#         pheno_imgs = '/data/shared/rsdata/lidar/icesat/phenocam_imgs'
#     )
