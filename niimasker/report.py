import os
import json
import matplotlib
from jinja2 import Template, Environment, FileSystemLoader

from niimasker.plots import (plot_region_overlay, plot_coord_overlay, 
                             plot_timeseries, plot_connectome, 
                             plot_regressor_corr)

pjoin = os.path.join

def generate_report(func_image, output_dir):

    # initialize directories
    report_dir = pjoin(output_dir, 'reports')
    fig_dir = pjoin(report_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    func_img_name = os.path.basename(func_image.fname).split('.')[0]
    fig_fname_base = pjoin(fig_dir, func_img_name)
    
    n_rois = func_image.timeseries.shape[1]

    connectome_fig = None
    reg_corr_fig = None

    if func_image.masker_type == 'NiftiMasker':
        roi_cmap = matplotlib.cm.get_cmap('gist_yarg')
        ylabel = 'Voxel'

        overlay_fig = plot_region_overlay(func_image.roi_img, func_image.img,
                                          fig_fname_base, cmap=roi_cmap)
        ts_fig = plot_timeseries(func_image.timeseries, fig_fname_base, roi_cmap, 
                                 ylabel)

    else:
        roi_cmap = 'nipy_spectral' if n_rois > 1 else 'autumn'
        roi_cmap = matplotlib.cm.get_cmap(roi_cmap)
        ylabel = 'Region'

        overlay_fig = plot_region_overlay(func_image.roi_img, func_image.img,
                                            fig_fname_base, cmap=roi_cmap)
        ts_fig = plot_timeseries(func_image.timeseries, fig_fname_base, roi_cmap, 
                                    ylabel)

        if n_rois > 1:
            connectome_fig = plot_connectome(func_image.timeseries, fig_fname_base,
                                             tick_cmap=roi_cmap)
            if func_image.regressors is not None:
                reg_corr_fig = plot_regressor_corr(func_image.timeseries,
                                                   func_image.regressors,
                                                   fig_fname_base, 
                                                   cmap=roi_cmap)

    param_file = os.path.join(output_dir, 'niimasker_data/parameters.json')
    with open(param_file, 'r') as f:
        parameters = json.load(f)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_loader = FileSystemLoader(os.path.join(dir_path, 'templates'))
    env = Environment(loader=file_loader)
    template = env.get_template('base.html')
    output = template.render(title=func_img_name,
                             parameters=parameters,
                             func_img=func_image.fname,
                             masker_type=func_image.masker_type,
                             regressor_file=func_image.regressor_file,
                             regressors=parameters['parameters']['regressors'],
                             overlay_fig=overlay_fig,
                             timeseries_fig=ts_fig,
                             connectome_fig=connectome_fig,
                             regressor_fig=reg_corr_fig
                             )

    save_file = os.path.join(report_dir, '{}_report.html'.format(func_img_name))
    with open(save_file, "w") as f:
        f.write(output)
