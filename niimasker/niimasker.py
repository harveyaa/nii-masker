"""Core module that contains all functions related to extracting out time
series data.
"""

import os
import warnings
import pydra
import load_confounds
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import load_img, math_img, resample_to_img
from nilearn.input_data import (NiftiMasker, NiftiLabelsMasker, NiftiSpheresMasker)
from nilearn.input_data.nifti_spheres_masker import _apply_mask_and_get_affinity
from niimasker.report import generate_report

class FunctionalImage(object):
    def __init__(self, fname):
        self.fname = fname
        img = nib.load(self.fname)
        self.img = img
        self.regressors = None
        self.regressor_file = None
    
    def set_regressors(self, regressor_fname, denoiser = None):
        """Create regressors for masking"""
        self.regressor_file = regressor_fname
        
        if denoiser is not None:
            self.regressors = denoiser.load(self.regressor_file)
        else:
            self.regressors = pd.read_csv(regressor_fname, sep=r'\t',engine='python').values
                    
    def discard_scans(self, n_scans):
        # crop scans from image
        arr = self.img.get_data()
        arr = arr[:, :, :, n_scans:]
        self.img = nib.Nifti1Image(arr, self.img.affine)

        if self.regressors is not None:
            # crop from regressors
            self.regressors = self.regressors[n_scans:, :]


    def extract(self, masker, as_voxels=False, labels=None):
        print('  Extracting from {}'.format(os.path.basename(self.fname)))

        timeseries = masker.fit_transform(self.img,confounds=self.regressors)

        # determine column names for timeseries
        if isinstance(masker, NiftiMasker):
            labels = ['voxel {}'.format(int(i)) for i in np.arange(timeseries.shape[1])]
            self.roi_img = masker.mask_img_
            self.masker_type = 'NiftiMasker'
            
        elif isinstance(masker, NiftiLabelsMasker):
            if labels is None:
                labels = ['roi {}'.format(int(i)) for i in masker.labels_]
            self.roi_img = masker.labels_img
            self.masker_type = 'NiftiLabelsMasker'

        elif isinstance(masker, NiftiSpheresMasker):
            if labels is None:
                labels = ['roi {}'.format(int(i)) for i in range(len(masker.seeds))]
            self.roi_img = masker.spheres_img
            self.masker_type = 'NiftiSpheresMasker'

        self.masker = masker
        self.data = pd.DataFrame(timeseries, columns=[str(i) for i in labels])


# MASKING FUNCTIONS
def _get_spheres_from_masker(masker, img):
    """Re-extract spheres from coordinates to make niimg. 

    Note that this will take a while, as it uses the exact same function that
    nilearn calls to extract data for NiftiSpheresMasker
    """
    ref_img = nib.load(img) 
    ref_img = nib.Nifti1Image(ref_img.get_fdata()[:, :, :, [0]], ref_img.affine)

    X, A = _apply_mask_and_get_affinity(masker.seeds, ref_img, masker.radius, 
                                        masker.allow_overlap)
    # label sphere masks
    spheres = A.toarray()
    spheres *= np.arange(1, len(masker.seeds) + 1)[:, np.newaxis]

    # combine masks, taking the maximum if overlap occurs
    arr = np.zeros(spheres.shape[1])
    for i in np.arange(spheres.shape[0]):
        arr = np.maximum(arr, spheres[i, :])
    arr = arr.reshape(ref_img.shape[:-1])
    spheres_img = nib.Nifti1Image(arr, ref_img.affine)
    
    if masker.mask_img is not None:
        mask_img_ = resample_to_img(masker.mask_img, spheres_img)
        spheres_img = math_img('img1 * img2', img1=spheres_img, img2=mask_img_)

    return spheres_img


def _read_coords(roi_file):
    """Parse and validate coordinates from file"""
    coords = pd.read_table(roi_file)
    
    # validate columns
    columns = [x for x in coords.columns if x in ['x', 'y', 'z']]
    if (len(columns) != 3) or (len(np.unique(columns)) != 3):
        raise ValueError('Provided coordinates do not have 3 columns with names `x`, `y`, and `z`')

    # convert to list of lists for nilearn input
    return coords.values.tolist()

def _masker_from_coords(roi,input_files,output_dir,**kwargs):
    #makes a new mask from the coords and save it
    n_rois = len(roi)
    print('{} region(s) detected from coordinates'.format(n_rois))
        
    if kwargs.get('radius') is None:
        warnings.warn('No radius specified for coordinates; setting to nilearn.input_data.NiftiSphereMasker default '
                        'of extracting from a single voxel')
        
    masker = NiftiSpheresMasker(roi, **kwargs)
    masker.spheres_img = _get_spheres_from_masker(masker, input_files[0])
    masker.spheres_img.to_filename(os.path.join(output_dir, 'niimasker_data','spheres_image.nii.gz'))
    return masker

def _set_masker(roi_file, input_files, output_dir, as_voxels=False, **kwargs):
    """Check and see if multiple ROIs exist in atlas file"""
    # 1) NIfTI image that is an atlas
    # 2) query string formatted as `nilearn:<atlas-name>:<atlas-parameters>
    # 3) a file path to a .tsv file that contains roi_file coordinates in MNI space
    
    # OPTION 3
    if roi_file.endswith('.tsv'):
        masker = _masker_from_coords(roi,input_files,output_dir,**kwargs)
    
    # OPTION 1 & 2
    else:
        roi = load_img(roi_file)
        n_rois = len(np.unique(roi.get_data())) - 1
        print('  {} region(s) detected from {}'.format(n_rois,roi.get_filename()))

        if 'radius' in kwargs:
            kwargs.pop('radius')
        
        if 'allow_overlap' in kwargs:
            kwargs.pop('allow_overlap')
        
        if (n_rois == 0):
            raise ValueError('No ROI detected; check ROI file')
        elif (n_rois == 1) & as_voxels:
            if 'mask_img' in kwargs:
                kwargs.pop('mask_img')
            masker = NiftiMasker(roi, **kwargs)
        else:
            masker = NiftiLabelsMasker(roi, **kwargs)
    
    return masker

def _make_denoiser(denoising_strategy):
    if denoising_strategy is None:
        denoiser = None
    #predefined strategy
    elif (len(denoising_strategy)==1):
        denoising_strategy = denoising_strategy[0]
        if denoising_strategy in ['Params2','Params6','Params9','Params24','Params36','AnatCompCor','TempCompCor']:
            denoiser = eval('load_confounds.{}()'.format(denoising_strategy))
        else:
            raise ValueError('Provided denoising strategy is not recognized.')
    #flexible strategy
    else:
        if set(denoising_strategy) <= set(['motion','high_pass','wm_csf', 'compcor', 'global']):
            denoiser = load_confounds.Confounds(strategy=denoising_strategy)
        else:
            raise ValueError('Provided denoising strategy is not recognized.')
    return denoiser
    
def _mask_and_save(masker, denoiser, img_name, output_dir, regressor_file=None,
                   labels=None, discard_scans=None):
    """Runs the full masking process and saves output for a single image;
    the main function used by `make_timeseries`
    """
    img = FunctionalImage(img_name)

    if regressor_file is not None:
        img.set_regressors(regressor_file, denoiser)
        
    if discard_scans is not None:
        if discard_scans > 0:
            img.discard_scans(discard_scans)

    img.extract(masker, as_voxels=as_voxels, labels=labels)

    # export data and report
    out_fname = os.path.basename(img.fname).split('.')[0] + '_timeseries.tsv'
    img.data.to_csv(os.path.join(output_dir, out_fname), sep='\t', index=False,
                    float_format='%.8f')
    generate_report(img, output_dir)
    #RETURN THE TIMESERIES
    return img.data

