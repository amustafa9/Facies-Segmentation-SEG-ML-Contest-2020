# File contains utility functions for various visualization, training, and evaluation purposes

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from mayavi import mlab

def image_crop(Im, height, width):
    """Generate 10 different versions of image crops of a given seismic section

    Parameters
    ----------
    Im: ndarray
       2D ndarray representing the seismic section desired to be cropped.

    height: int
          An integer specifying the height of the cropped sections

    width: int
         An integer specifying the width of the cropped sections

    Returns
    -------
    images: tuple
          A tuple of ndarrays representing crops is returned in this order: top-left,
           top-right, bottom-left, bottom-right, center, and then the 5 horizontally flipped versions"""

    # Obtain corner crops
    top_left = Im[:height, :width]
    top_right = Im[:height, -width:]
    bottom_left = Im[-height:, :width]
    bottom_right = Im[-height:, -width:]

    # Obtain the center crop
    Im_h, Im_w = Im.shape
    start_height = Im_h // 2 - height // 2
    start_width = Im_w // 2 - width // 2
    center = Im[start_height:start_height + height, start_width:start_width + width]

    # Obtain the horizontally flipped versions
    im_list = [top_left, top_right, bottom_left, bottom_right, center]
    im_list_copy = [np.copy(im) for im in im_list]
    im_list_flipped = [np.flip(im, axis=1) for im in im_list_copy]
    images = im_list + im_list_flipped

    return images



def section_predict(model, section, patch_length):
    """Function evaluates the given segmentation model on the complete
    seismic section.

    Paramters
    ---------
    model: torch.nn.module object.
         The trained segmentation model to be applied. Model assumed to be on GPU

    section: torch.tensor of shape (depth, width)
         A 2D tensor representing the section to apply the model on. Also assumed to be GPU

    patch_length: int
                Length of patches model was trained on

    Returns
    -------
    segmented_section: ndarray
                     A 2D ndarray corresponding to the segmented seismic section input
                     to the function."""

    height, width = section.shape
    row_strides = np.ceil(height / patch_length).astype(
        int)  # Full column strides a kernel of patch_length can make in the section
    column_strides = np.ceil(width / patch_length).astype(
        int)  # Full row strides a kernel of patch_length can make in the section

    segmented_section = torch.randn(section.shape[0], section.shape[1]).cuda().float()

    for row_stride in range(row_strides):
        for column_stride in range(column_strides):

            if column_stride + 1 > width / patch_length:
                start_column_index = -patch_length
                end_column_index = None
            else:
                start_column_index = patch_length * column_stride
                end_column_index = patch_length * (column_stride + 1)

            if row_stride + 1 > height / patch_length:
                start_row_index = -patch_length
                end_row_index = None
            else:
                start_row_index = patch_length * row_stride
                end_row_index = patch_length * (row_stride + 1)

            section_patch = section[start_row_index:end_row_index, start_column_index:end_column_index]
            with torch.no_grad():
                segmented_patch = model(section_patch.unsqueeze(0).unsqueeze(0)).argmax(dim=1).squeeze()

            segmented_section[start_row_index:end_row_index, start_column_index:end_column_index] = segmented_patch

    return segmented_section


def compute_labels(data, model, patch_size, orientation='crossline'):
    """computed mIOU and mPA for segmentation accuracy

    Parameters
    ----------
    data: ndarray, 3D
        3D array of data of dimensions (Crossline x Inlines x Depth). The segmentation
        labels will be evaluated for each of the 2D Inlines x Depth section. It is the
        user's responsibility to input data in the orientaion segmentations are desired
        to be estimated.

    model: a PyTorch nn.modules object
        A trained segmentation model on cuda

    patch_size: int
        integer specifying patch size of for model estimations


    orientation: str, optional
               Specifes the orientation of the datacube. Assumes crossline by default.
               If inline, it transposes the first 2 axes of data.

    Returns
    -------
    label_volume: ndarray, 3D
                Volume of labels. Same shape as data
        """

    assert orientation in ['crossline', 'inline'], "Please specify the correct data cube orientation"

    # create a label volume of the same dimensions as the seismic data cube itself
    device = torch.device("cuda")
    label_volume = torch.zeros(data.shape, dtype=torch.float).to(device)

    num_sections = data.shape[0] if orientation == 'crossline' else data.shape[1]

    for section_num in range(num_sections):
        with torch.no_grad():
            if orientation == 'crossline':
                section = torch.from_numpy(
                    data[section_num].T).cuda().float()  # obtain section and convert to torch.float tensor on cuda
                label_volume[section_num] = section_predict(model, section, patch_size).T

            else:
                section = torch.from_numpy(data[:, section_num,
                                           :].T).cuda().float()  # obtain section and convert to torch.float tensor on cuda
                label_volume[:, section_num, :] = section_predict(model, section, patch_size).T

    return label_volume.detach().cpu().numpy()


def seismic_normalize(data, num_stds):
    '''normalizes seismic data volume between mean - num_std*std to mean + num_std*std before finally
    standardizing between 0 and 1'''

    mean = data.mean()
    std = data.std()

    data = np.clip(data, a_min=mean - num_stds * std, a_max=mean + num_stds * std)
    data = (data - data.min()) / (data.max() - data.min())

    return data


def create_3d_visualization(scalars, pos):
    """Function creates a 3-D visualization of the 3-D array scalars, displaying volumetric slices at the
    positions specified by the tuple pos

    Parameters
    ----------
    scalars: ndarray
           A 3-D numpy array containing the seismic volume to be visualized of shape (num_inlines, num_xlines, depth)

    pos: tuple of ints
       A tuple containing in this order, the x_slice, inline_slice, and z_slice positions to be displayed
       in the scalars volume
    """

    # check that slice positions are within range of the volume dimensions
    assert all([pos[i] < scalars.shape[i] for i in range(len(pos))])==True,"Please enter positions within volume dimensions!"

    fig = mlab.figure(figure='labels', bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    mlab.volume_slice(scalars, slice_index=0, plane_orientation='x_axes', figure=fig)
    mlab.volume_slice(scalars, slice_index=0, plane_orientation='y_axes', figure=fig)
    mlab.volume_slice(scalars, slice_index=0, plane_orientation='z_axes', figure=fig)
    mlab.outline()
    mlab.axes(xlabel='Inline', ylabel='Crossline', zlabel='Depth', nb_labels=10)
    mlab.show()

