# File defines the dataset classes used to load seismic patches

# import statements
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from utils import image_crop

class SeismicPatch(Dataset):
    """Dataset class for loading F3 patches and labels"""

    def __init__(self, seismic_cube, label_cube, orientation, crop_size, num_sections):
        """Initializer function for the dataset class

        Parameters
        ----------
        seismic_cube: ndarray, (crosslines, inline, depth)
                    3D ndarray of floats representing seismic amplitudes

        label_cube: ndarray, shape (crosslines, inline, depth)
                 3D ndarray same dimensions as seismic_cube containing label information.
                 Each value is [0,num_classes]

        orientation: {'inline', 'crossline},
                  a string denoting the orientation of the seismic cube

        crop_size: int
                 Integer specifying crop size in seismic section

        num_sections: int
                    Integer specifying the number of sections to train on
        """

        self.seismic = seismic_cube
        self.label = label_cube
        assert orientation in ['inline',
                               'xline'], "Invalid Orientation Type! Please choose either of 'inline' and 'xline'."
        self.orientation = orientation
        self.crop_size = crop_size

        if self.orientation == 'xline':
            assert 0 < num_sections and num_sections <= seismic_cube.shape[
                0], "Number of sections exceeds number of crosslines in seismic cube"
        else:
            assert 0 < num_sections and num_sections <= seismic_cube.shape[
                1], "Number of sections exceeds number of inlines in seismic cube"

        self.num_sections = num_sections

    def __getitem__(self, index):
        """Obtains the image crops relating to each section in the given orientation.

        Parameters
        ----------
        index: int
             Integer specifies the section number along the given orientation.

        Returns
        -------
        images: ndarray of shape (10, H, W)
              Returns an ndarray of 10 image crops in the section specified by index"""

        section = self.seismic[index].T if self.orientation == 'xline' else self.seismic[:, index, :].T
        label_section = self.label[index].T if self.orientation == 'xline' else self.label[:, index, :].T
        img_crops = image_crop(section, self.crop_size, self.crop_size)
        label_crops = image_crop(label_section, self.crop_size, self.crop_size)

        seismic_patches = np.zeros((len(img_crops), self.crop_size, self.crop_size))
        label_patches = np.zeros((len(img_crops), self.crop_size, self.crop_size))

        for i, (img, label) in enumerate(zip(img_crops, label_crops)):
            seismic_patches[i] = img
            label_patches[i] = label

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        seismic_patches = torch.from_numpy(seismic_patches).to(device).type(torch.float)
        label_patches = torch.from_numpy(label_patches).to(device).type(torch.long)

        return seismic_patches, label_patches

    def __len__(self):
        """Retrieves total number of training samples"""

        return self.num_sections



