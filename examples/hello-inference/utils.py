import numpy as np
import SimpleITK as sitk
from PIL import Image


def read_multiple_dicom(filenames):
    def sort_images_by_z_axis(filenames):

        images = []
        for fname in filenames:
            dicom_reader = sitk.ImageFileReader()
            dicom_reader.SetFileName(fname)
            dicom_reader.ReadImageInformation()

            images.append([dicom_reader, fname])

        zs = [float(dr.GetMetaData(key="0020|0032").split("\\")[-1]) for dr, _ in images]

        sort_inds = np.argsort(zs)[::-1]
        images = [images[s] for s in sort_inds]

        return images

    images = sort_images_by_z_axis(filenames)

    drs, fnames = zip(*images)
    fnames = list(fnames)

    simages = [sitk.GetArrayFromImage(dr.Execute()).squeeze() for dr in drs]
    volume = np.stack(simages)

    volume = np.expand_dims(volume, axis=0)

    return volume, fnames


def read_img(filename, channel_last=False, **kwargs):

    image = Image.open(filename)

    # pad channel dimension if it is gray image
    image = np.expand_dims(np.array(image), axis=0)
    if len(image.shape) == 4 and image.shape[-1] in [3, 4]:
        # for RGB+alpha
        image = image[0, :, :, :3]
        image = np.transpose(image, [2, 0, 1])

    if image.shape[0] == 1:
        image = np.repeat(image, 3, axis=0)

    if channel_last:
        image = np.transpose(image, [1, 2, 0])

    image = np.expand_dims(image, axis=0).astype(np.float32)

    return image
