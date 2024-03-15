import numpy as np
import SimpleITK as sitk


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
