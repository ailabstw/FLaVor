import numpy as np
import SimpleITK as sitk


def read_multiple_dicom(filenames):
    def sort_images_by_z_axis(filenames):

        images = []
        for fname in filenames:
            dicom_reader = sitk.ImageFileReader()
            dicom_reader.SetFileName(fname)
            dicom_reader.ReadImageInformation()

            images.append(dicom_reader)

        zs = [float(dr.GetMetaData(key="0020|0032").split("\\")[-1]) for dr in images]

        sort_inds = np.argsort(zs)[::-1]
        images = [images[s] for s in sort_inds]

        return images

    drs = sort_images_by_z_axis(filenames)

    simages = [sitk.GetArrayFromImage(dr.Execute()).squeeze() for dr in drs]
    volume = np.stack(simages)

    volume = np.expand_dims(volume, axis=0)

    return {"data": volume}
