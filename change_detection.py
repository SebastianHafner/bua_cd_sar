import os
import numpy as np
import pandas as pd
from scipy import ndimage
from joblib import Parallel, delayed

import ruptures as rpt
import calendar
from skimage import morphology
from sklearn.cluster import k_means
from skimage import measure
import seaborn as sns

from utils import paths, geofiles, visualization
from pathlib import Path
from tqdm import tqdm


# y: backsatter coefficient (dB); x: time
def change_detection_algorithm(y: np.ndarray, x: np.ndarray):
    # get rid of possible nan values and convert it into intensities
    idx = np.isfinite(y)
    y, x = y[idx], x[idx]

    # undo dB for backscatter coefficient
    y = 10 ** (y / 10)

    # change point detection
    # see: https://ctruong.perso.math.cnrs.fr/ruptures-docs/build/html/detection/bottomup.html
    algo = rpt.BottomUp(model="normal", min_size=round(len(y) / 10), jump=1).fit(y)  # Dynp, Binseg, BottomUp, Window
    points1 = np.array(algo.predict(n_bkps=1))[0]

    # Ratio
    ratio11 = np.mean(y[points1:]) / np.mean(y[0: points1])
    ratio12 = 1 / ratio11

    # Kullback-Liebler ddivergence
    mu1 = np.mean(np.log(y[0: points1]))
    std1 = np.std(np.log(y[0: points1]), ddof=1)
    mu2 = np.mean(np.log(y[points1:]))
    std2 = np.std(np.log(y[points1:]), ddof=1)
    kld1 = ((mu1 - mu2) ** 2 + std1 ** 2 - std2 ** 2) / std2 ** 2 + (
            (mu1 - mu2) ** 2 + std2 ** 2 - std1 ** 2) / std1 ** 2

    return [ratio11, ratio12, kld1, points1]


# returns change variables image with bands: ["Ratio+ (CPT)", "Ratio- (CPT)", "KLD (CPT)", "Point (CPT)"]
def extract_change_variables(images: np.ndarray, dates: list, parallelize: bool = True) -> np.ndarray:

    # prepare image
    m, n, n_images = images.shape
    n_pixels = m * n

    # extract images dates
    ref = pd.to_datetime("2015-01-01")
    dates = [pd.to_datetime(date_str) for date_str in dates]
    dates = pd.to_datetime(dates)
    t = np.array((dates - ref).days) / 365.25

    # compute change variables
    if parallelize:
        images = images.transpose((2, 0, 1))
        imagesT = np.reshape(images, (n_images, n_pixels)).transpose()
        change_variables = Parallel(n_jobs=8)(delayed(change_detection_algorithm)(y=imagesT[i, :], x=t) for i in range(n_pixels))
        change_variables = np.array(change_variables)  # (n_pixels, 4)
        change_variables = change_variables.transpose().reshape(4, m, n).transpose((1, 2, 0))
    else:
        change_variables = np.zeros((m, n, 4), dtype=np.single)
        for i in tqdm(range(n_pixels)):
            i = i // n
            j = i % n
            backscatter_timeseries = images[i, j, :]
            change_variables[i, j, :] = change_detection_algorithm(backscatter_timeseries, t)

    return change_variables


# produces change maps with kmean clustering
def change_mapping(change_variables: np.ndarray) -> np.ndarray:

    m, n, n_variables = change_variables.shape
    change_maps = []

    # Loop over chaneg variables to be classified
    for x in range(3):
        # Clean change variable
        var = change_variables[:, :, x].reshape(-1, 1)
        t1 = np.percentile(var, 2)
        t2 = np.percentile(var, 98)
        var[var < t1] = t1
        var[var > t2] = t2

        # Clustering using kmean
        clf = k_means(var, n_clusters=4, tol=0.0001, init="k-means++", n_jobs=4, algorithm="auto")
        change_map = clf[1] == np.argmax(clf[0])
        change_map = change_map.reshape((m, n))
        change_map = ndimage.median_filter(change_map == 1, size=7)
        change_map = morphology.remove_small_objects(change_map == 1, min_size=16, connectivity=2)
        change_maps.append(change_map)

    change_maps = np.stack(change_maps, axis=-1)

    return change_maps


def change_dating(change_variables: np.ndarray) -> np.ndarray:
    change_variables
    ChangeTime = np.zeros((0, Images.shape[1], Images.shape[2])).astype(int)
    for x in range(7):

        if x in [0, 1, 2]:
            Temp = ChangeImage[3, :, :] * (ChangeMap[x, :, :] == 1)
            Temp = measure.label(Temp > 0)
            Range = np.unique(Temp)[1:]
            CT = np.zeros((Temp.shape))
            for y in Range:
                idx = np.where(Temp == y)
                CT[idx] = np.median(ChangeImage[3, :, :][idx])
            ChangeTime = np.vstack((ChangeTime, CT[np.newaxis, ...]))

            # Writing change time to a file
    Drive = gdal.GetDriverByName("GTiff")
    Filen = File.replace("Data", "Results").replace(".tif", " - Change Time (Kmean)")
    OutData = Drive.Create(Filen, ChangeTime.shape[2], ChangeTime.shape[1], ChangeTime.shape[0], gdal.GDT_UInt16)
    OutData.SetProjection(Data.GetProjection())
    OutData.SetGeoTransform(Data.GetGeoTransform())
    for x in range(ChangeTime.shape[0]):
        rb = OutData.GetRasterBand(x + 1)
        rb.SetDescription(BandNames[x])
        OutData.GetRasterBand(x + 1).WriteArray(ChangeTime[x, :, :])
    OutData.FlushCache()
    OutData = None
    if os.path.exists(Filen + ".aux.xml"): os.remove(Filen + ".aux.xml")
    #   Adding metadata item to the heder file (file type : classification)
    Temp = (np.array(sns.color_palette(n_colors=Images.shape[0])) * 255).astype(int).flatten()
    color = "{0, 0, 0"
    for x in range(len(Temp)): color = color + ", " + str(Temp[x])
    color = color + "}"
    with open(Filen + '.hdr', "a") as myfile:
        myfile.write("classes = " + str(Images.shape[0] + 1) + "\n")
        myfile.write("class names = " + ClassNames + "\n")
        myfile.write("class lookup = " + color + "\n")
    pass


def change_aggregation() -> np.ndarray:

    step = 3
    ClassNames = "{Unchanged"
    CT = np.zeros((ChangeTime.shape)).astype(int)

    for y in [2017, 2018, 2019]:
        for m in np.arange(1, 12, step):

            idx1 = np.logical_and(np.logical_and(Dates.month >= m, Dates.month <= m + step - 1), Dates.year == y)
            idx1 = np.argwhere(idx1) + 1
            if len(idx1) == 0:
                continue
            Max = np.max(idx1)
            Min = np.min(idx1)
            idx2 = np.logical_and(ChangeTime >= Min, ChangeTime <= Max)
            if (~idx2).all():
                continue
            CT[idx2] = np.max(CT) + 1
            ClassNames = ClassNames + ", " + str(y) + ": " + calendar.month_abbr[m] + " - " + calendar.month_abbr[
                m + step - 1]
    ClassNames = ClassNames + "}"


def write_header_file(date_strings: list):
    class_names = "{Unchanged"
    print(images.shape)
    for date_str in date_strings:
        class_names = class_names + ", " + date_str
    class_names = class_names + "}"

    # change aggregation
    Temp = (np.array(sns.color_palette(n_colors=np.max(CT))) * 255).astype(int).flatten()
    color = "{0, 0, 0"
    for x in range(len(Temp)): color = color + ", " + str(Temp[x])
    color = color + "}"
    with open(Filen + '.hdr', "a") as myfile:
        myfile.write("classes = " + str(np.max(CT + 1)) + "\n")
        myfile.write("class names = " + ClassNames + "\n")
        myfile.write("class lookup = " + color + "\n")
    pass

    pass


if __name__ == '__main__':

    aoi_id = 'test'

    dirs = paths.load_paths()
    timeseries_file = Path(dirs.DATA) / f'{aoi_id}.tif'

    images, transform, crs, date_strings = geofiles.read_tif(timeseries_file)

    # change_variables = extract_change_variables(images, date_strings, parallelize=True)
    # visualization.visualize_change_variables(change_variables)

    cv_file = Path(dirs.OUTPUT) / 'change_variables' / f'change_variables_{aoi_id}.tif'
    # geofiles.write_tif(cv_file, change_variables, transform, crs)

    change_variables, *_ = geofiles.read_tif(cv_file)
    change_map = change_mapping(change_variables)




