import os
import numpy as np
import pandas as pd
from scipy import ndimage
from joblib import Parallel, delayed

import ruptures as rpt
import calendar
from skimage import morphology
from sklearn import cluster
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
        # change shape from (n_pixels, 4) to (m, n, 4) where 4 is n change variables
        change_variables = np.array(change_variables).transpose().reshape(4, m, n).transpose((1, 2, 0))
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

        # clustering using kmean: highest cluster for change variable is change
        centers, labels, _ = cluster.k_means(var, n_clusters=4, tol=0.0001, init="k-means++", algorithm="auto")
        change_map = labels == np.argmax(centers)
        change_map = change_map.reshape((m, n))
        change_map = ndimage.median_filter(change_map == 1, size=7)
        change_map = morphology.remove_small_objects(change_map == 1, min_size=16, connectivity=2)
        change_maps.append(change_map)

    change_maps = np.stack(change_maps, axis=-1)

    return change_maps.astype(np.int8)


def change_dating(change_variables: np.ndarray, change_maps: np.ndarray) -> np.ndarray:
    m, n , n_variables = change_variables.shape
    break_point = change_variables[:, :, 3]
    change_times = []
    for x in range(3):
        is_change = change_maps[:, :, x] == 1
        temp = is_change * break_point

        # find connected regions
        # https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label
        temp = measure.label(temp > 0)

        time_range = np.unique(temp)[1:]
        change_time = np.zeros((m, n))
        for y in time_range:
            idx = np.where(temp == y)
            change_time[idx] = np.median(break_point[idx])
        change_times.append(change_time)
    change_times = np.stack(change_times)
    return change_times


# temporally aggregate changes
def change_aggregation(change_time: np.ndarray, dates: list, months: int = 3) -> np.ndarray:

    change_times_agg = []

    # TODO: get years from dates
    for y in [2017, 2018, 2019]:
        for m in np.arange(1, 12, months):
            idx1 = np.logical_and(np.logical_and(dates.month >= m, dates.month <= m + months - 1), dates.year == y)
            idx1 = np.argwhere(idx1) + 1
            if len(idx1) == 0:
                continue
            Max = np.max(idx1)
            Min = np.min(idx1)
            idx2 = np.logical_and(change_time >= Min, change_time <= Max)
            if (~idx2).all():
                continue
            change_time[idx2] = np.max(change_time) + 1

    return change_times_agg


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

    Temp = (np.array(sns.color_palette(n_colors=Images.shape[0])) * 255).astype(int).flatten()
    color = "{0, 0, 0"
    for x in range(len(Temp)): color = color + ", " + str(Temp[x])
    color = color + "}"
    with open(Filen + '.hdr', "a") as myfile:
        myfile.write("classes = " + str(Images.shape[0] + 1) + "\n")
        myfile.write("class names = " + ClassNames + "\n")
        myfile.write("class lookup = " + color + "\n")
    pass
    class_names = "{Unchanged"
    ClassNames = ClassNames + ", " + str(y) + ": " + calendar.month_abbr[m] + " - " + calendar.month_abbr[
        m + step - 1]


    ClassNames = ClassNames + "}"

if __name__ == '__main__':

    aoi_id = 'stockholm_test'

    dirs = paths.load_paths()
    timeseries_file = Path(dirs.DATA) / f'{aoi_id}.tif'

    images, transform, crs, date_strings = geofiles.read_tif(timeseries_file)

    # change_variables = extract_change_variables(images, date_strings, parallelize=True)
    # visualization.visualize_change_variables(change_variables)

    cv_file = Path(dirs.OUTPUT) / 'change_variables' / f'change_variables_{aoi_id}.tif'
    # geofiles.write_tif(cv_file, change_variables, transform, crs)

    change_variables, *_ = geofiles.read_tif(cv_file)
    change_maps = change_mapping(change_variables)

    # cm_file = Path(dirs.OUTPUT) / 'change_maps' / f'change_maps_{aoi_id}.tif'
    # geofiles.write_tif(cm_file, change_maps, transform, crs)
    change_times = change_dating(change_variables, change_maps)

    ct_file = Path(dirs.OUTPUT) / 'change_times' / f'change_times_{aoi_id}.tif'
    geofiles.write_tif(ct_file, change_times, transform, crs)



