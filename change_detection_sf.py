import numpy as np
from utils import paths, geofiles, sn7_helpers
import scipy
from pathlib import Path
from tqdm import tqdm


def compute_kl_div(images: np.ndarray, break_points: np.ndarray) -> np.ndarray:
    # Kullback-Liebler ddivergence
    mu1=np.mean(np.log(y[0: points1]))
    std1 = np.std(np.log(y[0: points1]), ddof=1)


    mu2 = np.mean(np.log(y[points1:]))
    std2 = np.std(np.log(y[points1:]), ddof=1)
    kld1 = ((mu1 - mu2) ** 2 + std1 ** 2 - std2 ** 2) / std2 ** 2 + (
            (mu1 - mu2) ** 2 + std2 ** 2 - std1 ** 2) / std1 ** 2

    return kld1


def compute_mse(y: np.ndarray, y_hat: np.ndarray) -> float:
    return np.sum(np.square(y_hat - y), axis=-1) / y.shape[-1]


def change_detection(images: np.ndarray, min_diff: int = 5):
    length_ts = images.shape[-1]

    # images = scipy.signal.convolve2d(images, in2, mode='same', boundary='symm')

    errors = []
    mean_diffs = []

    # break point detection
    for i in range(1, length_ts):
        # compute predicted
        presegment = images[:, :, :i]
        mean_presegment = np.mean(presegment, axis=-1)
        pred_presegment = np.repeat(mean_presegment[:, :, np.newaxis], i, axis=-1)

        postsegment = probs_cube[:, :, i:]
        mean_postsegment = np.mean(postsegment, axis=-1)
        pred_postsegment = np.repeat(mean_postsegment[:, :, np.newaxis], length_ts - i, axis=-1)

        # maybe use absolute value here
        mean_diffs.append(mean_postsegment - mean_presegment)

        pred_break = np.concatenate((pred_presegment, pred_postsegment), axis=-1)
        mse_break = compute_mse(images, pred_break)
        errors.append(mse_break)

    errors = np.stack(errors, axis=-1)
    best_fit = np.argmin(errors, axis=-1)

    min_error = np.min(errors, axis=-1)
    mean_diffs = np.stack(mean_diffs, axis=-1)
    m, n = mean_diffs.shape[:2]
    mean_diff = mean_diffs[np.arange(m)[:, None], np.arange(n), best_fit]
    change = mean_diff > min_diff


if __name__ == '__main__':
    dirs = paths.load_paths()
    for aoi_id in sn7_helpers.get_aoi_ids():
        print(aoi_id)
        sar_images_file = Path(dirs.DATA) / 'images' / f'{aoi_id}.tif'
        sar_images, transform, crs, date_strings = geofiles.read_tif(sar_data_file)
        change = change_detection(sar_images)
