from utils import sn7_helpers, paths, geofiles
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_sar_image(ax, images: np.ndarray, i: int):
    ax.imshow(images[i, :, :], cmap='gray')
    ax.set_axis_off()


def visualize_change_variables(variables: np.ndarray):
    variable_names = ["Ratio+ (CPT)", "Ratio- (CPT)", "KLD (CPT)", "Point (CPT)"]
    fig, axs = plt.subplots(1, 4)
    fig.tight_layout()
    for i in range(4):
        axs[i].imshow(variables[:, :, i])
        axs[i].set_title(variable_names[i])
        axs[i].set_axis_off()
    plt.show()
    plt.close(fig)


def visualize_sn7_planet_mosaic(ax, aoi_id: str, year: int, month: int):
    data = sn7_helpers.load_sn7_planet_mosaic(aoi_id, year, month)
    ax.imshow(data)
    ax.set_axis_off()


def visualize_change_label(ax, aoi_id: str):
    label = sn7_helpers.load_change_label(aoi_id)
    ax.imshow(label, cmap='gray')
    ax.set_axis_off()


def visualize_change_map(ax, aoi_id: str, index: int):
    dirs = paths.load_paths()
    file = Path(dirs.OUTPUT) / 'results' / f'{aoi_id}_change_maps.tif'
    change_maps, *_ = geofiles.read_tif(file)
    ax.imshow(change_maps[:, :, index], cmap='gray')
    ax.set_axis_off()


def visualize_change(ax, change_map: np.ndarray):
    ax.imshow(change_map, cmap='gray')
    ax.set_axis_off()
