import matplotlib.pyplot as plt
from utils import paths, visualization, sn7_helpers


def visualize_label(aoi_id: str):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    start_year, start_month = sn7_helpers.get_start_date(aoi_id)
    end_year, end_month = sn7_helpers.get_end_date(aoi_id)
    visualization.visualize_sn7_planet_mosaic(axs[0], aoi_id, start_year, start_month)
    visualization.visualize_sn7_planet_mosaic(axs[1], aoi_id, end_year, end_month)
    visualization.visualize_change_label(axs[2], aoi_id)
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    for aoi_id in sn7_helpers.get_aoi_ids():
        visualize_label(aoi_id)