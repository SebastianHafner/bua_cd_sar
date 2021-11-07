from pathlib import Path
import matplotlib.pyplot as plt
import change_detection as cd
from utils import paths, geofiles, sn7_helpers, visualization
FONTSIZE = 16


def qualitative_validation(aoi_id: str):

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    start_year, start_month = sn7_helpers.get_start_date(aoi_id)
    visualization.visualize_sn7_planet_mosaic(axs[0, 0], aoi_id, start_year, start_month)
    axs[0, 0].set_title(f'Planet {start_year}-{start_month:02d}')
    end_year, end_month = sn7_helpers.get_end_date(aoi_id)
    visualization.visualize_sn7_planet_mosaic(axs[0, 1], aoi_id, end_year, end_month)
    axs[0, 1].set_title(f'Planet mosaic {end_year}-{end_month:02d}')

    visualization.visualize_change_label(axs[0, 2], aoi_id)
    axs[0, 2].set_title('Ground Truth')

    visualization.visualize_change_map(axs[1, 0], aoi_id, 0)
    axs[1, 0].set_title('Ratio+ (CPT)')
    visualization.visualize_change_map(axs[1, 1], aoi_id, 1)
    axs[1, 1].set_title('Ratio- (CPT)')
    visualization.visualize_change_map(axs[1, 2], aoi_id, 2)
    axs[1, 2].set_title('KLD (CPT)')

    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    for aoi_id in sn7_helpers.get_aoi_ids():
        qualitative_validation(aoi_id)