# Import the Earth Engine API and initialize it.
import ee
from utils import gee, sn7_helpers

ee.Initialize()


def download_sn7_images(aoi_id: str):

    start_year, start_month = sn7_helpers.get_start_date(aoi_id)
    end_year, end_month = sn7_helpers.get_end_date(aoi_id)

    orbit = sn7_helpers.get_orbit(aoi_id)

    coords = sn7_helpers.get_coords(aoi_id)
    crs = sn7_helpers.get_crs(aoi_id)

    metadata = {
        'aoi_id': aoi_id,
        'coords': coords,
        'crs': crs,
        'start_date': f'{start_year}-{start_month:2d}-01',
        'end_date': f'{end_year}-{end_month:2d}-01',
        'orbit': orbit,
        'polarization': 'VV',
    }

    img = gee.collect_data(**metadata)

    XX = img.bandNames()
    print(XX.getInfo())

    imageTask = ee.batch.Export.image.toDrive(
        image=img,
        description=aoi_id,
        folder="bua_cd_sar",
        scale=10,
        maxPixels=1e12,
        crs=crs,
        region=gee.create_bbox(*metadata['coords'])
    )

    import time

    # Start the task.
    imageTask.start()

    while imageTask.active():
        print('Polling for task (id: {}).'.format(imageTask.id))
        time.sleep(10)
    print('Done with image export.')


# TODO: implement
def downnload_sn7_labels(aoi_id: str):
    pass


if __name__ == '__main__':
    # coords: [west, south, east, north]
    for i, aoi_id in enumerate(sn7_helpers.get_aoi_ids()):
        download_sn7_images(aoi_id)
        if i > 0:
            break
