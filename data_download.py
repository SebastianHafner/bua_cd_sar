# Import the Earth Engine API and initialize it.
import ee
from utils import gee

ee.Initialize()


def download_data(metadata: dict):

    aoi_id = metadata['aoi_id']
    del metadata['aoi_id']

    img = gee.collect_data(**metadata)

    XX = img.bandNames()
    print(XX.getInfo())

    imageTask = ee.batch.Export.image.toDrive(
        image=img,
        description=aoi_id,
        folder="bua_cd_sar",
        scale=10,
        maxPixels=1e12,
        region=gee.create_bbox(*metadata['coords'])
    )

    import time

    # Start the task.
    imageTask.start()

    while imageTask.active():
        print('Polling for task (id: {}).'.format(imageTask.id))
        time.sleep(10)
    print('Done with image export.')


if __name__ == '__main__':
    # coords: [west, south, east, north]

    test = {
        'aoi_id': 'test',
        'coords': [-116.45507813235129, 43.644026768116866, -116.41113370284677, 43.67582227697864],
        'start_date': '2018-01-01',
        'end_date': '2021-01-31',
        'orbit': 71,
        'polarization': 'VV',
    }

    stockholm_test = {
        'aoi_id': 'stockholm_test',
        'coords': [17.92781, 59.36769, 17.95472, 59.38975],
        'start_date': '2017-01-01',
        'end_date': '2019-12-31',
        'orbit': 22,
        'polarization': 'VV',
    }

    download_data(stockholm_test)


