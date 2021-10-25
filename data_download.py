# Import the Earth Engine API and initialize it.
import ee

ee.Initialize()


def create_bbox(east: float, south: float, west: float, north: float) -> ee.Geometry:
    pass


def collect_data(coords: list, orbit: int, polarization: str, start_date: str, end_date: str) -> ee.Image:

    patch_size = 2
    index1 = 0 / patch_size
    index2 = 0 / patch_size
    PatchROI = [1, 2, 3, 4]
    PatchROI[0] = coords[0] * (1 - index1) + coords[2] * index1
    PatchROI[2] = coords[0] * (1 - 1 / patch_size - index1) + coords[2] * (index1 + 1 / patch_size)
    PatchROI[1] = coords[1] * (1 - index2) + coords[3] * index2
    PatchROI[3] = coords[1] * (1 - 1 / patch_size - index2) + coords[3] * (index2 + 1 / patch_size)

    OutputROI = ee.Geometry.Rectangle(PatchROI)
    coordList = ee.List(OutputROI.coordinates().get(0))
    p0 = ee.Geometry.Point(coordList.get(0))
    p1 = ee.Geometry.Point(coordList.get(1))
    p2 = ee.Geometry.Point(coordList.get(2))
    p3 = ee.Geometry.Point(coordList.get(3))

    s1 = ee.ImageCollection('COPERNICUS/S1_GRD')\
        .filterBounds(p0).filterBounds(p1).filterBounds(p2).filterBounds(p3)\
        .filterDate(start_date, end_date)\
        .filter(ee.Filter.eq('relativeOrbitNumber_stop', orbit))\
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', polarization))

    s1 = s1.sort('SLC_Processing_start', True)
    img = s1.first().select([polarization])
    img = img.rename(img.date().format().slice(0, 10))

    s1 = s1.select([polarization]).toList(200)
    n_scenes = s1.length().getInfo()
    print(n_scenes)

    for i in range(n_scenes - 1):
        img2 = ee.Image(s1.get(i + 1))
        img = img.addBands(img2.rename(img2.date().format().slice(0, 10)))

    return img


def download_data(metadata: dict):

    aoi_id = metadata['aoi_id']
    del metadata['aoi_id']

    img = collect_data(**metadata)

    XX = img.bandNames()
    print(XX.getInfo())

    imageTask = ee.batch.Export.image.toDrive(
        image=img,
        description=aoi_id,
        folder="bua_cd_sar",
        scale=10,
        maxPixels=1e12,
        region=create_bbox(*metadata['coords'])
    )

    import time

    # Start the task.
    imageTask.start()

    while imageTask.active():
        print('Polling for task (id: {}).'.format(imageTask.id))
        time.sleep(10)
    print('Done with image export.')


if __name__ == '__main__':

    test = {
        'aoi_id': 'test',
        'coords': [-116.45507813235129, 43.644026768116866, -116.41113370284677, 43.67582227697864],
        'start_date': '2018-01-01',
        'end_date': '2021-01-31',
        'orbit': 71,
        'polarization': 'VV',
    }

    download_data(test)


