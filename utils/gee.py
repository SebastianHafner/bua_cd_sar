import ee
ee.Initialize()


def create_bbox(west: float, south: float, east: float, north: float) -> ee.Geometry:
    return ee.Geometry.BBox(west, south, east, north)


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

    s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(p0).filterBounds(p1).filterBounds(p2).filterBounds(p3) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.eq('relativeOrbitNumber_stop', orbit)) \
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

if __name__ == '__main__':
    pass