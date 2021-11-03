import ee
import utm
import numpy as np
from utils import sn7_helpers

ee.Initialize()


def bounding_box(coords: list, crs: str):
    return ee.Geometry.Rectangle(coords, proj=str(crs)).transform('EPSG:4326')


def epsg_utm(bbox: ee.Geometry):
    center_point = bbox.centroid()
    coords = center_point.getInfo()['coordinates']
    lon, lat = coords
    easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
    return f'EPSG:326{zone_number}' if lat > 0 else f'EPSG:327{zone_number}'


def new_buildings(aoi_id: str, start_year: int, start_month: int, end_year: int, end_month: int) -> ee.FeatureCollection:
    building_footprints = sn7_helpers.get_new_buildings(aoi_id, start_year, start_month, end_year, end_month)
    features = building_footprints['features']
    new_features = []
    for feature in features:
        coords = feature['geometry']['coordinates']
        geom = ee.Geometry.Polygon(coords, proj='EPSG:3857').transform('EPSG:4326')
        new_feature = ee.Feature(geom)
        new_features.append(new_feature)
    return ee.FeatureCollection(new_features)


def filterFullyOverlap(image_collection: ee.ImageCollection, geom: ee.Geometry.Polygon) -> ee.ImageCollection:
    coordsList = ee.List(geom.coordinates().get(0))
    for i in range(4):
        coords = coordsList.get(i)
        point = ee.Geometry.Point(coords, proj=geom.projection())
        image_collection = image_collection.filterBounds(point)
    return image_collection


def collect_sar_images(roi: ee.Geometry, polarization: str, start_date: str, end_date: str,
                       orbit: int = None) -> ee.Image:

    s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterDate(start_date, end_date)\
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', polarization))\
        .filterBounds(roi)

    s1 = filterFullyOverlap(s1, roi)

    if orbit is not None:
        s1 = s1.filter(ee.Filter.eq('relativeOrbitNumber_stop', orbit))
    else:
        # select orbit with best data availability
        orbits = s1.toList(s1.size()).map(lambda img: ee.Image(img).get('relativeOrbitNumber_stop'))
        orbits = orbits.distinct().getInfo()
        scenes_per_orbit = []
        for orbit_cand in orbits:
            n_scenes = s1.filter(ee.Filter.eq('relativeOrbitNumber_stop', orbit_cand)).size().getInfo()
            scenes_per_orbit.append(n_scenes)
        i_max = np.argmax(scenes_per_orbit)
        orbit = orbits[i_max]
        s1 = s1.filter(ee.Filter.eq('relativeOrbitNumber_stop', orbit))

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


def generate_label(aoi_id: str, start_year: int, start_month: int, end_year: int, end_month: int):

    pass

if __name__ == '__main__':
    pass