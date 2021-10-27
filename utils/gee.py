import ee
import utm
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


def collect_sar_images(roi: ee.Geometry, orbit: int, polarization: str, start_date: str, end_date: str) -> ee.Image:

    s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(roi)\
        .filterDate(start_date, end_date)\
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


def generate_label(aoi_id: str, start_year: int, start_month: int, end_year: int, end_month: int):

    pass

if __name__ == '__main__':
    pass