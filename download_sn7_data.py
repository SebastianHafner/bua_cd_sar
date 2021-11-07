import ee
from utils import gee, sn7_helpers

ee.Initialize()


def download_sn7_images(aoi_id: str):

    start_year, start_month = sn7_helpers.get_start_date(aoi_id)
    end_year, end_month = sn7_helpers.get_end_date(aoi_id)

    orbit = sn7_helpers.get_orbit(aoi_id)

    coords, crs = sn7_helpers.get_geo(aoi_id)
    roi = gee.bounding_box(coords, crs)
    crs = gee.epsg_utm(roi)
    roi = roi.transform(crs, 0.001)

    metadata = {
        'roi': roi,
        'start_date': f'{start_year}-{start_month:02d}-01',
        'end_date': f'{end_year}-{end_month:02d}-01',
        'orbit': None,
        'polarization': 'VV',
    }

    img = gee.collect_sar_images(**metadata)

    XX = img.bandNames()
    print(XX.getInfo())

    imageTask = ee.batch.Export.image.toDrive(
        image=img,
        description=aoi_id,
        folder="bua_cd_sar_images",
        scale=10,
        maxPixels=1e12,
        crs=crs,
        region=roi,
    )

    # Start the task.
    imageTask.start()


def downnload_sn7_labels(aoi_id: str):

    start_year, start_month = sn7_helpers.get_start_date(aoi_id)
    end_year, end_month = sn7_helpers.get_end_date(aoi_id)

    coords, crs = sn7_helpers.get_geo(aoi_id)
    roi = gee.bounding_box(coords, crs)
    crs = gee.epsg_utm(roi)
    roi = roi.transform(crs, 0.001)

    fc = gee.new_buildings(aoi_id, start_year, start_month, end_year, end_month)
    fc = fc.filterBounds(roi)
    print(fc.size().getInfo())

    fc = fc.map(lambda f: ee.Feature(f).set({'new_bua': 1}))

    bua_new = fc.reduceToImage(['new_bua'], ee.Reducer.first()).unmask().float().rename('new_bua')

    bua_new = bua_new \
        .reproject(crs=crs, scale=1) \
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1000) \
        .reproject(crs=crs, scale=10) \
        .rename('new_bua_percentage')

    img_name = f'label_{aoi_id}'
    dl_desc = f'{aoi_id}LabelDownload'

    imageTask = ee.batch.Export.image.toDrive(
        image=bua_new,
        fileNamePrefix=img_name,
        description=dl_desc,
        folder="bua_cd_sar_labels",
        scale=10,
        maxPixels=1e12,
        crs=crs,
        region=roi,
    )

    # Start the task.
    imageTask.start()


if __name__ == '__main__':
    for i, aoi_id in enumerate(sn7_helpers.get_aoi_ids()):
        print(i, aoi_id)
        # download_sn7_images(aoi_id)
        downnload_sn7_labels(aoi_id)
