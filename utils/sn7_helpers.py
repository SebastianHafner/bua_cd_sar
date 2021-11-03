from pathlib import Path
from utils import paths, geofiles


def get_sn7_timestamps() -> dict:
    dirs = paths.load_paths()
    sn7_timestamps = geofiles.load_json(Path(dirs.SN7_TIMESTAMPS_FILE))
    return sn7_timestamps


def get_timestamps(aoi_id: str) -> list:
    timestamps = get_sn7_timestamps()[aoi_id]
    clean_indices = [i for i, (_, _, mask) in enumerate(timestamps) if not mask]
    min_clean, max_clean = min(clean_indices), max(clean_indices)
    timestamps = timestamps[min_clean:max_clean + 1]
    return timestamps


def get_start_date(aoi_id: str) -> tuple:
    sn7_timestamps = get_timestamps(aoi_id)
    year, month, _ = sn7_timestamps[0]
    return year, month


def get_end_date(aoi_id: str) -> tuple:
    sn7_timestamps = get_timestamps(aoi_id)
    year, month, _ = sn7_timestamps[-1]
    return year, month


def get_aoi_ids() -> list:
    sn7_timestamps = get_sn7_timestamps()
    aoi_ids = sorted(sn7_timestamps.keys())
    return aoi_ids


def get_sn7_orbits() -> dict:
    dirs = paths.load_paths()
    sn7_orbits = geofiles.load_json(Path(dirs.SN7_ORBITS_FILE))
    return sn7_orbits


def get_orbit(aoi_id: str) -> int:
    orbit = get_sn7_orbits()[aoi_id]
    return orbit


def get_geo(aoi_id: str) -> tuple:
    dirs = paths.load_paths()
    year, month = get_start_date(aoi_id)
    file = Path(dirs.SN7_RAW) / 'train' / aoi_id / 'images' / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}.tif'
    arr, transform, crs, _ = geofiles.read_tif(file)
    m, n, _ = arr.shape
    x_res, _, x_min, _, y_res, y_max, *_ = transform

    west = x_min
    south = y_max + m * y_res
    east = x_min + n * x_res
    north = y_max
    return [west, south, east, north], crs


def get_building_footprints(aoi_id: str, year: int, month: int) -> dict:
    dirs = paths.load_paths()
    label_folder = Path(dirs.SN7_RAW) / 'train' / aoi_id / 'labels_match'
    label_file = label_folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_Buildings.geojson'
    label_data = geofiles.load_json(label_file)
    return label_data


def get_building_ids(aoi_id: str, year: int, month: int) -> list:
    building_footprints = get_building_footprints(aoi_id, year, month)
    ids = [f['properties']['Id'] for f in building_footprints['features']]
    return ids


def get_new_buildings(aoi_id: str, start_year: int, start_month: int, end_year: int, end_month: int):
    building_footprints = get_building_footprints(aoi_id, end_year, end_month)
    start_ids = get_building_ids(aoi_id, start_year, start_month)
    end_ids = get_building_ids(aoi_id, end_year, end_month)
    new_ids = [building_id for building_id in end_ids if building_id not in start_ids]
    new_buildings = [f for f in building_footprints['features'] if f['properties']['Id'] in new_ids]
    building_footprints['features'] = new_buildings
    return building_footprints


def load_sn7_planet_mosaic(aoi_id: str, year: int, month: int):
    dirs = paths.load_paths()
    images_folder = Path(dirs.SN7_RAW) / 'train' / aoi_id / 'images'
    file = images_folder / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}.tif'
    data, *_ = geofiles.read_tif(file)
    return data


if __name__ == '__main__':
    # get_building_footprints('L15-0331E-1257N_1327_3160_13', 2018, 1)
    get_new_buildings('L15-0331E-1257N_1327_3160_13', 2018, 1, 2020, 1)
