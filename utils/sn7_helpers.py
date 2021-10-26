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


def get_coords(aoi_id: str) -> list:
    dirs = paths.load_paths()
    year, month = get_start_date(aoi_id)
    file = Path(dirs.SN7_RAW) / 'train' / aoi_id / 'images' / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}.tif'
    arr, transform, *_ = geofiles.read_tif(file)
    m, n, _ = arr.shape
    x_res, _, x_min, _, y_res, y_max, *_ = transform

    west = x_min
    south = y_max + m * y_res
    east = x_min + n * x_res
    north = y_max
    return [west, south, east, north]


def get_crs(aoi_id: str) -> str:
    dirs = paths.load_paths()
    year, month = get_start_date(aoi_id)
    file = Path(dirs.SN7_RAW) / 'train' / aoi_id / 'images' / f'global_monthly_{year}_{month:02d}_mosaic_{aoi_id}.tif'
    _, _, crs, _ = geofiles.read_tif(file)
    return crs
