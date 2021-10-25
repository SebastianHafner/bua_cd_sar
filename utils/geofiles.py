import rasterio
from pathlib import Path


# reading in geotiff file as numpy array
def read_tif(file: Path):
    if not file.exists():
        raise FileNotFoundError(f'File {file} not found')

    with rasterio.open(file) as dataset:
        arr = dataset.read()  # (bands X height X width)
        transform = dataset.transform
        crs = dataset.crs
        descriptions = dataset.descriptions

    return arr.transpose((1, 2, 0)), transform, crs, list(descriptions)


# writing an array to a geo tiff file
def write_tif(file: Path, arr, transform, crs):
    if not file.parent.exists():
        file.parent.mkdir()

    if len(arr.shape) == 3:
        height, width, bands = arr.shape
    else:
        height, width = arr.shape
        bands = 1
        arr = arr[:, :, None]
    with rasterio.open(
            file,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=bands,
            dtype=arr.dtype,
            crs=crs,
            transform=transform,
    ) as dst:
        for i in range(bands):
            dst.write(arr[:, :, i], i + 1)