import rasterio
from rasterio.plot import show
from rasterio.warp import reproject, Resampling, calculate_default_transform

import ee

ee.Authenticate()
ee.Initialize(project='ee-quinnledingham')

import requests
from datetime import datetime, timedelta
import numpy as np

def plot_satellite_image(ax, file, target_crs, target_bounds, target_shape, target_transform):
    with rasterio.open(file) as src:
        raster = src.read([1, 2, 3])

        print(f"\nImage metadata:")
        print(f"  Shape: {raster.shape}")
        print(f"  CRS: {src.crs}")
        print(f"  Bounds: {src.bounds}")

        reprojected = np.zeros((3, target_shape[0], target_shape[1]), dtype=raster.dtype)
        for i in range(3):
            reproject(
                source=raster[i],
                destination=reprojected[i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear
            )

        raster_plot = np.transpose(reprojected, (1, 2, 0))
        raster_plot = np.clip(raster_plot / 3000, 0, 1)
        extent = [target_bounds.left, target_bounds.right, target_bounds.bottom, target_bounds.top]
        ax.imshow(raster_plot, extent=extent, zorder=2, alpha=0.5)

def get_satellite_image(file, bounds, start_date, end_date):
    region = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])

    image = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2") \
        .filterBounds(region) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUD_COVER', 20))

    bands = ['SR_B4, SR_B5, SR_B7']  # NIR, SWIR1, SWIR2
    rgb = image.median().select(bands).clip(region)
    url = rgb.getDownloadURL({
        'scale': 10,
        'region': region,
        'format': 'GEO_TIFF'
    })

    response = requests.get(url)
    with open(file, 'wb') as f:
        f.write(response.content)


def get_fire_images(fire_year, bounds):
    print("FIRE")
    fire_date = f"{fire_year}-07-15"
    fire_dt = datetime.strptime(fire_date, "%Y-%m-%d")

    pre_fire_end = fire_dt - timedelta(days=30)
    pre_fire_start = pre_fire_end - timedelta(days=90)
    post_fire_start = fire_dt + timedelta(days=30)
    post_fire_end = post_fire_start + timedelta(days=150)

    get_satellite_image('/home/quinn/pwfdf/data/before.tif', bounds, pre_fire_start, pre_fire_end)
    get_satellite_image('/home/quinn/pwfdf/data/after.tif', bounds, post_fire_start, post_fire_end)