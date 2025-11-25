
import os

import rasterio
from rasterio.plot import show
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.io import MemoryFile

import ee

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

def get_satellite_image(bounds, start_date, end_date):
    region = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])

    image = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2") \
        .filterBounds(region) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUD_COVER', 20))

    bands = ['SR_B4', 'SR_B5', 'SR_B7']  # NIR, SWIR1, SWIR2
    rgb = image.median().select(bands).clip(region)
    url = rgb.getDownloadURL({
        'scale': 10,
        'region': region,
        'format': 'GEO_TIFF'
    })

    response = requests.get(url)
    memfile = MemoryFile(response.content)
    return memfile

def output_memfile(out_file, memfile: MemoryFile):
    with memfile.open() as src:
        profile = src.profile
        data = src.read()
        
        # Write to TIF
        with rasterio.open(out_file, 'w', **profile) as dst:
            dst.write(data)

def input_memfile(in_file) -> MemoryFile:
    with rasterio.open(in_file, 'r') as src:
        profile = src.profile
        data = src.read()
        
        memfile = MemoryFile()
        with memfile.open(**profile) as dst:
            dst.write(data)
        
        return memfile

def calculate_nbr(image: MemoryFile):
    """
    Calculate NBR (Normalized Burn Ratio) from a satellite image.
    NBR = (NIR - SWIR) / (NIR + SWIR)
    
    For Landsat 7:
    - Band 1 (SR_B4): NIR
    - Band 2 (SR_B5): SWIR1
    """
    with image.open() as src:
        nir     = src.read(1).astype(float) # NIR (band 1)
        swir    = src.read(2).astype(float) # SWIR (band 2)
        profile = src.profile # Store metadata for later use

    # Calculate NBR (Avoid division by zero)
    denominator = nir + swir
    nbr = np.where(denominator != 0, (nir - swir) / denominator, 0)
    
    return nbr, profile

def calculate_dnbr_og(before_file, after_file, output_path):
    """
    Calculate dNBR (differenced Normalized Burn Ratio).
    dNBR = NBR_prefire - NBR_postfire
    """
    if os.path.isfile(output_path):
        print("Already have dnbr file")
        return

    before_image = input_memfile(before_file)
    after_image = input_memfile(after_file)

    nbr_prefire,  profile = calculate_nbr(before_image)
    nbr_postfire, _       = calculate_nbr(after_image)

    dnbr = nbr_prefire - nbr_postfire
    
    profile.update(dtype=rasterio.float32, count=1, compress='lzw')
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(dnbr.astype(rasterio.float32), 1)
    
    print(f"dNBR range: {np.nanmin(dnbr):.4f} to {np.nanmax(dnbr):.4f}")

def calculate_dnbr(before_file, after_file, output_path):
    """
    Simplified version: Calculate dNBR and reproject to EPSG:5070.
    """
    if os.path.isfile(output_path):
        print("Already have dnbr file")
        return

    before_image = input_memfile(before_file)
    after_image = input_memfile(after_file)

    nbr_prefire, profile = calculate_nbr(before_image)
    nbr_postfire, _ = calculate_nbr(after_image)

    dnbr = nbr_prefire - nbr_postfire
    
    print(f"dNBR range: {np.nanmin(dnbr):.4f} to {np.nanmax(dnbr):.4f}")
    
    # Write temporary dNBR in original CRS
    temp_profile = profile.copy()
    temp_profile.update(dtype=rasterio.float32, count=1, compress='lzw')
    
    # Create in-memory dataset
    from rasterio.io import MemoryFile
    
    with MemoryFile() as memfile:
        with memfile.open(**temp_profile) as mem_dataset:
            mem_dataset.write(dnbr.astype(rasterio.float32), 1)
        
        # Reproject to EPSG:5070
        with memfile.open() as src:
            # Calculate transform for EPSG:5070
            transform, width, height = calculate_default_transform(
                src.crs, 'EPSG:5070',
                src.width, src.height,
                *src.bounds
            )
            
            # Update kwargs for destination
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': 'EPSG:5070',
                'transform': transform,
                'width': width,
                'height': height
            })
            
            # Write reprojected output
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs='EPSG:5070',
                    resampling=Resampling.bilinear
                )

def download_fire_images(before_file, after_file, fire_year, bounds):
    fire_date = f"{fire_year}-07-15"
    fire_dt = datetime.strptime(fire_date, "%Y-%m-%d")

    pre_fire_end = fire_dt - timedelta(days=30)
    pre_fire_start = pre_fire_end - timedelta(days=90)
    post_fire_start = fire_dt + timedelta(days=30)
    post_fire_end = post_fire_start + timedelta(days=150)

    if not os.path.isfile(before_file):
        before_image = get_satellite_image(bounds, pre_fire_start, pre_fire_end)
        output_memfile(before_file, before_image)
    else:
        print(f"before fire image already downloaded ({fire_year})")
        
    if not os.path.isfile(after_file):
        after_image = get_satellite_image(bounds, post_fire_start, post_fire_end)
        output_memfile(after_file, after_image)
    else:
        print(f"after fire image already downloaded ({fire_year})")
