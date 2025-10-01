import matplotlib.pyplot as plt
from pysheds.grid import Grid

import requests

def plot_catchment(grid, clipped_catch):
    fig, ax = plt.subplots(figsize=(8,6))
    fig.patch.set_alpha(0)
    plt.grid('on', zorder=0)
    im = ax.imshow(np.where(clipped_catch, clipped_catch, np.nan), extent=grid.extent, zorder=1, cmap='Greys_r')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Delineated Catchment', size=14)

def plot_catchment_dist(grid, dist):
    fig, ax = plt.subplots(figsize=(8,6))
    fig.patch.set_alpha(0)
    plt.grid('on', zorder=0)
    im = ax.imshow(dist, extent=grid.extent, zorder=2, cmap='cubehelix_r')
    plt.colorbar(im, ax=ax, label='Distance to outlet (cells)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Flow Distance', size=14)

def delineate_watershed(dem_file, pour_point_coords):
    grid = Grid.from_raster(dem_file)
    dem = grid.read_raster(dem_file)

    pit_filled_dem = grid.fill_pits(dem)
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    inflated_dem = grid.resolve_flats(flooded_dem)

    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    flow_dir  = grid.flowdir(inflated_dem, dirmap=dirmap)
    flow_acc = grid.accumulation(flow_dir, dirmap=dirmap)

    threshold = 100
    branches = grid.extract_river_network(flow_dir, flow_acc > threshold, dirmap=dirmap)

    x, y = pour_point_coords
    x_snap, y_snap = grid.snap_to_mask(flow_acc > 10, pour_point_coords) # Snap pour point to high accumulation cell
    catch = grid.catchment(x=x_snap, y=y_snap, fdir=flow_dir, dirmap=dirmap, xytype='coordinate')

    dist = grid.distance_to_outlet(x=x_snap, y=y_snap, fdir=flow_dir, dirmap=dirmap, xytype='coordinate')

    # plot DEM
    print("PLOTTING DEM")
    fig, ax = plt.subplots(figsize=(8,8))
    fig.patch.set_alpha(0)
    plt.imshow(dem, extent=grid.extent, cmap='terrain', zorder=1)
    plt.colorbar(label='Elevation (m)')
    plt.grid(zorder=0)
    #im = ax.imshow(np.where(clipped_catch, clipped_catch, np.nan), extent=grid.extent, zorder=1, cmap='Greys_r')
    im = ax.imshow(dist, extent=grid.extent, zorder=2, cmap='cubehelix_r')
    plt.title('Digital elevation map', size=14)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    #print(f"x: {x}, y: {y}")
    plt.plot(x, y, 'ro', markersize=1, zorder=2)  # 'ro' = red circle
    plt.plot(x_snap, y_snap, 'go', markersize=1, zorder=2)  # 'ro' = red circle
    plt.tight_layout()
    plt.show()

def get_watershed_from_usgs_api(longitude: float, latitude: float):
    try:
        # USGS StreamStats API endpoint
        url_format = "https://test.streamstats.usgs.gov/streamstatsservices/watershed{0}?rcode={1}&xlocation={2}&ylocation={3}&crs={4}&includeparameters={5}&includeflowtypes={6}&includefeatures={7}&simplify={8}"
        api_url = url_format.format(
            '.geojson',          # {0} - format extension
            "CA",                # {1} - region code
            longitude,           # {2} - x location
            latitude,            # {3} - y location
            "4326",              # {4} - coordinate reference system
            str("true").lower(), # {5} - include parameters
            str("false").lower(),# {6} - include flow types
            str("true").lower(), # {7} - include features
            str("true").lower()  # {8} - simplify geometry
        )
        response = requests.get(api_url, timeout=120)
        response.raise_for_status()

        data = response.json()

        if 'featurecollection' in data and len(data['featurecollection']) > 0:
            watershed_features = data['featurecollection'][0]
            return watershed_features
        else:
            print("No watershed found for the given coordinates")
            return None

    except Exception as e:
        print(f"Error querying USGS API: {e}")
        return None

def plot_dem(output_file, x, y):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_alpha(0)

    with rasterio.open(output_file) as src:
        target_crs = src.crs
        target_bounds = src.bounds
        target_shape = src.shape
        target_transform = src.transform
        show(src, ax=ax, cmap='terrain', zorder=1)

    plot_satellite_image(ax, '/home/quinn/pwfdf/data/satellite.tif', target_crs, target_bounds, target_shape, target_transform)

    plt.colorbar(ax.images[0], label='Elevation (m)')

    ax.plot(x, y, 'ro', markersize=10, zorder=3)
    plt.grid(zorder=0)
    plt.title('DEM with Satellite Overlay', size=14)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.show()