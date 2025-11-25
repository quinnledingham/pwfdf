from data import *
from dnbr import *
from watershed import *

def main():
    data = PWFDF_Data()
    i = 2
    entry = data.get(i)

    print(f"Using entry: {i}, fire name: {entry['Fire Name']}, seg id: {entry['Fire_SegID']}")
    bounds = entry.bounds(5)

    data_dir = f"./data/{entry['Fire_SegID']}/"
    segid = entry['Fire_SegID']

    dem_file           = os.path.join(data_dir, f"dem_{segid}.tif")
    before_file        = os.path.join(data_dir, f"before_{segid}.tif")
    after_file         = os.path.join(data_dir, f"after_{segid}.tif")
    dnbr_file          = os.path.join(data_dir, f"dnbr_{segid}.tif")
    clipped_dnbr_file  = os.path.join(data_dir, f"clipped_dnbr_{segid}.tif")

    if True:
    #if not os.path.isdir(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        download_watershed(dem_file, bounds)
        download_fire_images(before_file, after_file, entry['Year'], bounds)
        calculate_dnbr(before_file, after_file, dnbr_file)
        catch, grid = delineate_watershed(dem_file, entry.coordinates_5070())
        clip_dnbr_by_watershed(dnbr_file, catch, grid, clipped_dnbr_file)

    #catch, grid = delineate_watershed(dem_file, entry.coordinates_wgs84())


if __name__ == '__main__':
    main()
