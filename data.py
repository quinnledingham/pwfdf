import os
import sciencebasepy
from pathlib import Path

import pandas as pd
from pyproj import Proj, transform, Transformer

from dnbr import *

# Downloads all of the hazard assessments to dir
def download_pwfdf_collection():
    dir = 'data/collection'
    sb = sciencebasepy.SbSession()
    id = '6818f950d4be0208bc3e0165' #Post-Wildfire Debris-Flow Hazard Assessment (PWFDF) Collection
    item = sb.get_item(id)

    print(f"Title: {item.get('title', 'No title')}")
    print(f"Summary: {item.get('summary', 'No summary')}")
    print(f"Item URL: https://www.sciencebase.gov/catalog/item/{id}")

    child_ids = sb.get_child_ids(id)

    path = Path(dir)
    path.mkdir(exist_ok=True)

    for child_id in child_ids:
        child_item = sb.get_item(child_id)
        child_title = child_item.get('title', 'No title')
        child_path = Path(dir + '/' + child_title)
        if not (os.path.isdir(child_path)):
            print(f"Downloading: {child_title} to {child_path}")
            child_path.mkdir(exist_ok=True)
            sb.get_item_files(child_item, child_path)

class PWFDF_Entry:
    def __init__(self, d):
        self.d = d
        self.utm_x = self.d['UTM_X']
        self.utm_y = self.d['UTM_Y']
        self.zone = self.d['UTM_Zone']

        utm_proj = Proj(proj='utm', zone=self.zone, ellps='WGS84', datum='WGS84', south=False)
        wgs84_proj = Proj(proj='latlong', ellps='WGS84', datum='WGS84')
        self.transformer = Transformer.from_proj(utm_proj, wgs84_proj, always_xy=True)

    def coordinates_wgs84(self):
        return self.transformer.transform(self.utm_x, self.utm_y)

    def bounds(self, buffer_km):
        buffer_m = buffer_km * 1000 # Convert buffer from km to meters

        # Calculate bounds in UTM
        utm_west  = self.utm_x - buffer_m
        utm_south = self.utm_y - buffer_m
        utm_east  = self.utm_x + buffer_m
        utm_north = self.utm_y + buffer_m

        west, south =  self.transformer.transform(utm_west, utm_south)
        east, north =  self.transformer.transform(utm_east, utm_north)

        return (west, south, east, north)

    def __getitem__(self, key):
        return self.d[key]

class PWFDF_Data:
    path = 'data/ofr20161106_appx-1.xlsx'
    sheet_name = 'Appendix1_ModelData'

    def __init__(self):
        self.df = pd.read_excel(self.path, sheet_name=self.sheet_name)

    def get(self, i):
        return PWFDF_Entry(self.df.iloc[i].to_dict())
    
    def get_data_target(self):
        from sklearn.preprocessing import StandardScaler, LabelEncoder

        numerical_features = [
            'UTM_X', 'UTM_Y', 'GaugeDist_m', 'StormDur_H', 'StormAccum_mm',
            'StormAvgI_mm/h', 'Peak_I15_mm/h', 'Peak_I30_mm/h', 'Peak_I60_mm/h',
            'ContributingArea_km2', 'PropHM23', 'dNBR/1000', 'KF',
            'Acc015_mm', 'Acc030_mm', 'Acc060_mm'
        ]

        categorical_features = ['State', 'UTM_Zone']
        label_encoders = {}
        for col in categorical_features:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col + '_encoded'] = le.fit_transform(self.df[col].astype(str))
                label_encoders[col] = le

        self.df[numerical_features] = self.df[numerical_features].fillna(self.df[numerical_features].mean()) # handle missing values
        feature_cols = numerical_features + [col + '_encoded' for col in categorical_features if col in self.df.columns] # combine features
        
        data = self.df[feature_cols].values
        target = self.df['Response'].values

        #normalize
        scaler = StandardScaler() 
        data = scaler.fit_transform(data)

        return data, target

def main():
    data = PWFDF_Data()
    i = 2
    entry = data.get(i)

    

    #print(f"Using entry: {i}, fire name: {entry['Fire Name']}, seg id: {entry['Fire_SegID']}")
    bounds = entry.bounds(5)
    #output_file = '/home/quinn/pwfdf/data/dem.tif'
    #elevation.clip(bounds=bounds, output=output_file, product='SRTM1')

    #get_satellite_image('/home/quinn/pwfdf/data/satellite.tif', bounds)
    get_fire_images(entry['Year'], bounds)

    #x, y = entry.coordinates_wgs84()
    #plot_dem(output_file, x, y)

    #delineate_watershed(output_file, entry.coordinates_wgs84())

if __name__ == '__main__':
    main()
