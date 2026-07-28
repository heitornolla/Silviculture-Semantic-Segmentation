import os
import glob
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.windows import Window

tifs = 'data/Dataset_Silvicultura/'
gpkg = 'data/floresta_plantada_ief_2025.gpkg'
out_img = 'data/patches/images/'
out_mask = 'data/patches/masks/'

os.makedirs(out_img, exist_ok=True)
os.makedirs(out_mask, exist_ok=True)

PATCH_SIZE = 128

gdf_silv = gpd.read_file(gpkg)
gdf_silv = gdf_silv[~gdf_silv.geometry.is_empty & gdf_silv.geometry.is_valid]

# GPKG CRS must be the same as the TIF's (EPSG:31983)
gdf_silv = gdf_silv.to_crs(epsg=31983)

files = glob.glob(os.path.join(tifs, '*.tif'))
cities = list(set([os.path.basename(f).split('_')[1] for f in files]))

patch_id = 0

for city in cities:
    print(f"Processing: {city}")
    
    # Order images chronologically
    city_tifs = sorted(glob.glob(os.path.join(tifs, f'S2_{city}_*.tif')))
    
    # Uses the first image as a mold
    with rasterio.open(city_tifs[0]) as src_mold:
        meta = src_mold.meta
        width = meta['width']
        height = meta['height']
        transform = meta['transform']
        
        # Rasterizes GT 
        city_geom = gdf_silv.geometry
        
        # 1 = Silviculture, 0 = Background
        city_mask = features.rasterize(
            shapes=[(geom, 1) for geom in city_geom],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype='uint8'
        )
        
    # Get patches
    for i in range(0, height - PATCH_SIZE + 1, PATCH_SIZE):
        for j in range(0, width - PATCH_SIZE + 1, PATCH_SIZE):
            
            window = Window(j, i, PATCH_SIZE, PATCH_SIZE)
            patch_mask = city_mask[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            
            # to ignore bg-only patches:
            # if patch_mask.sum() == 0: continue

            # Only accepts patches with at least
            # 128 * 128 = 16384 pixels. 5% = ~819 pixels.
            # if patch_mask.sum() < 800: 
            #     continue
            
            time_series = []
            inv_patch = False
            
            for temporal_tif in city_tifs:
                with rasterio.open(temporal_tif) as src:
                    patch_img = src.read(window=window)

                    patch_img = np.nan_to_num(patch_img, nan=0).astype(np.int16)
                    
                    if np.all(patch_img[0] == 0):
                        inv_patch = True
                        break
                        
                    time_series.append(patch_img)
            
            if inv_patch:
                continue

            # Stacks over time (T, C, H, W)
            # city_tifs has 10 files (5 years x 2 seasons) = (10, 10, 128, 128)
            patch_tensor = np.stack(time_series, axis=0)
            
            np.save(os.path.join(out_img, f'patch_{patch_id:05d}.npy'), patch_tensor)
            np.save(os.path.join(out_mask, f'mask_{patch_id:05d}.npy'), patch_mask)
            
            patch_id += 1

print(f"{patch_id} patches generated")
