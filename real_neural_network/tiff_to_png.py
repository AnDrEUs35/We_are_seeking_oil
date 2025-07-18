from osgeo import gdal
import os
import rasterio

def only_2_band(path_to_dir, out_path):
    for file in os.listdir(path_to_dir):
        path_to_file = os.path.join(path_to_dir, file)
        with rasterio.open(path_to_file) as src:
            img_array = src.read(2)
            profile = src.profile
            profile.update({
                'count': 1
            })
        path_to_out_file = os.path.join(out_path, file)
        with rasterio.open(path_to_out_file, 'w', **profile) as img:
            img.write(img_array, 1)

def tiff_to_png():
    gdal.UseExceptions()
    masks = ''
    images = "C:\\Users\\Sirius\\Desktop\\[eqyz/"
    for f in os.listdir(images):
        in_path = images+f
        out_path = 'C:\\Users\\Sirius\\Desktop\\[eqyz2/'+os.path.splitext(f)[0]+'.png'
        ds = gdal.Translate(out_path, in_path, options="-scale -ot Byte")
        

if __name__ == '__main__':
    #only_2_band('C:\\Users\\Sirius\\Downloads\\Novoros.tif', 'C:\\Users\\Sirius\\Desktop\\Novoros_2band')
    tiff_to_png()