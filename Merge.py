import gdal
import numpy as np
from skimage import io
import os
from tqdm import tqdm

in_ds = gdal.Open("/sniper/data/Aden/Aden110515YN1.tiff")
print("open tif file succeed")

cols = in_ds.RasterXSize
rows = in_ds.RasterYSize
bands = in_ds.RasterCount

# image = in_band1.ReadAsArray() #<class 'tuple'>: (88576, 194304)
# shape = (88576, 194304)
# Pre = np.zeros((rows,cols,bands)).astype(np.uint8)

target_size = 10000# target_size = 1024



gtif_driver = gdal.GetDriverByName("GTiff")


out_ds = gtif_driver.Create('/sniper/data/Aden_result/Pre.tif', cols,rows, 3)
print("create new tif file succeed")


out_band1 = out_ds.GetRasterBand(1)
out_band2 = out_ds.GetRasterBand(2)
out_band3 = out_ds.GetRasterBand(3)

xBlockSize = target_size
yBlockSize = target_size

# for i in range(0, rows, yBlockSize):
#    if i + yBlockSize < rows:
#         numRows = yBlockSize
#    else:
#         numRows = rows -i
#    for j in range(0, cols, xBlockSize):
#         if j + xBlockSize < cols:
#              numCols = xBlockSize
#         else:
#              numCols = cols -j
#         # data = band.ReadAsArray(j, i, numCols, numRows)
#         # do calculations here to create outData array
#
#         patch_ds = gdal.Open("/data/Aden/Aden110515YN1.tiff")
#         # out1.WriteArray(patch_ds, j, i)
#
#         out_band1.WriteArray(patch_ds.GetRasterBand(1), j, i)
#         out_band2.WriteArray(patch_ds.GetRasterBand(2), j, i)
#         out_band3.WriteArray(patch_ds.GetRasterBand(3), j, i)


# for i in range(0, rows, yBlockSize):
#    if i + yBlockSize < rows:
#         numRows = yBlockSize
#    else:
#         numRows = rows -i
#    for j in range(0, cols, xBlockSize):
#         if j + xBlockSize < cols:
#              numCols = xBlockSize
#         else:
#              numCols = cols -j
        # data = band.ReadAsArray(j, i, numCols, numRows)
        # do calculations here to create outData array

file_dir = '/sniper/data/result'
for root, dirs, files in os.walk(file_dir):
    for file in tqdm(files):
        path = os.path.join(root, file)
        row_begin,col_begin=file.split('_')[1], file.split('_')[3]  #(row_begin, row_end, col_begin, col_end)  'Aden_60000_70000_50000_60000.tif'
        patch_ds = gdal.Open(path)
            # out1.WriteArray(patch_ds, j, i)
        # patch_band1 = patch_ds.GetRasterBand(1)
        # patch_band2 = patch_ds.GetRasterBand(2)
        # patch_band3 = patch_ds.GetRasterBand(3)

        out_band1.WriteArray(patch_ds.GetRasterBand(1).ReadAsArray(), int(col_begin), int(row_begin))
        out_band2.WriteArray(patch_ds.GetRasterBand(2).ReadAsArray(), int(col_begin), int(row_begin))
        out_band3.WriteArray(patch_ds.GetRasterBand(3).ReadAsArray(), int(col_begin), int(row_begin))

ori_transform = in_ds.GetGeoTransform()
if ori_transform:
    print (ori_transform)
    print("Origin = ({}, {})".format(ori_transform[0], ori_transform[3]))
    print("Pixel Size = ({}, {})".format(ori_transform[1], ori_transform[5]))


out_ds.SetGeoTransform(in_ds.GetGeoTransform())


out_ds.SetProjection(in_ds.GetProjection())


out_ds.GetRasterBand(1).WriteArray(out_band1.ReadAsArray())
out_ds.GetRasterBand(2).WriteArray(out_band2.ReadAsArray())
out_ds.GetRasterBand(3).WriteArray(out_band3.ReadAsArray())

out_ds.FlushCache()
print("FlushCache succeed")



del out_ds

print("End!")

