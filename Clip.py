import gdal
import numpy as np
from skimage import io

in_ds = gdal.Open("/sniper/data/Aden/Aden110515YN1.tiff")
print("open tif file succeed")

in_band1 = in_ds.GetRasterBand(1)
in_band2 = in_ds.GetRasterBand(2)
in_band3 = in_ds.GetRasterBand(3)
# in_band4 = in_ds.GetRasterBand(4)

# image = in_band1.ReadAsArray() #<class 'tuple'>: (88576, 194304)
shape = (88576, 194304)

target_size = 10000# target_size = 1024

patchs = []
for row_begin in range(0, shape[0],target_size):
    for col_begin in range(0, shape[1],target_size):
        row_end = row_begin + target_size
        col_end = col_begin + target_size
        if row_end > shape[0]:
            row_end = shape[0]
            # row_begin = row_end - target_size
        if col_end > shape[1]:
            col_end = shape[1]
            # col_begin = col_end - target_size
        patchs.append((row_begin, row_end, col_begin, col_end))
for b in patchs:


    offset_x = b[2]#col_begin
    offset_y = b[0]#row_begin


    block_xsize = b[3]-b[2]# target_size 
    block_ysize = b[1]-b[0]# target_size  

    out_band1 = in_band1.ReadAsArray(offset_x, offset_y, block_xsize, block_ysize)[:,:, np.newaxis]
    out_band2 = in_band2.ReadAsArray(offset_x, offset_y, block_xsize, block_ysize)[:,:, np.newaxis]
    out_band3 = in_band3.ReadAsArray(offset_x, offset_y, block_xsize, block_ysize)[:,:, np.newaxis]

    img = np.concatenate((out_band1,out_band2,out_band3),axis=2)
    if np.max(img)!=0:
        path = '/sniper/data/Patches/'+'Aden_'+'_'.join('%s' % i for i in b)+'.tif'
        io.imsave(path,img)
        print("succeed"+path)

print("End!")


