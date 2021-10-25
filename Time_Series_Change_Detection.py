# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import os
import numpy as np
import pandas as pd
from scipy import ndimage
from osgeo import gdal
import time
from joblib import Parallel, delayed
import ruptures as rpt
import calendar
from skimage import morphology
import glob
from sklearn.cluster import k_means
from skimage import measure
import seaborn as sns
from utils import paths
from pathlib import Path


"""
Read data file and prepare the input image
---------------------------------------------------------------------------------------------------------------------------------
"""
dirs = paths.load_paths()
Path = Path(dirs.DATA)
File    = glob.glob(Path + "/*.tif")[0]
Data    = gdal.Open(File, gdal.GA_ReadOnly)
Images  = Data.ReadAsArray()
ImagesT = np.reshape(Images, (Images.shape[0], Images.shape[1] * Images.shape[2])).transpose()

# Extract images dates
Ref         = pd.to_datetime("2015-01-01")
Dates       = []
ClassNames  = "{Unchanged"
for x in range(Images.shape[0]):
    Temp        = pd.to_datetime(Data.GetRasterBand(x + 1).GetDescription())     
    ClassNames  = ClassNames + ", " + Data.GetRasterBand(x + 1).GetDescription()    
    Dates.append(Temp) 
ClassNames  = ClassNames + "}"
Dates       = pd.to_datetime(Dates)
#Dates   = np.array((Dates - Ref).days) / 365.25
"""
-----------------------------------------------------------------------------------------------------------------------------------------------------
"""









"""
[1] Change variables 
-----------------------------------------------------------------------------------------------------------------------------------------------------
"""

def BreakPoint(y, x):
    
    # Preprocess the data to get rid of possible nan values and convert it into intensities
    idx     = np.isfinite(y) 
    y       = y[idx] 
    x       = x[idx]     
    y       = 10 ** (y / 10)
    
      
    
    # Change point detection
    # ...............................................................................................................................................
    algo    = rpt.BottomUp(model = "normal", min_size = round(len(y) / 10), jump = 1).fit(y) # Dynp, Binseg, BottomUp, Window
    Points1 = np.array(algo.predict(n_bkps = 1))[0] 

    # Ratio       
    Ratio11 = np.mean(y[Points1:]) / np.mean(y[0: Points1])
    Ratio12 = 1 / Ratio11 
    
    
    # Kullback-Liebler ddivergence
    Mu1     = np.mean(np.log(y[0: Points1]))
    std1    = np.std(np.log(y[0: Points1]), ddof = 1)
    Mu2     = np.mean(np.log(y[Points1:])) 
    std2    = np.std(np.log(y[Points1:]), ddof = 1)    
    KLD1    = ((Mu1 - Mu2) ** 2 + std1 ** 2 - std2 ** 2) / std2 ** 2 + ((Mu1 - Mu2) ** 2 + std2 ** 2 - std1 ** 2) / std1 ** 2

        
    return [Ratio11, Ratio12, KLD1, Points1]



start = time.time()
Time    = np.array((Dates - Ref).days) / 365.25
Result  = Parallel(n_jobs = 8)(delayed(BreakPoint)(y = ImagesT[i, :], x = Time) for i in range(ImagesT.shape[0]))
Result  = np.array(Result)
Result  = Result.transpose().reshape(Result.shape[1], Images.shape[1], Images.shape[2])
print(time.time() - start)

        
# Writing change variable to a file
BNames  = ["Ratio+ (CPT)", "Ratio- (CPT)", "KLD (CPT)", "Point (CPT)"] 
Drive   = gdal.GetDriverByName("ENVI")
Filen   = File.replace("Data", "Results").replace(".tif", " - Change Image")
OutData = Drive.Create(Filen, Result.shape[2], Result.shape[1], Result.shape[0], gdal.GDT_Float32)    
OutData.SetProjection(Data.GetProjection())
OutData.SetGeoTransform(Data.GetGeoTransform())
for x in range(Result.shape[0]):
    rb  = OutData.GetRasterBand(x + 1)
    rb.SetDescription(BNames[x])
    OutData.GetRasterBand(x + 1).WriteArray(Result[x, :, :])
OutData.FlushCache()                                                
OutData = None     
if os.path.exists(Filen + ".aux.xml"): os.remove(Filen + ".aux.xml")

"""
-----------------------------------------------------------------------------------------------------------------------------------------------------
"""









"""
[2] Change maps (Kmean clustering)
-----------------------------------------------------------------------------------------------------------------------------------------------------
"""

# Open chaneg variable file
Data        = gdal.Open(File.replace("Data", "Results").replace(".tif", " - Change Image"), gdal.GA_ReadOnly)
ChangeImage = Data.ReadAsArray()


# Loop over chaneg vafriable to be classified
ChangeMap   = np.zeros((0, Images.shape[1], Images.shape[2])).astype(int)
BandNames   = []
for x in [0, 1, 2]:
    
    # Clean change variable
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    var             = ChangeImage[x, :, :].reshape(-1, 1)     
    T1              = np.percentile(var, 2)
    T2              = np.percentile(var, 98)
    var[var < T1]   = T1
    var[var > T2]   = T2   
    
    # Clustering using kmean
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    clf             = k_means(var, n_clusters = 4, tol = 0.0001, init = "k-means++", n_jobs = 4, algorithm = "auto")
    CM              = clf[1] == np.argmax(clf[0])
    CM              = CM.reshape((ChangeImage[0, :, :].shape))      
    CM              = ndimage.median_filter(CM == 1, size = 7)
    CM              = morphology.remove_small_objects(CM == 1, min_size = 16, connectivity = 2)    
    ChangeMap       = np.vstack((ChangeMap, CM[np.newaxis, ...]))
    BandNames.append(Data.GetRasterBand(x + 1).GetDescription())  
    


# Writing change map to a file
Drive   = gdal.GetDriverByName("ENVI")
Filen   = File.replace("Data", "Results").replace(".tif", " - Change Map (Kmean)")
OutData = Drive.Create(Filen, ChangeMap.shape[2], ChangeMap.shape[1], ChangeMap.shape[0], gdal.GDT_Byte)    
OutData.SetProjection(Data.GetProjection())
OutData.SetGeoTransform(Data.GetGeoTransform())
for x in range(ChangeMap.shape[0]):
    rb  = OutData.GetRasterBand(x + 1)
    rb.SetDescription(BandNames[x])
    OutData.GetRasterBand(x + 1).WriteArray(ChangeMap[x, :, :])
OutData.FlushCache()                                                
OutData = None     
if os.path.exists(Filen + ".aux.xml"): os.remove(Filen + ".aux.xml")


# ---------------------------------------------------------------------------------------------------------------------------------------------------

ChangeTime      = np.zeros((0, Images.shape[1], Images.shape[2])).astype(int)
for x in range(7):
    
    if x in [0, 1, 2]:
        Temp    = ChangeImage[3, :, :] * (ChangeMap[x, :, :] == 1)
        Temp    = measure.label(Temp > 0)        
        Range   = np.unique(Temp)[1:]
        CT      = np.zeros((Temp.shape))
        for y in Range:
            idx     = np.where(Temp == y)
            CT[idx] = np.median(ChangeImage[3, :, :][idx])       
        ChangeTime  = np.vstack((ChangeTime, CT[np.newaxis, ...]))        



# Writing change time to a file
Drive   = gdal.GetDriverByName("ENVI")
Filen   = File.replace("Data", "Results").replace(".tif", " - Change Time (Kmean)")
OutData = Drive.Create(Filen, ChangeTime.shape[2], ChangeTime.shape[1], ChangeTime.shape[0], gdal.GDT_UInt16)    
OutData.SetProjection(Data.GetProjection())
OutData.SetGeoTransform(Data.GetGeoTransform())
for x in range(ChangeTime.shape[0]):
    rb  = OutData.GetRasterBand(x + 1)
    rb.SetDescription(BandNames[x])
    OutData.GetRasterBand(x + 1).WriteArray(ChangeTime[x, :, :])
OutData.FlushCache()                                                
OutData = None     
if os.path.exists(Filen + ".aux.xml"): os.remove(Filen + ".aux.xml")    
#   Adding metadata item to the heder file (file type : classification) 
Temp    = (np.array(sns.color_palette(n_colors=Images.shape[0])) * 255).astype(int).flatten()
color   = "{0, 0, 0"
for x in range(len(Temp)): color = color + ", " + str(Temp[x]) 
color   = color + "}"
with open(Filen + '.hdr', "a") as myfile:
    myfile.write("classes = " + str(Images.shape[0] + 1) + "\n")
    myfile.write("class names = " + ClassNames + "\n")
    myfile.write("class lookup = " + color + "\n")       

"""
-----------------------------------------------------------------------------------------------------------------------------------------------------
"""









"""
[3] Change maps aggregation
-----------------------------------------------------------------------------------------------------------------------------------------------------
"""

# Open change time map
Filen       = File.replace("Data", "Results").replace(".tif", " - Change Time (Kmean)")
Data        = gdal.Open(Filen, gdal.GA_ReadOnly)
ChangeTime  = Data.ReadAsArray()


step        = 3
ClassNames  = "{Unchanged"
CT          = np.zeros((ChangeTime.shape)).astype(int)


for y in [2016, 2017, 2018, 2019]:    
    for m in np.arange(1, 12, step):
        
        idx1        = np.logical_and(np.logical_and(Dates.month >= m, Dates.month <= m + step - 1), Dates.year == y)
        idx1        = np.argwhere(idx1) + 1
        if len(idx1) == 0:
            continue
        Max         = np.max(idx1)
        Min         = np.min(idx1)
        idx2        = np.logical_and(ChangeTime >= Min, ChangeTime <= Max)
        if (~idx2).all():
            continue
        CT[idx2]    = np.max(CT) + 1
        ClassNames  = ClassNames + ", " + str(y) + ": " + calendar.month_abbr[m] + " - " + calendar.month_abbr[m + step - 1]
ClassNames  = ClassNames + "}"      

    
# Writing aggregated change time to a file
Drive   = gdal.GetDriverByName("ENVI")
Filen   = File.replace("Data", "Results").replace(".tif", " - Change Time (Kmean) - agg")
OutData = Drive.Create(Filen, CT.shape[2], CT.shape[1], CT.shape[0], gdal.GDT_UInt16)    
OutData.SetProjection(Data.GetProjection())
OutData.SetGeoTransform(Data.GetGeoTransform())
for x in range(CT.shape[0]):
    rb  = OutData.GetRasterBand(x + 1)
    rb.SetDescription(Data.GetRasterBand(x + 1).GetDescription())
    OutData.GetRasterBand(x + 1).WriteArray(CT[x, :, :])
OutData.FlushCache()                                                
OutData = None     
if os.path.exists(Filen + ".aux.xml"): os.remove(Filen + ".aux.xml")    
#   Adding metadata item to the heder file (file type : classification) 
Temp    = (np.array(sns.color_palette(n_colors = np.max(CT))) * 255).astype(int).flatten()
color   = "{0, 0, 0"
for x in range(len(Temp)): color = color + ", " + str(Temp[x]) 
color   = color + "}"
with open(Filen + '.hdr', "a") as myfile:
    myfile.write("classes = " + str(np.max(CT+ 1)) + "\n")
    myfile.write("class names = " + ClassNames + "\n")
    myfile.write("class lookup = " + color + "\n")       

"""
-----------------------------------------------------------------------------------------------------------------------------------------------------
"""   





















