# Import the Earth Engine API and initialize it.
import ee

ee.Initialize()

#----Test area----
List4STcoordinates = [-116.45507813235129, 43.644026768116866, -116.41113370284677, 43.67582227697864]

'''
#----Stockholm----
List4STcoordinates = [17.76331983597109,59.31499591460535,17.978926525424214,59.4314815159509]
List4STcoordinates = [17.763347578325742,59.217699163355164,18.010196638384336,59.31910557738377]
List4STcoordinates = [17.996003123641895,59.21795420903652,18.23907563340752,59.31585605950281]
List4STcoordinates = [17.968953749817928,59.30727289813106,18.23605885235699,59.43111917763542]
'''

'''
#----Beijing----
List4STcoordinates = [116.271911974095,39.95981711663888,116.59944188132157,40.015371360300525]
Point = ee.Geometry.Point([116.4642472188732,39.89762401074957])
List4STcoordinates = [116.271911974095,39.845453847746754,116.59944188132157,40.08595655781435]
Point = ee.Geometry.Point([116.4642472188732,39.89762401074957])
'''

'''
# ----Shanghai----
List4STcoordinates = [121.35409971045301, 31.086658835567963, 121.62601133154676, 31.320407970243643]
Point = ee.Geometry.Point([121.49828475478472, 31.218493025349897])
'''

PatchSize = 2;
index1 = 0 / PatchSize
index2 = 0 / PatchSize
PatchROI = [1, 2, 3, 4]
PatchROI[0] = List4STcoordinates[0] * (1 - index1) + List4STcoordinates[2] * index1;
PatchROI[2] = List4STcoordinates[0] * (1 - 1 / PatchSize - index1) + List4STcoordinates[2] * (index1 + 1 / PatchSize);
PatchROI[1] = List4STcoordinates[1] * (1 - index2) + List4STcoordinates[3] * index2;
PatchROI[3] = List4STcoordinates[1] * (1 - 1 / PatchSize - index2) + List4STcoordinates[3] * (index2 + 1 / PatchSize);

OutputROI = ee.Geometry.Rectangle(PatchROI)
coordList = ee.List(OutputROI.coordinates().get(0));
p0 = ee.Geometry.Point(coordList.get(0));
p1 = ee.Geometry.Point(coordList.get(1));
p2 = ee.Geometry.Point(coordList.get(2));
p3 = ee.Geometry.Point(coordList.get(3));
bands = ['VV'];
sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD')

# ----Test area----
imageDataset = sentinel1\
    .filterBounds(p0).filterBounds(p1).filterBounds(p2).filterBounds(p3)\
    .filterDate('2018-01-01', '2021-01-31')\
    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))\
    .filter(ee.Filter.eq('relativeOrbitNumber_stop', 71))\
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))

'''
#----Stockholm----
imageDataset = sentinel1.filterBounds(p0).filterBounds(p1).filterBounds(p2).filterBounds(p3).filterDate('2017-01-01', '2019-12-31').filter(ee.Filter.eq('orbitProperties_pass','DESCENDING')).filter(ee.Filter.eq('relativeOrbitNumber_stop',22)).filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'));

#----Beijing----
imageDataset = sentinel1.filterBounds(Point).filterDate('2017-01-01', '2019-12-31').filter(ee.Filter.eq('orbitProperties_pass','DESCENDING')).filter(ee.Filter.eq('relativeOrbitNumber_stop',47)).filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'));
'''
'''
# ----Shanghai----
imageDataset = sentinel1.filterBounds(Point).filterDate('2017-01-01', '2019-12-31').filter(
    ee.Filter.eq('orbitProperties_pass', 'ASCENDING')).filter(ee.Filter.eq('relativeOrbitNumber_stop', 171)).filter(
    ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'));
'''
imageDataset = imageDataset.sort('SLC_Processing_start', True)
img = imageDataset.first().select(bands)
img = img.rename(img.date().format().slice(0, 10))

List = imageDataset.select(bands).toList(200)
Listlen = List.length()
print(Listlen.getInfo())

for Numimage in range(Listlen.getInfo() - 1):
    img2 = ee.Image(List.get(Numimage + 1))
    img = img.addBands(img2.rename(img2.date().format().slice(0, 10)))

FileName = 'ROI-' + str(index1 * PatchSize) + '-' + str(index2 * PatchSize)
XX = img.bandNames()
print(XX.getInfo())

# Setup the task.
# Specify patch and file dimensions.
imageTask = ee.batch.Export.image.toDrive(
    image=img,
    description=FileName,
    folder="KM",
    scale=10,
    maxPixels=1e12,
    region=OutputROI
)

import time

# Start the task.
imageTask.start()

while imageTask.active():
    print('Polling for task (id: {}).'.format(imageTask.id))
    time.sleep(10)
print('Done with image export.')