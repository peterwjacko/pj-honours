# %%
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import georasters as gr
import geopandas as gp

import scipy.stats as stats
# %%
# import data

lidarRaster = gr.from_file(r'D:\Dropbox\Honours\Peter_Woodfordia\Data\Lidar\Rasters\CHM_lidar_ClippedBB.tif')
droneRaster = gr.from_file(r'D:\Dropbox\Honours\Peter_Woodfordia\Data\Drone\DroneCHM_merged_1m.tif')
crowns = gp.read_file(r'D:\Dropbox\Honours\Peter_Woodfordia\Data\Vector\RandomBuffs.shp')
# %%
# subset crown polygons to Friday or Monday

buffsFriday = crowns.loc[crowns.ROI.isin(['A', 'B'])]
buffsMonday = crowns.loc[crowns.ROI.isin(['C', 'D'])]

buffsFriday = gp.GeoDataFrame(pd.DataFrame(buffsFriday.sample(n=41, axis='index')))

print(buffsFriday.head())
print(buffsFriday.shape, buffsMonday.shape)
# %%
# clip raster

# Drone CHM #
droneFriday = droneRaster.clip(buffsFriday, keep=True)
droneMonday = droneRaster.clip(buffsMonday, keep=True)

# Lidar CHM #
lidarFriday = lidarRaster.clip(buffsFriday, keep=True)
lidarMonday = lidarRaster.clip(buffsMonday, keep=True)
# %%
buffsMonday.head()
# %%
droneMonday.loc[droneMonday['ObjectNum'] == 334, 'GeoRaster'].item().plot()
lidarMonday.loc[lidarMonday['ObjectNum'] == 334, 'GeoRaster'].item().plot()
# %%
# sort by objectNum

for clippedCHM in [droneFriday, droneMonday, lidarFriday, lidarMonday]:
    clippedCHM.sort_values(by='ObjectNum', inplace=True)

print(droneFriday.shape)
print(lidarFriday.shape)
print(droneMonday.shape)
print(lidarMonday.shape)
# %%
# extract values from raster to dataframe for each tree

# Drone CHM #
fridayDroneCHM = pd.DataFrame()
mondayDroneCHM = pd.DataFrame()

for count, ObjNum in enumerate(droneFriday['ObjectNum']):
    clipped_df = gr.to_pandas(droneFriday['GeoRaster'].iloc[count])
    fridayDroneCHM[ObjNum] = clipped_df['value']

for count, ObjNum in enumerate(droneMonday['ObjectNum']):
    clipped_df = gr.to_pandas(droneMonday['GeoRaster'].iloc[count])
    mondayDroneCHM[ObjNum] = clipped_df['value']

# Lidar CHM #
fridayLidarCHM = pd.DataFrame()
mondayLidarCHM = pd.DataFrame()

for count, ObjNum in enumerate(lidarFriday['ObjectNum']):
    clipped_df = gr.to_pandas(lidarFriday['GeoRaster'].iloc[count])
    fridayLidarCHM[ObjNum] = clipped_df['value']

for count, ObjNum in enumerate(lidarMonday['ObjectNum']):
    clipped_df = gr.to_pandas(lidarMonday['GeoRaster'].iloc[count])
    mondayLidarCHM[ObjNum] = clipped_df['value']
    
    
# fill NA with 0 because 0 represents ground level in a CHM
for CHMdataset in [fridayDroneCHM, mondayDroneCHM, fridayLidarCHM, mondayLidarCHM]:
    CHMdataset.fillna(0, inplace=True)
       
print("FridayDroneCHM data shape:", fridayDroneCHM.shape)
#print(fridayDroneCHM.head())
print("MondayDroneCHM data shape:", mondayDroneCHM.shape)
#print(mondayDroneCHM.head())
print("FridayLidarCHM data shape:", fridayLidarCHM.shape)
#print(fridayLidarCHM.head())
print("MondayLidarCHM data shape:", mondayLidarCHM.shape)
#print(mondayLidarCHM.head())
# %%
print(mondayDroneCHM.head())
print(mondayLidarCHM.head())
# %%
# take means of each tree

# Drone CHM #
fridayDroneMeans = pd.DataFrame(fridayDroneCHM.mean())
mondayDroneMeans = pd.DataFrame(mondayDroneCHM.mean())

# Lidar CHM #
fridayLidarMeans = pd.DataFrame(fridayLidarCHM.mean())
mondayLidarMeans = pd.DataFrame(mondayLidarCHM.mean())

print(len(mondayDroneMeans)) 
print(fridayLidarMeans)    
# %%
# melt data into two columns (x 2 dataframes)

# Drone CHM #
fridayDroneMelt = fridayDroneMeans.melt()
fridayDroneMelt['variable'] = 'Day 1'
fridayDroneMelt['CHM'] = 'UAV'
fridayDroneMelt['ObjNum'] = list(fridayDroneMeans.index.values)
mondayDroneMelt = mondayDroneMeans.melt()
mondayDroneMelt['variable'] = 'Day 2'
mondayDroneMelt['CHM'] = 'UAV'
mondayDroneMelt['ObjNum'] = list(mondayDroneMeans.index.values)

# Lidar CHM #
fridayLidarMelt = fridayLidarMeans.melt()
fridayLidarMelt['variable'] = 'Day 1'
fridayLidarMelt['CHM'] = 'ALS'
fridayLidarMelt['ObjNum'] = list(fridayLidarMeans.index.values)
mondayLidarMelt = mondayLidarMeans.melt()
mondayLidarMelt['variable'] = 'Day 2'
mondayLidarMelt['CHM'] = 'ALS'
mondayLidarMelt['ObjNum'] = list(mondayLidarMeans.index.values)
# %%
# stack both days for each CHM

# Drone CHM #
droneHeightsStacked = pd.concat([fridayDroneMelt, mondayDroneMelt])
droneHeightsStacked.rename(columns = {'variable': 'Day',
                                      'value': 'Height'},
                           inplace=True)

# Lidar CHM #
lidarHeightsStacked = pd.concat([fridayLidarMelt, mondayLidarMelt])
lidarHeightsStacked.rename(columns = {'variable': 'Day',
                                      'value': 'Height'},
                           inplace=True)

# stacked both (for plots) #
lidarDroneStacked = pd.concat([fridayDroneMelt, mondayDroneMelt, fridayLidarMelt, mondayLidarMelt])
lidarDroneStacked.rename(columns = {'variable': 'Day',
                                    'value': 'Height'},
                         inplace=True)

print(droneHeightsStacked.head())
print(lidarHeightsStacked.head())
# %%
# subset data for wilcoxon (must be single series of values)

# DroneCHM #
fridayDroneValues = droneHeightsStacked.loc[droneHeightsStacked['Day'] == 'Day 1', 'Height']
mondayDroneValues = droneHeightsStacked.loc[droneHeightsStacked['Day'] == 'Day 2', 'Height']

# Lidar CHM #
fridayLidarValues = lidarHeightsStacked.loc[lidarHeightsStacked['Day'] == 'Day 1', 'Height']
mondayLidarValues = lidarHeightsStacked.loc[lidarHeightsStacked['Day'] == 'Day 2', 'Height']

print(fridayDroneValues.head())
print(fridayLidarValues.head())
# %%
# perform wilcoxon

# not windy #
notWindyWilcox = stats.wilcoxon(fridayDroneValues, fridayLidarValues, alternative="two-sided")

# windy wilcox
windyWilcox = stats.wilcoxon(mondayDroneValues, mondayLidarValues, alternative="two-sided")

print(notWindyWilcox)
print(windyWilcox)
# %%
# Histogram of drone heights between two days
plt.figure(figsize=(9,6))
sns.set_theme(style="ticks")
bothDaysDroneHist = sns.histplot(x='Height',
                            data=droneHeightsStacked,
                            bins=25,
                            hue='Day',
                            kde=True)
bothDaysDroneHistfig = bothDaysDroneHist.get_figure()
# %%
# boxplot of drone heights between two days
bothDaysBox = sns.boxplot(x=droneHeightsStacked['Day'],
                            y=droneHeightsStacked['Height'],
                            hue=None,
                            orient="v")
plt.show()
bothDaysBoxfig = bothDaysBox.get_figure()
# %%
# Histogram of friday CHM: Drone Vs Lidar
plt.figure(figsize=(9,6))

fridayHist = sns.histplot(x='Height',
                            data=lidarDroneStacked[lidarDroneStacked['Day']=='Day 1'],
                            bins=25,
                            hue='CHM',
                            kde=True)
fridayHistfig = fridayHist.get_figure()
# %%
# boxplot of friday CHM: Drone Vs Lidar
fridayBox = sns.boxplot(x='CHM',
                        y='Height',
                        data=lidarDroneStacked[lidarDroneStacked['Day']=='Day 1'],
                        hue=None,
                        orient="v")
plt.show()
fridayBoxfig = fridayBox.get_figure()
# %%
# Histogram of monday CHM: Drone Vs Lidar
plt.figure(figsize=(9,6))

mondayHist = sns.histplot(x='Height',
                            data=lidarDroneStacked[lidarDroneStacked['Day']=='Day 2'],
                            bins=25,
                            hue='CHM',
                            kde=True)
mondayHistfig = mondayHist.get_figure()
# %%
# boxplot of monday CHM: Drone Vs Lidar
mondayBox = sns.boxplot(x='CHM',
                        y='Height',
                        data=lidarDroneStacked[lidarDroneStacked['Day']=='Day 2'],
                        hue=None,
                        orient="v")
plt.show()
mondayBoxfig = mondayBox.get_figure()
# %%
# Histogram of both days: Drone Vs Lidar
plt.figure(figsize=(9,6))

bothDaysBothCHMHist = sns.histplot(x='Height',
                            data=lidarDroneStacked,
                            bins=25,
                            hue='CHM',
                            kde=True)
bothDaysBothCHMHistfig = bothDaysBothCHMHist.get_figure()
# %%
# boxplot of both days: Drone Vs Lidar
bothDaysBothCHMBox = sns.boxplot(x='CHM',
                            y='Height',
                            data=lidarDroneStacked,
                            hue=None,
                            orient="v")
plt.show()
bothDaysBothCHMBoxfig = bothDaysBothCHMBox.get_figure()
# %%
lidarDroneStacked.to_csv(r"D:\Dropbox\Honours\Peter_Woodfordia\Output\lidarDroneMeanHeights.csv")
fridayDroneCHM.to_csv(r"D:\Dropbox\Honours\Peter_Woodfordia\Output\fridayAllValues_Drone.csv")
mondayDroneCHM.to_csv(r"D:\Dropbox\Honours\Peter_Woodfordia\Output\mondayAllValues_Drone.csv")
fridayLidarCHM.to_csv(r"D:\Dropbox\Honours\Peter_Woodfordia\Output\fridayAllValues_Lidar.csv")
mondayLidarCHM.to_csv(r"D:\Dropbox\Honours\Peter_Woodfordia\Output\mondayAllValues_Lidar.csv")
bothDaysDroneHistfig.savefig(r'C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\bothDaysDroneHist.pdf', bbox_inches='tight', dpi=300)
bothDaysBoxfig.savefig(r'C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\bothDaysDroneBox.pdf', bbox_inches='tight', dpi=300)
fridayHistfig.savefig(r'C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\fridayHist.pdf', bbox_inches='tight', dpi=300)
fridayBoxfig.savefig(r'C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\fridayBox.pdf', bbox_inches='tight', dpi=300)
mondayHistfig.savefig(r'C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\mondayHist.pdf', bbox_inches='tight', dpi=300)
mondayBoxfig.savefig(r'C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\mondayBox.pdf', bbox_inches='tight', dpi=300)
bothDaysBothCHMHistfig.savefig(r'C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\bothDaysBothCHMHist.pdf', bbox_inches='tight', dpi=300)
bothDaysBothCHMBoxfig.savefig(r'C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\bothDaysBothCHMBox.pdf', bbox_inches='tight', dpi=300)
# %%

for count, raster in enumerate(droneFriday['GeoRaster']):
    name = droneFriday['ObjectNum'].iloc[count]
    raster.to_tiff(fr'D:\Dropbox\Honours\Peter_Woodfordia\Output\HeightComparison\Datasets\UAV-Day1\{name}.tif')

for count, raster in enumerate(droneMonday['GeoRaster']):
    name = droneMonday['ObjectNum'].iloc[count]
    raster.to_tiff(fr'D:\Dropbox\Honours\Peter_Woodfordia\Output\HeightComparison\Datasets\UAV-Day2\{name}.tif')

for count, raster in enumerate(lidarFriday['GeoRaster']):
    name = lidarFriday['ObjectNum'].iloc[count]
    raster.to_tiff(fr'D:\Dropbox\Honours\Peter_Woodfordia\Output\HeightComparison\Datasets\ALS-Day1\{name}.tif')

for count, raster in enumerate(lidarMonday['GeoRaster']):
    name = lidarMonday['ObjectNum'].iloc[count]
    raster.to_tiff(fr'D:\Dropbox\Honours\Peter_Woodfordia\Output\HeightComparison\Datasets\ALS-Day2\{name}.tif')
# %%
