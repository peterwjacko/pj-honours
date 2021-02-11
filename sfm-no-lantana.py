# %%
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import georasters as gr
import geopandas as gp

import pylab
import scipy.stats as stats
import statsmodels.api as sm

# %%
# import data

lidarRaster = gr.from_file(r'D:\Dropbox\Honours\Peter_Woodfordia\Data\Lidar\Rasters\CHM_lidar_ClippedBB.tif')
droneRaster = gr.from_file(r'D:\Dropbox\Honours\Peter_Woodfordia\Data\Drone\DroneCHM_merged_1m.tif')
crowns = gp.read_file(r'D:\Dropbox\Honours\Peter_Woodfordia\Data\Vector\RandomBuffs.shp')
# %%
# drop lantana

crowns = crowns.loc[~crowns.Genus.isin(['Lantana'])]
print(crowns['Genus'].value_counts())
# %%
# subset crown polygons to Friday or Monday

buffsFriday = crowns.loc[crowns.ROI.isin(['A', 'B'])]
buffsMonday = crowns.loc[crowns.ROI.isin(['C', 'D'])]

buffsFriday.plot()
buffsMonday.plot()

print(len(buffsFriday), len(buffsMonday))
# %%
# clip raster

# Drone CHM #
droneFriday = droneRaster.clip(buffsFriday, keep=False)
droneMonday = droneRaster.clip(buffsMonday, keep=False)

# Lidar CHM #
lidarFriday = lidarRaster.clip(buffsFriday, keep=False)
lidarMonday = lidarRaster.clip(buffsMonday, keep=False)

droneFriday[1].plot()
lidarFriday[1].plot()
# %%
# extract values from raster to dataframe for each tree

# Drone CHM #
fridayDroneCHM = pd.DataFrame()
mondayDroneCHM = pd.DataFrame()

for crown in range(0, len(droneFriday), 1):
    clipped_df = gr.to_pandas(droneFriday[crown])
    fridayDroneCHM[crown] = clipped_df['value']

for crown in range(0, len(droneMonday), 1):
    clipped_df = gr.to_pandas(droneMonday[crown])
    mondayDroneCHM[crown] = clipped_df['value']

# Lidar CHM #
fridayLidarCHM = pd.DataFrame()
mondayLidarCHM = pd.DataFrame()

for crown in range(0, len(lidarFriday), 1):
    clipped_df = gr.to_pandas(lidarFriday[crown])
    fridayLidarCHM[crown] = clipped_df['value']

for crown in range(0, len(lidarMonday), 1):
    clipped_df = gr.to_pandas(lidarMonday[crown])
    mondayLidarCHM[crown] = clipped_df['value']
    
# fill NA with 0 because 0 represents ground level in a CHM
for CHMdataset in [fridayDroneCHM, mondayDroneCHM, fridayLidarCHM, mondayLidarCHM]:
    CHMdataset.fillna(0, inplace=True)
    
print("FridayDroneCHM data shape:", fridayDroneCHM.shape)
print(fridayDroneCHM.head())
print("MondayDroneCHM data shape:", mondayDroneCHM.shape)
print(mondayDroneCHM.head())
print("FridayLidarCHM data shape:", fridayLidarCHM.shape)
print(fridayLidarCHM.head())
print("MondayLidarCHM data shape:", mondayLidarCHM.shape)
print(mondayLidarCHM.head())
# %%
# take means of each tree

# Drone CHM #
fridayDroneMeans = pd.DataFrame(fridayDroneCHM.mean())
#fridayDroneMeans = pd.DataFrame(fridayDroneMeans.sample(n=41)) # friday has more samples so randomly subsetted to be the same size as monday (n=41)
mondayDroneMeans = pd.DataFrame(mondayDroneCHM.mean())
# Lidar CHM #
fridayLidarMeans = pd.DataFrame(fridayLidarCHM.mean())
#fridayLidarMeans = pd.DataFrame(fridayLidarMeans.sample(n=41)) # friday has more samples so randomly subsetted to be the same size as monday (n=41)
mondayLidarMeans = pd.DataFrame(mondayLidarCHM.mean())
print(fridayDroneMeans.head())  
# %%
# melt data into two columns (x 2 dataframes)

# Drone CHM #
fridayDroneMelt = fridayDroneMeans.melt()
fridayDroneMelt['variable'] = 'Friday'
fridayDroneMelt['CHM'] = 'Drone'
mondayDroneMelt = mondayDroneMeans.melt()
mondayDroneMelt['variable'] = 'Monday'
mondayDroneMelt['CHM'] = 'Drone'

# Lidar CHM #
fridayLidarMelt = fridayLidarMeans.melt()
fridayLidarMelt['variable'] = 'Friday'
fridayLidarMelt['CHM'] = 'Lidar'
mondayLidarMelt = mondayLidarMeans.melt()
mondayLidarMelt['variable'] = 'Monday'
mondayLidarMelt['CHM'] = 'Lidar'

print(fridayDroneMelt.head())
print(fridayLidarMelt.head())
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
fridayDroneValues = droneHeightsStacked.loc[droneHeightsStacked['Day'] == 'Friday', 'Height']
mondayDroneValues = droneHeightsStacked.loc[droneHeightsStacked['Day'] == 'Monday', 'Height']

# Lidar CHM #
fridayLidarValues = lidarHeightsStacked.loc[lidarHeightsStacked['Day'] == 'Friday', 'Height']
mondayLidarValues = lidarHeightsStacked.loc[lidarHeightsStacked['Day'] == 'Monday', 'Height']

print(fridayDroneValues.head())
print(fridayLidarValues.head())
# %%
# test for normal distribution
for count, heightValues in enumerate([fridayDroneValues, mondayDroneValues, fridayLidarValues, mondayLidarValues]):
    print("Shapiro:", count, stats.shapiro(heightValues))
    sm.qqplot(heightValues, loc=4, scale=3, line='s')
    plt.figure(figsize=(9,6))
    sns.histplot(x=heightValues,
                    bins=25,
                    kde=True)
            
    
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

sns.histplot(x='Height',
             data=droneHeightsStacked,
              bins=25,
              hue='Day',
              kde=True).set_title("Drone CHM: Less windy (Friday) Vs Windy (Monday)")
# %%
# boxplot of drone heights between two days
sns.boxplot(x=droneHeightsStacked['Day'],
            y=droneHeightsStacked['Height'],
            hue=None,
            orient="v").set_title("Drone CHM: Less windy (Friday) Vs Windy (Monday)")
plt.show()
# %%
# Histogram of friday CHM: Drone Vs Lidar
plt.figure(figsize=(9,6))

sns.histplot(x='Height',
             data=lidarDroneStacked[lidarDroneStacked['Day']=='Friday'],
              bins=25,
              hue='CHM',
              kde=True).set_title("Less windy day (Friday): Drone-CHM Vs Lidar-CHM")
# %%
# boxplot of friday CHM: Drone Vs Lidar
sns.boxplot(x='CHM',
            y='Height',
            data=lidarDroneStacked[lidarDroneStacked['Day']=='Friday'],
            hue=None,
            orient="v").set_title("Less windy day (Friday): Drone-CHM Vs Lidar-CHM")
plt.show()
# %%
# Histogram of monday CHM: Drone Vs Lidar
plt.figure(figsize=(9,6))

sns.histplot(x='Height',
             data=lidarDroneStacked[lidarDroneStacked['Day']=='Monday'],
              bins=25,
              hue='CHM',
              kde=True).set_title("Windy day (Monday): Drone-CHM Vs Lidar-CHM")
# %%
# boxplot of monday CHM: Drone Vs Lidar
sns.boxplot(x='CHM',
            y='Height',
            data=lidarDroneStacked[lidarDroneStacked['Day']=='Monday'],
            hue=None,
            orient="v").set_title("Windy day (Monday): Drone-CHM Vs Lidar-CHM")
plt.show()
# %%
# Histogram of both days: Drone Vs Lidar
plt.figure(figsize=(9,6))

sns.histplot(x='Height',
             data=lidarDroneStacked,
              bins=25,
              hue='CHM',
              kde=True).set_title("Both days: Drone-CHM Vs Lidar-CHM")
# %%
# boxplot of both days: Drone Vs Lidar
sns.boxplot(x='CHM',
            y='Height',
            data=lidarDroneStacked,
            hue=None,
            orient="v").set_title("Both days: Drone-CHM Vs Lidar-CHM")
plt.show()
# %%
lidarDroneStacked.to_csv(r"D:\Dropbox\Honours\Peter_Woodfordia\Output\lidarDroneMeanHeights.csv")
fridayDroneCHM.to_csv(r"D:\Dropbox\Honours\Peter_Woodfordia\Output\fridayAllValues_Drone.csv")
mondayDroneCHM.to_csv(r"D:\Dropbox\Honours\Peter_Woodfordia\Output\mondayAllValues_Drone.csv")
fridayLidarCHM.to_csv(r"D:\Dropbox\Honours\Peter_Woodfordia\Output\fridayAllValues_Lidar.csv")
mondayLidarCHM.to_csv(r"D:\Dropbox\Honours\Peter_Woodfordia\Output\mondayAllValues_Lidar.csv")
# %%
