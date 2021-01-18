
# https://reneshbedre.github.io/blog/anova.html

# %%
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import georasters as gr
import geopandas as gp

import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import spm1d

# %%
# import data

lidarRaster = gr.from_file(r'D:\Dropbox\Honours\Peter_Woodfordia\Data\Lidar\Rasters\LidarCHM_ROI_A_Clipped.tif')
droneRaster = gr.from_file(r'D:\Dropbox\Honours\Peter_Woodfordia\Data\Drone\ROI_A\DroneCHM_ROI_A_Clipped_1m.tif')
crowns = gp.read_file(r'D:\Dropbox\Honours\Peter_Woodfordia\Data\Vector\RandomBuffs.shp')
#heights = pd.read_csv(r'D:\Dropbox\Honours\Peter_Woodfordia\Data\MaxHeights_AllMethods.csv')

# %%
heights = heights[['Height', 'DroneCHMROI_2', 'LidarCHMROI_2']]
heights.rename(columns = {'Height': 'Hypsometer',
                          'DroneCHMROI_2': 'SfM CHM',
                          'LidarCHMROI_2': 'Lidar CHM'},
               inplace=True)

# %%
heightsMelt = heights.melt()
heightsMelt.rename(columns = {'variable': 'CHM',
                              'value': 'Height'},
                   inplace=True)

# %%
# plot heights
sns.boxplot(x=heightsMelt['CHM'],
            y=heightsMelt['Height'],
            orient="v")
plt.show()
# %%
# one-way ANOVA

lm = ols('Height ~ C(CHM)', heightsMelt).fit()
resultsTable = sm.stats.anova_lm(lm, typ=2)
print(table)

# %%
# check normality

shapiroResults = stats.shapiro(lm.resid)
shapiroResults

# %%
# plot residuals

fig = plt.figure(figsize= (10, 10))
ax = fig.add_subplot(111)

normality_plot, stat = stats.probplot(lm.resid, plot= plt, rvalue= True)
ax.set_title("Probability plot of model residual's", fontsize= 20)
ax.set

plt.show()

# %%
# check homogeneity of variance

leveneResults = stats.levene(heights['Hypsometer'],
                             heights['SfM CHM'],
                             heights['Lidar CHM'])
leveneResults

# %%
plt.figure(figsize=(9,6))

sns.histplot(x='Height',
             data=heightsMelt,
              bins=25,
              hue='CHM',
              kde=True)

# %%
# subset crowns to ROI A for testing

BuffsROIA = crowns.loc[crowns['ROI'] == 'A']
BuffsROIA.head()

# %%
# clip raster with shape

lidarClipped = lidarRaster.clip(BuffsROIA, keep=False)
droneClipped = droneRaster.clip(BuffsROIA, keep=False)
# %%
lidarClipped[8].plot()

# %%
lidarROIA = pd.DataFrame()
droneROIA = pd.DataFrame()

for crown in range(0, len(lidarClipped), 1):
    clipped_df = gr.to_pandas(lidarClipped[crown])
    lidarROIA[crown] = clipped_df['value']

for crown in range(0, len(droneClipped), 1):
    clipped_df = gr.to_pandas(droneClipped[crown])
    droneROIA[crown] = clipped_df['value']
    
# %%
lidarROIAdrop = lidarROIA.sample(n = 97)
# %%
lidarROIAMelt = lidarROIAdrop.melt()
lidarROIAMelt['CHM'] = '1'

droneROIAMelt = droneROIA.melt()
droneROIAMelt['CHM'] = '2'

stacked = pd.concat([lidarROIAMelt, droneROIAMelt])
stacked.rename(columns = {'variable': 'Tree',
                              'value': 'Height'},
                   inplace=True)

# %%
lidarfreq = lidarROIAMelt.value_counts('variable')
dronefreq = droneROIAMelt.value_counts('variable')

# %%
A = stacked['CHM']
B = stacked['Tree']
Y = stacked['Height']
# %%
# The factor B is nested inside factor A.
# https://stackoverflow.com/questions/48273276/nested-anova-in-python-with-spm1d-cant-print-f-statistics-and-p-values

alpha        = 0.05
FF           = spm1d.stats.anova2nested(Y, A, B, equal_var=True)
FFi          = FF.inference(0.05)

p = FFi.get_p_values()
f = FFi.get_f_values()

# %%
plt.figure(figsize=(9,6))

sns.histplot(x='Height',
             data=stacked,
              bins=25,
              hue='CHM',
              kde=True)
# %%
sns.boxplot(x=stacked['CHM'],
            y=stacked['Height'],
            hue=None,
            orient="v")
plt.show()
# %%
print(FFi)