# %%
import pandas as pd
import numpy as np
import statistics as stat

import seaborn as sns
import matplotlib.pyplot as plt
# %%
accValues = pd.read_csv(r'C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\Table5_ModelPerf\Table5_OverallModelPerf.csv')
#accValues.loc['OA', 'F1 Weighted', 'MCC', 'Kappa', 'OA RGB delta', 'OA CHM delta', 'OA GLCM delta', 'OA VI delta', 'OA GEOM delta'] = accValues.loc['OA', 'F1 Weighted', 'MCC', 'Kappa', 'OA RGB delta', 'OA CHM delta', 'OA GLCM delta', 'OA VI delta', 'OA GEOM delta'].apply(lambda x: x*100)
#accValues.loc[~accValues.columns.isin(['Features', 'RGBvCHM', 'RGBvGLCM'])] = accValues.loc[~accValues.columns.isin(['Features', 'RGBvCHM', 'RGBvGLCM'])].apply(lambda x: x*100)
# %%
accValues.head()
#['OA', 'F1 Weighted', 'MCC', 'Kappa', 'OA RGB delta', 'OA CHM delta', 'OA GLCM delta', 'OA VI delta', 'OA GEOM delta']

# %%
accValues.sort_values(axis='index', by='Lan-RGB-diff', inplace=True)
plt.figure(figsize=(13,6))
sns.set_theme(style="ticks")
OADeltaRGB = sns.barplot(x = accValues['Features'], y = accValues['Lan-RGB-diff'], palette='mako', orient='v')
OADeltaRGB.set(xlabel='Model', ylabel='Difference (%)')
loc, labels = plt.xticks()
OADeltaRGB.set_xticklabels(accValues['Features'], rotation=45, ha='right')
sns.despine()
OADeltaRGBFig = OADeltaRGB.get_figure()
OADeltaRGBFig.savefig(r"C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\Table5_ModelPerf\Lan-RGB-Diff.pdf", bbox_inches='tight', dpi=300)
# %%
accValues.sort_values(axis='index', by='Lan-CHM-diff', inplace=True)
plt.figure(figsize=(13,6))
sns.set_theme(style="ticks")
OADeltaCHM = sns.barplot(x = accValues['Features'], y = accValues['Lan-CHM-diff'], palette='mako', orient='v')
OADeltaCHM.set(xlabel='Model', ylabel='Difference (%)')
loc, labels = plt.xticks()
OADeltaCHM.set_xticklabels(accValues['Features'], rotation=45, ha='right')
sns.despine()
OADeltaCHMFig = OADeltaCHM.get_figure()
OADeltaCHMFig.savefig(r"C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\Table5_ModelPerf\Lan-CHM-Diff.pdf", bbox_inches='tight', dpi=300)
# %%
accValues.sort_values(axis='index', by='Pine-CHM-diff', inplace=True)
plt.figure(figsize=(13,6))
sns.set_theme(style="ticks")
OADeltaGLCM = sns.barplot(x = accValues['Features'], y = accValues['Pine-CHM-diff'], palette='mako', orient='v')
OADeltaGLCM.set(xlabel='Model', ylabel='Difference (%)')
loc, labels = plt.xticks()
OADeltaGLCM.set_xticklabels(accValues['Features'], rotation=45, ha='right')
sns.despine()
OADeltaGLCMFig = OADeltaGLCM.get_figure()
OADeltaGLCMFig.savefig(r"C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\Table5_ModelPerf\Pine-CHM-Diff.pdf", bbox_inches='tight', dpi=300)
# %%
accValues.sort_values(axis='index', by='Pine-RGB-diff', inplace=True)
plt.figure(figsize=(13,6))
sns.set_theme(style="ticks")
OADeltaVI = sns.barplot(x = accValues['Features'], y = accValues['Pine-RGB-diff'], palette='mako', orient='v')
OADeltaVI.set(xlabel='Model', ylabel='Difference (%)')
loc, labels = plt.xticks()
OADeltaVI.set_xticklabels(accValues['Features'], rotation=45, ha='right')
sns.despine()
OADeltaVIFig = OADeltaVI.get_figure()
OADeltaVIFig.savefig(r"C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\Table5_ModelPerf\Pine-RGB-Diff.pdf", bbox_inches='tight', dpi=300)
# %%
AccumFeat = ['RGB-CHM', 'RGB-CHM-TEX', 'RGB-CHM-TEX-VI', 'RGB-CHM-TEX-VI-GEOM']
subset = accValues[accValues["Features"].isin(AccumFeat)]
#y = y['OA'].apply(lambda x: x*100)
plt.figure(figsize=(13,6))
sns.set_theme(style="ticks")
OADeltaAccum = sns.barplot(x=subset['Features'], y=subset['RGB-OA-diff'], order=AccumFeat, color='#3182bd', orient='v')
OADeltaAccum.set(xlabel='Model', ylabel='ΔOA (%)')
loc, labels = plt.xticks()
OADeltaAccum.set_xticklabels(AccumFeat, rotation=45, ha='right')
sns.despine()
OADeltaAccumFig = OADeltaAccum.get_figure()
OADeltaAccumFig.savefig(r"C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\Table5_ModelPerf\OA-Accum-DeltaSymb.pdf", bbox_inches='tight', dpi=300)
# %%
AccumFeat = ['RGB-CHM', 'RGB-CHM-TEX', 'RGB-CHM-TEX-VI', 'RGB-CHM-TEX-VI-GEOM']
subset = accValues[accValues["Features"].isin(AccumFeat)]
plt.figure(figsize=(13,6))
sns.set_theme(style="ticks")
OADeltaAccum = sns.barplot(x=subset['Features'], y=subset['RGB-F1-diff'], order=AccumFeat, color='#3182bd', orient='v')
OADeltaAccum.set(xlabel='Model', ylabel='ΔF1-score')
loc, labels = plt.xticks()
OADeltaAccum.set_xticklabels(AccumFeat, rotation=45, ha='right')
sns.despine()
OADeltaAccumFig = OADeltaAccum.get_figure()
OADeltaAccumFig.savefig(r"C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\Table5_ModelPerf\F1-Accum-DeltaSymb.pdf", bbox_inches='tight', dpi=300)

# %%
x = accValues['RGBvCHM']
x.dropna(inplace=True)
y = accValues[accValues.RGBvCHM.isin(x)]
y = y['OA round']
plt.figure(figsize=(13,6))
sns.set_theme(style="ticks")
OARGBvCHM = sns.barplot(x, y, palette='colorblind', order=['RGB', 'CHM', 'RGB-CHM'], capsize=0.025, orient='v')
OARGBvCHM.set(xlabel='Feature groups', ylabel='OA (%)')
loc, labels = plt.xticks()
sns.despine()
OARGBvCHMFig = OARGBvCHM.get_figure()
OARGBvCHMFig.savefig(r"C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\Table5_ModelPerf\OA-RGBvCHM-Delta.pdf", bbox_inches='tight', dpi=300)

# %%
x = accValues['RGBvTEX']
x.dropna(inplace=True)
y = accValues[accValues.RGBvTEX.isin(x)]
y = y['OA round']
plt.figure(figsize=(13,6))
sns.set_theme(style="ticks")
OARGBvGLCM = sns.barplot(x, y, palette='colorblind', order=['RGB', 'TEX', 'RGB-TEX'], capsize=0.025, orient='v')
OARGBvGLCM.set(xlabel='Feature groups', ylabel='OA (%)')
loc, labels = plt.xticks()
sns.despine()
OARGBvGLCMFig = OARGBvGLCM.get_figure()
OARGBvGLCMFig.savefig(r"C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\Table5_ModelPerf\OA-RGBvTEX-Delta.pdf", bbox_inches='tight', dpi=300)

# %%
x = accValues['RGBvCHM']
x.dropna(inplace=True)
y = accValues[accValues.RGBvCHM.isin(x)]
y = y['F1 round']
plt.figure(figsize=(13,6))
sns.set_theme(style="ticks")
OARGBvCHM = sns.barplot(x, y, palette='colorblind', order=['RGB', 'CHM', 'RGB-CHM'], capsize=0.025, orient='v')
OARGBvCHM.set(xlabel='Feature groups', ylabel='F1-score (weighted)')
loc, labels = plt.xticks()
sns.despine()
OARGBvCHMFig = OARGBvCHM.get_figure()
OARGBvCHMFig.savefig(r"C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\Table5_ModelPerf\F1-RGBvCHM-Delta.pdf", bbox_inches='tight', dpi=300)
# %%
x = accValues['RGBvTEX']
x.dropna(inplace=True)
y = accValues[accValues.RGBvTEX.isin(x)]
y = y['F1 round']
plt.figure(figsize=(13,6))
sns.set_theme(style="ticks")
OARGBvCHM = sns.barplot(x, y, palette='colorblind', order=['RGB', 'TEX', 'RGB-TEX'], capsize=0.025, orient='v')
OARGBvCHM.set(xlabel='Feature groups', ylabel='F1-score (weighted)')
loc, labels = plt.xticks()
sns.despine()
OARGBvCHMFig = OARGBvCHM.get_figure()
OARGBvCHMFig.savefig(r"C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\Table5_ModelPerf\F1-RGBvTEX-Delta.pdf", bbox_inches='tight', dpi=300)

# %%
accValues.sort_values(axis='index', by='OA round', inplace=True)
plt.figure(figsize=(13,6))
sns.set_theme(style="ticks")
OADeltaVI = sns.barplot(x = accValues['Features'], y = accValues['OA round'], palette='rocket', orient='v')
OADeltaVI.set(xlabel='Model', ylabel='OA (%)')
loc, labels = plt.xticks()
OADeltaVI.set_xticklabels(accValues['Features'], rotation=45, ha='right')
sns.despine()
OADeltaVIFig = OADeltaVI.get_figure()
OADeltaVIFig.savefig(r"C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\Table5_ModelPerf\OA-All.pdf", bbox_inches='tight', dpi=300)
# %%
accValues.sort_values(axis='index', by='F1 round', inplace=True)
plt.figure(figsize=(13,6))
sns.set_theme(style="ticks")
OADeltaVI = sns.barplot(x = accValues['Features'], y = accValues['F1 round'], palette='rocket', orient='v')
OADeltaVI.set(xlabel='Model', ylabel='F1-score')
loc, labels = plt.xticks()
OADeltaVI.set_xticklabels(accValues['Features'], rotation=45, ha='right')
sns.despine()
OADeltaVIFig = OADeltaVI.get_figure()
OADeltaVIFig.savefig(r"C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\Table5_ModelPerf\F1-All.pdf", bbox_inches='tight', dpi=300)
# %%
accValues.sort_values(axis='index', by='UA-Pine round', inplace=True)
plt.figure(figsize=(13,6))
sns.set_theme(style="ticks")
OADeltaVI = sns.boxplot(x=accValues['CHM'], y = accValues['UA-Pine round'], color='#3182bd', orient='v')
OADeltaVI.set(xlabel='Model', ylabel='PA (%)')
#loc, labels = plt.xticks()
#OADeltaVI.set_xticklabels(accValues['CHM'], rotation=45, ha='right')
sns.despine()
OADeltaVIFig = OADeltaVI.get_figure()
OADeltaVIFig.savefig(r"C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\Table5_ModelPerf\UA-Pine-CHM.pdf", bbox_inches='tight', dpi=300)
# %%
accValues.sort_values(axis='index', by='UA-Pine round', inplace=True)
plt.figure(figsize=(13,6))
sns.set_theme(style="ticks")
OADeltaVI = sns.barplot(x = accValues['Features'], y = accValues['UA-Pine round'], palette='rocket', orient='v')
OADeltaVI.set(xlabel='Model', ylabel='PA (%)')
loc, labels = plt.xticks()
OADeltaVI.set_xticklabels(accValues['Features'], rotation=45, ha='right')
sns.despine()
OADeltaVIFig = OADeltaVI.get_figure()
OADeltaVIFig.savefig(r"C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\Table5_ModelPerf\UA-Pine.pdf", bbox_inches='tight', dpi=300)
# %%
accValues.sort_values(axis='index', by='Num-feat', inplace=True)
plt.figure(figsize=(13,6))
sns.set_theme(style="ticks")
OADeltaRGB = sns.barplot(x = accValues['Features'], y = accValues['UA-Lan round'], palette='rocket', orient='v')
OADeltaRGB.set(xlabel='Model', ylabel='UA (%)')
loc, labels = plt.xticks()
OADeltaRGB.set_xticklabels(accValues['Features'], rotation=45, ha='right')
sns.despine()
OADeltaRGBFig = OADeltaRGB.get_figure()
OADeltaRGBFig.savefig(r"C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\Table5_ModelPerf\Lan-UA-NumFeat.pdf", bbox_inches='tight', dpi=300)

# %%
accValues.sort_values(axis='index', by='Num-feat', inplace=True)
plt.figure(figsize=(13,6))
sns.set_theme(style="ticks")
OADeltaRGB = sns.barplot(x = accValues['Features'], y = accValues['UA-Pine round'], palette='rocket', orient='v')
OADeltaRGB.set(xlabel='Model', ylabel='UA (%)')
loc, labels = plt.xticks()
OADeltaRGB.set_xticklabels(accValues['Features'], rotation=45, ha='right')
sns.despine()
OADeltaRGBFig = OADeltaRGB.get_figure()
OADeltaRGBFig.savefig(r"C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\Table5_ModelPerf\Pine-UA-NumFeat.pdf", bbox_inches='tight', dpi=300)

# %%
accValues.sort_values(axis='index', by='Num-feat', inplace=True)
plt.figure(figsize=(13,6))
sns.set_theme(style="ticks")
OADeltaRGB = sns.barplot(x = accValues['Features'], y = accValues['OA round'], palette='rocket', orient='v')
OADeltaRGB.set(xlabel='Model', ylabel='OA (%)')
loc, labels = plt.xticks()
OADeltaRGB.set_xticklabels(accValues['Features'], rotation=45, ha='right')
sns.despine()
OADeltaRGBFig = OADeltaRGB.get_figure()
OADeltaRGBFig.savefig(r"C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\Table5_ModelPerf\OA-NumFeat.pdf", bbox_inches='tight', dpi=300)

# %%
accValues.sort_values(axis='index', by='Num-feat', inplace=True)
plt.figure(figsize=(13,6))
sns.set_theme(style="ticks")
OADeltaRGB = sns.scatterplot(x = accValues['Num-feat'], y = accValues['OA round'], color='#3182bd')
#OADeltaRGB.set(xlabel='Model', ylabel='UA (%)')
loc, labels = plt.xticks()
OADeltaRGB.set_xticklabels(accValues['Num-feat'], rotation=45, ha='right')
sns.despine()
OADeltaRGBFig = OADeltaRGB.get_figure()
OADeltaRGBFig.savefig(r"C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\Table5_ModelPerf\OA-NumFeat-Scatter.pdf", bbox_inches='tight', dpi=300)

# %%
x = accValues['RGBvTEX']
x.dropna(inplace=True)
y = accValues[accValues.RGBvTEX.isin(x)]
y = y['UA-Lan round']
plt.figure(figsize=(13,6))
sns.set_theme(style="ticks")
OARGBvCHM = sns.barplot(x, y, palette='colorblind', order=['RGB', 'TEX', 'RGB-TEX'], capsize=0.025, orient='v')
OARGBvCHM.set(xlabel='Feature groups', ylabel='UA (%)')
loc, labels = plt.xticks()
sns.despine()
OARGBvCHMFig = OARGBvCHM.get_figure()
OARGBvCHMFig.savefig(r"C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\Table5_ModelPerf\UA-Lan-RGBvTEX-Delta.pdf", bbox_inches='tight', dpi=300)

# %%
x = accValues['RGBvCHM']
x.dropna(inplace=True)
y = accValues[accValues.RGBvCHM.isin(x)]
y = y['UA-Lan round']
plt.figure(figsize=(13,6))
sns.set_theme(style="ticks")
OARGBvCHM = sns.barplot(x, y, palette='colorblind', order=['RGB', 'CHM', 'RGB-CHM'], capsize=0.025, orient='v')
OARGBvCHM.set(xlabel='Feature groups', ylabel='UA (%)')
loc, labels = plt.xticks()
sns.despine()
OARGBvCHMFig = OARGBvCHM.get_figure()
OARGBvCHMFig.savefig(r"C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\Table5_ModelPerf\UA-Lan-RGBvCHM-Delta.pdf", bbox_inches='tight', dpi=300)

# %%
x = accValues['RGBvTEX']
x.dropna(inplace=True)
y = accValues[accValues.RGBvTEX.isin(x)]
y = y['UA-Pine round']
plt.figure(figsize=(13,6))
sns.set_theme(style="ticks")
OARGBvCHM = sns.barplot(x, y, palette='colorblind', order=['RGB', 'TEX', 'RGB-TEX'], capsize=0.025, orient='v')
OARGBvCHM.set(xlabel='Feature groups', ylabel='UA (%)')
loc, labels = plt.xticks()
sns.despine()
OARGBvCHMFig = OARGBvCHM.get_figure()
OARGBvCHMFig.savefig(r"C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\Table5_ModelPerf\UA-Pine-RGBvTEX-Delta.pdf", bbox_inches='tight', dpi=300)

# %%
x = accValues['RGBvCHM']
x.dropna(inplace=True)
y = accValues[accValues.RGBvCHM.isin(x)]
y = y['UA-Pine round']
plt.figure(figsize=(13,6))
sns.set_theme(style="ticks")
OARGBvCHM = sns.barplot(x, y, palette='colorblind', order=['RGB', 'CHM', 'RGB-CHM'], capsize=0.025, orient='v')
OARGBvCHM.set(xlabel='Feature groups', ylabel='UA (%)')
loc, labels = plt.xticks()
sns.despine()
OARGBvCHMFig = OARGBvCHM.get_figure()
OARGBvCHMFig.savefig(r"C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\Table5_ModelPerf\UA-Pine-RGBvCHM-Delta.pdf", bbox_inches='tight', dpi=300)

# %%
plt.figure(figsize=(13,6))
sns.set_theme(style="ticks")
OARGBvCHM = sns.plot(accValues['UA-Lan round'], palette='colorblind', order=['RGB', 'TEX', 'RGB-TEX'], capsize=0.025, orient='v')
OARGBvCHM.set(xlabel='Feature groups', ylabel='UA (%)')
loc, labels = plt.xticks()
sns.despine()
OARGBvCHMFig = OARGBvCHM.get_figure()
OARGBvCHMFig.savefig(r"C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\Table5_ModelPerf\UA-Pine-RGBvTEX-Delta.pdf", bbox_inches='tight', dpi=300)
 
# %%
g = sns.FacetGrid(accValues, col=['Lantana', 'Slash pine'], col_wrap=2, height=1, ylim=(0, 100))
g.map(sns.pointplot, "solutions", "score", order=[1, 2, 3], color=".3", ci=None)

# %%
objectFeatures = pd.read_csv('D:\Dropbox\Honours\Peter_Woodfordia\Data\ALLROI_FS_Merged.csv')

# %%
trees = objectFeatures.loc[objectFeatures['Genus'].isin(['Angophora', 'Corymbia', 'Eucalyptus', 'Lophostemon', 'Pinus', 'Syncarpia']), ['Mean_LidarCHM']]
trees['Type'] = 'Tree canopy'
lantana = objectFeatures.loc[objectFeatures['Genus'].isin(['Lantana']), ['Mean_LidarCHM']]
lantana['Type'] = 'Lantana'

stacked = pd.concat([trees, lantana])
stacked

# %%
# %%
# boxplot of monday CHM: Drone Vs Lidar
sns.set_theme(style="ticks")
heightsplot = sns.boxplot(x='Type',
                        y='Mean_LidarCHM',
                        data=stacked,
                        hue=None,
                        orient="v")
heightsplot.set(xlabel='Cover type', ylabel='Mean CHM')
sns.despine()
plt.show()
heightsplotfig = heightsplot.get_figure()
heightsplotfig.savefig(r"C:\Users\Study\OneDrive - University of the Sunshine Coast\Documents\Thesis\Manuscript\Figures\heightsplot.pdf", bbox_inches='tight', dpi=300)
# %%
stat.mean(lantana['Mean_LidarCHM'])
stat.stdev(lantana['Mean_LidarCHM'])

# %%
stat.mean(trees['Mean_LidarCHM'])
#stat.stdev(trees['Mean_LidarCHM'])
# %%
