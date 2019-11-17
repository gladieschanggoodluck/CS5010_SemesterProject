#!/usr/bin/env python
# coding: utf-8

# # FIFA visulization and statistical analysis 

# In[54]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import datetime
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
 
from math import pi
 


# In[55]:


df = pd.read_csv('FIFA_1112.csv', index_col=0)
df.head()


# # Participants  - England is the top one followed by Germany then Argentina 

# In[56]:


plt.figure(figsize=(15,32))
sns.countplot(y = df.Country,palette="Set2") #Plot all the nations on Y Axis


# # Top three national participating- England, Germany and Spain

# In[57]:


# To show Different nations participating in the FIFA 2019

df['Country'].value_counts().plot.bar(color = 'orange', figsize = (35, 15 ))
plt.title('Different Nations Participating in FIFA')
plt.xlabel('Name of The Country')
plt.ylabel('count')
plt.show()


# # Different position acquired by the players 

# In[58]:


plt.figure(figsize = (12, 8))
sns.set(style = 'dark', palette = 'colorblind', color_codes = True)
ax = sns.countplot('Position', data = df, color = 'orange')
ax.set_xlabel(xlabel = 'Different Positions in Football', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)
ax.set_title(label = 'Comparison of Positions and Players', fontsize = 20)
plt.show()


# # Different position group acquired by the players 

# In[59]:


plt.figure(figsize = (12, 8))
sns.set(style = 'dark', palette = 'colorblind', color_codes = True)
ax = sns.countplot('Position Group', data = df, color = 'orange')
ax.set_xlabel(xlabel = 'Different Positions in Football', fontsize = 16)
ax.set_ylabel(ylabel = 'Count of Players', fontsize = 16)
ax.set_title(label = 'Comparison of Positions and Players', fontsize = 20)
plt.show()


# # Players Height distribution -180cm 

# In[87]:


# Height of Players

plt.figure(figsize = (20, 8))
ax = sns.countplot(x = 'Height', data = df, palette = 'dark')
ax.set_title(label = 'Count of players on Basis of Height', fontsize = 20)
ax.set_xlabel(xlabel = 'Height in Foot per inch', fontsize = 16)
ax.set_ylabel(ylabel = 'Count', fontsize = 16)

plt.show()


# # Make correlation plot to see overall rating related to the features 

# In[61]:


# plotting a correlation heatmap

plt.rcParams['figure.figsize'] = (15,10)
sns.heatmap(df[['Overall Rating', 'Pace', 'Shooting', 'Dribbling', 'Defending', 'Physicality',
                    'Height', 'Base Stats', 'In Game Stats']].corr(), annot = True)

plt.title('Histogram of the Dataset', fontsize = 30)
plt.show()


# # Best players per each position with their country, club based on the overall score - Here are players names

# In[62]:


df.iloc[df.groupby(df['Position'])['Overall Rating'].idxmax()][['Position', 'Name', 'Club', 'Country']]


# # Best players per each position group with their country, club based on the overall score - Here are player's names 

# In[63]:


df.iloc[df.groupby(df['Position Group'])['Overall Rating'].idxmax()][['Position Group', 'Name', 'Club', 'Country']]


# # Top 10 Countries based on participants and compare their overal scores - which country has the highest overall rating? --- Spain

# In[64]:


# Top 10 countries with highest number of players to compare their overall scores

df['Country'].value_counts().head(10)


# # Lets check Overall Rating of TOP 10 participant countries 

# In[86]:


# Every Nations' Player and their Weights

some_countries = ('England', 'Germany', 'Spain', 'France', 'Argentina', 'Italy', 'Colombia', 'Japan')
df_countries = df.loc[df['Country'].isin(some_countries) & df['Overall Rating']]

plt.rcParams['figure.figsize'] = (12, 7)
ax = sns.violinplot(x = df_countries['Country'], y = df_countries['Overall Rating'], palette = 'colorblind')
ax.set_xlabel(xlabel = 'Countries', fontsize = 9)
ax.set_ylabel(ylabel = 'Overall Rating', fontsize = 9)
ax.set_title(label = 'Distribution of Overall rating of players from different countries', fontsize = 20)
plt.show()


# In[66]:


#Data sanity check- Need to do continent conversion 
df.isnull().sum()


# # This is statistical summary of correlation matrix 

# In[67]:


#Compute pairwise correlation of Dataframe's attributes
corr = df.corr()
corr


# # Use heatmap to check correlation strength 

# In[68]:


#Compute pairwise correlation of Dataframe's attributes based on position group
fig, (ax) = plt.subplots(1, 1, figsize=(10,6))

hm = sns.heatmap(corr, 
                 ax=ax,           # Axes in which to draw the plot, otherwise use the currently-active Axes.
                 cmap="coolwarm", # Color Map.
                 #square=True,    # If True, set the Axes aspect to “equal” so each cell will be square-shaped.
                 annot=True, 
                 fmt='.2f',       # String formatting code to use when adding annotations.
                 #annot_kws={"size": 14},
                 linewidths=.05)

fig.subplots_adjust(top=0.93)
fig.suptitle('Overall Rating Correlation Heatmap', 
              fontsize=14, 
              fontweight='bold')


# # Correlation based on position group = goal keeper

# In[69]:


ColumnNames = list(df.columns.values)
df_goa= df[df['Position Group'] == 'Goal Keeper']
C_Data_goa = pd.concat([df_goa[['Position Group','Overall Rating']],df_goa[ColumnNames[11:17]]],axis=1)
#Compute pairwise correlation of Dataframe's attributes
corr_goa = C_Data_goa.corr()
corr_goa


# In[70]:


#Compute pairwise correlation of Dataframe's attributes based on position group
fig, (ax) = plt.subplots(1, 1, figsize=(10,6))

hm = sns.heatmap(corr_goa, 
                 ax=ax,           # Axes in which to draw the plot, otherwise use the currently-active Axes.
                 cmap="coolwarm", # Color Map.
                 #square=True,    # If True, set the Axes aspect to “equal” so each cell will be square-shaped.
                 annot=True, 
                 fmt='.2f',       # String formatting code to use when adding annotations.
                 #annot_kws={"size": 14},
                 linewidths=.05)

fig.subplots_adjust(top=0.93)
fig.suptitle('Overall Rating Correlation Heatmap for Goal Keeper', 
              fontsize=14, 
              fontweight='bold')


# # Correlation based on position group = Midfieder

# In[71]:


df_mid= df[df['Position Group'] == 'Midfieder']
C_Data_mid = pd.concat([df_mid[['Position Group','Overall Rating']],df_mid[ColumnNames[11:17]]],axis=1)
#Compute pairwise correlation of Dataframe's attributes
corr_mid = C_Data_mid.corr()
corr_mid


# In[72]:


#Compute pairwise correlation of Dataframe's attributes based on position group
fig, (ax) = plt.subplots(1, 1, figsize=(10,6))

hm = sns.heatmap(corr_mid, 
                 ax=ax,           # Axes in which to draw the plot, otherwise use the currently-active Axes.
                 cmap="coolwarm", # Color Map.
                 #square=True,    # If True, set the Axes aspect to “equal” so each cell will be square-shaped.
                 annot=True, 
                 fmt='.2f',       # String formatting code to use when adding annotations.
                 #annot_kws={"size": 14},
                 linewidths=.05)

fig.subplots_adjust(top=0.93)
fig.suptitle('Overall Rating Correlation Heatmap for Midfieder', 
              fontsize=14, 
              fontweight='bold')


# # Correlation based on position group = Defender

# In[73]:


df_def= df[df['Position Group'] == 'Defender']
C_Data_def = pd.concat([df_def[['Position Group','Overall Rating']],df_def[ColumnNames[11:17]]],axis=1)
#Compute pairwise correlation of Dataframe's attributes
corr_def = C_Data_def.corr()
corr_def


# In[74]:


#Compute pairwise correlation of Dataframe's attributes based on position group
fig, (ax) = plt.subplots(1, 1, figsize=(10,6))

hm = sns.heatmap(corr_def, 
                 ax=ax,           # Axes in which to draw the plot, otherwise use the currently-active Axes.
                 cmap="coolwarm", # Color Map.
                 #square=True,    # If True, set the Axes aspect to “equal” so each cell will be square-shaped.
                 annot=True, 
                 fmt='.2f',       # String formatting code to use when adding annotations.
                 #annot_kws={"size": 14},
                 linewidths=.05)

fig.subplots_adjust(top=0.93)
fig.suptitle('Overall Rating Correlation Heatmap for Goal Keeper', 
              fontsize=14, 
              fontweight='bold')


# # Correlation based on position group  = Midfieder, Goal Keeper, Defender, Attacker

# In[75]:


ColumnNames = list(df.columns.values)
C_Data = pd.concat([df[['Position Group','Overall Rating']],df[ColumnNames[11:17]]],axis=1)
HeatmapData = C_Data.groupby('Position Group').mean()
sns.heatmap(HeatmapData,cmap='Oranges',xticklabels = True,yticklabels = True)


# In[78]:


labels = np.array(HeatmapData.columns.values)
N = len(labels)

Position = 'Attacker'
stats=HeatmapData.loc[Position,labels]

angles = [n / float(N) * 2 * pi for n in range(N)]

stats=np.concatenate((stats,[stats[0]]))
angles=np.concatenate((angles,[angles[0]]))


fig=plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, stats, 'p-', linewidth=1)
ax.fill(angles, stats, alpha=0.5)
ax.set_thetagrids(angles * 180/np.pi, labels)
ax.set_title(Position)
ax.grid(True)


# # Midfieder correlation strength to each feature 

# In[79]:


labels = np.array(HeatmapData.columns.values)
N = len(labels)

Position = 'Midfieder'
stats=HeatmapData.loc[Position,labels]

angles = [n / float(N) * 2 * pi for n in range(N)]

stats=np.concatenate((stats,[stats[0]]))
angles=np.concatenate((angles,[angles[0]]))


fig=plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, stats, 'o-', linewidth=1)
ax.fill(angles, stats, alpha=0.5)
ax.set_thetagrids(angles * 180/np.pi, labels)
ax.set_title(Position)
ax.grid(True)


# # Goal Keeper  correlation strength to each feature 

# In[80]:


labels = np.array(HeatmapData.columns.values)
N = len(labels)

Position = 'Goal Keeper'
stats=HeatmapData.loc[Position,labels]

angles = [n / float(N) * 2 * pi for n in range(N)]

stats=np.concatenate((stats,[stats[0]]))
angles=np.concatenate((angles,[angles[0]]))


fig=plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, stats, 'o-', linewidth=1)
ax.fill(angles, stats, alpha=0.5)
ax.set_thetagrids(angles * 180/np.pi, labels)
ax.set_title(Position)
ax.grid(True)


# # Defender correlation strength to each feature 

# In[81]:


labels = np.array(HeatmapData.columns.values)
N = len(labels)

Position = 'Defender'
stats=HeatmapData.loc[Position,labels]

angles = [n / float(N) * 2 * pi for n in range(N)]

stats=np.concatenate((stats,[stats[0]]))
angles=np.concatenate((angles,[angles[0]]))


fig=plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, stats, 'o-', linewidth=1)
ax.fill(angles, stats, alpha=0.5)
ax.set_thetagrids(angles * 180/np.pi, labels)
ax.set_title(Position)
ax.grid(True)


# # pair plot to see all correlations 

# In[82]:


g = sns.pairplot(C_Data, hue="Position Group")


# # Players top 4 features based on position
# For example CAM: pace, dribbling, passing, Shooting

# In[85]:


# defining the features of players

player_features = ('Pace', 'Shooting', 'Passing', 
                   'Dribbling', 'Defending',  
                    'Physicality', 
)

# Top five features for every position in football

for i, val in df.groupby(df['Position'])[player_features].mean().iterrows():
    print('Position {}: {}, {}, {}, {}'.format(i, *tuple(val.nlargest(4).index)))


# # Position group top 4 features
# Attacker - pace, dribbling, shooting, nad physicality

# In[84]:


# defining the features of players

player_features = ('Pace', 'Shooting', 'Passing', 
                   'Dribbling', 'Defending', 'Physicality', 
)

# Top five features for every position in football

for i, val in df.groupby(df['Position Group'])[player_features].mean().iterrows():
    print('Position {}: {}, {}, {}, {}'.format(i, *tuple(val.nlargest(4).index)))


# In[ ]:




