#!/usr/bin/env python
# coding: utf-8

# # EDA on suicide dataset
# 
# Suicide is a complex and devastating public health issue impacting individuals and communities worldwide. Understanding the factors associated with suicide risk is crucial for developing effective 
# 
# 
# link to dataset-->https://www.kaggle.com/datasets/russellyates88/suicide-rates-overview-1985-to-2016
# 
The notebooks folds in two steps 
1.Performing below operations
     *Indexing(multiindex ,reset index)
     *Handling missing values
     *Groupby Agg
     *Stack and Unstacking 
     *Melting 
2.Visulaization
# In[446]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')


# In[396]:


df=pd.read_csv("C:\\Users\\Pratibh\\Downloads\\suicide.csv")


# In[397]:


df


# In[398]:


df.head()


# In[399]:


df.info()


# In[400]:


df.describe()


# In[401]:


df.shape


# In[402]:


df[['sex','suicides_no']]


# In[403]:


# multiindex
mi = df.groupby(['sex','year']).agg({'suicides_no':'mean', 'age':'first'})
mi


# In[404]:


ri=mi.reset_index()
ri


# In[405]:


# handling missing data
df.isnull().sum()


# In[406]:


df['HDI for year']

Only HDI for year column has some null values and we will fill that by mean of the col 
# In[407]:


df['HDI for year']=df['HDI for year'].fillna(df['HDI for year'].mean())


# In[408]:


df['HDI for year']


# In[409]:


df.isnull().sum()

Apart from filling missing values by mean we can also use bfill method
# In[410]:


# groupby
gb=df.groupby(['sex','year'])
gb


# In[411]:


# agg
agg=gb.agg({'suicides_no':['mean','max'],'age':'first'})
agg


# In[412]:


# transform
transformed_df = df[['suicides_no', 'population']].transform(lambda x:x*2)
transformed_df


# In[413]:


# stackings
stacked = df.set_index(['sex', 'year'])[['age', 'suicides_no']].stack().reset_index().rename(columns={'level_2': 'level', 0: 'Value'})
stacked


# In[414]:


df.stack()


# In[415]:


df.set_index(['generation']).stack()


# In[416]:


df.set_index(['generation']).stack().reset_index()


# In[417]:


# Unstacking
stacked.unstack()


# In[418]:


# Melting
melted_df = df.melt(id_vars=['country', 'year', 'sex', 'age', 'suicides_no'],  
                    var_name='variable', 
                    value_name='value')
melted_df


# In[419]:


# pivot
pivoted_df = melted_df.pivot_table(index=['country', 'year', 'sex', 'age', 'suicides_no'], 
                                   columns='variable', 
                                   values='value')
pivoted_df


# # Visualization

# In[420]:


df_10 = df.groupby('country')['suicides_no'].sum().nlargest(10).reset_index(name='total_suicides')
df_10


# In[421]:


# barplot
plt.figure(figsize=(10, 6))
sns.barplot(x='total_suicides', y='country', data=df_10, palette='viridis')
plt.xlabel('Total Suicides')
plt.ylabel('Country')
plt.title('Top 10 Countries with Highest Number of Suicides')
plt.show()

The above bar graph shows top 10 countries with highest no of suicide over the years
# In[422]:


# pie chart
explode = [0.1, 0.1,0.1,0.1,0.1,0.1]
plt.figure(figsize=(8, 8))
df['generation'].value_counts().plot(kind='pie',autopct='%1.1f%%',explode=explode,shadow=True)
plt.title('Generation with their suicide  no')
plt.ylabel('') 
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))
plt.show()

The pie chart givs an idea about the which generation has commited how much suicide
# In[423]:


# hist plot
plt.figure(figsize=(8, 6))
plt.title("Suicide Relationship with Age")
sns.histplot(x='age', data=df, weights='suicides_no', bins=6)
plt.xlabel('Age Group')
plt.ylabel('Number of Suicides')
plt.show()

The above hist plot shows age group and their suicides committed by that particular age group
# In[424]:


# doghnut
plt.figure(figsize=(8, 8))
df['generation'].value_counts().plot(kind='pie',autopct='%1.1f%%',wedgeprops={'width': 0.2})
plt.title('Generation with their suicide  no')
plt.ylabel('') 
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))
plt.show()


# In[425]:


# Heat map
# Convert categorical variables 'generation' and 'sex' into numerical values
df_encoded = df.copy()
df_encoded['generation'] = pd.Categorical(df_encoded['generation']).codes
df_encoded['sex'] = pd.Categorical(df_encoded['sex']).codes

columns = ['suicides_no', 'population', 'generation', 'sex']

df_c = df_encoded[columns]

correlation_matrix = df_c.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
plt.show()

From the heat map we can infer following things
*There is a positive correlation between the number of suicides and population size.
*Correlations between population size and generation/sex may also exist.
*Generation and sex may exhibit correlations, indicating differences in suicide rates among different generations and sexes.
# In[426]:


# box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='generation', y='suicides/100k pop', data=df,showfliers=False)
plt.title('Box Plot of Suicides per 100k Population by Generation')
plt.xlabel('Generation')
plt.ylabel('Suicides per 100k Population')
plt.show()

The box plot allows shows us the distribution of suicides per 100k population within each generation.
# In[427]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='age', y='gdp_per_capita ($)', data=df,showfliers=False)
plt.title('Box Plot of GDP per Capita by Age Group')
plt.xlabel('Age Group')
plt.ylabel('GDP per Capita ($)')
plt.show()


# In[428]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='sex', y='suicides_no', data=df, showfliers=False)
plt.title('Suicides Number by Sex')
plt.xlabel('Sex')
plt.ylabel('Suicides Number')
plt.legend(labels=['Male', 'Female'])
plt.show()

The above graph shows males have commited more suicide than female
# In[429]:


# joint plot
sns.jointplot(x='gdp_per_capita ($)', y='suicides/100k pop', data=df, kind='scatter')
plt.xlabel('GDP per Capita ($)')
plt.ylabel('Suicides per 100k Population')
plt.show()

the graph can give us insigys such as higher GDP per capita associated with lower suicide rates,
# In[430]:


# distplot
sns.distplot(df['suicides/100k pop'],kde=False, color ='red')
plt.grid()

the above graph  represents the distribution of the 'suicides/100k pop' variable.
# In[431]:


sns.distplot(df['gdp_per_capita ($)'], kde=True, color='blue')
plt.title('Distribution of GDP per Capita')
plt.xlabel('GDP per Capita ($)')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# In[432]:


# pointplot
plt.figure(figsize=(8, 4))
sns.pointplot(x='age', y='suicides_no', data=df, hue='sex')
plt.title('Suicides Number by Age Group and Sex')
plt.xlabel('Age Group')
plt.ylabel('Suicides Number')
plt.legend()
plt.show()

Suicide number for different age group for male and female
# In[433]:


plt.figure(figsize=(10, 6))
sns.pointplot(x='generation', y='gdp_per_capita ($)', data=df, hue='sex')
plt.title('GDP per Capita by Generation and Sex')
plt.xlabel('Generation')
plt.ylabel('GDP per Capita ($)')
plt.legend()
plt.show()

The above graph shows us variation of gdp for each generation
# In[434]:


sns.lmplot(x='population', y='gdp_per_capita ($)', hue='age', data=df, palette='Set1')
plt.xlabel('Population')
plt.ylabel('GDP per Capita ($)')
plt.title('Linear Regression Plot of Population vs GDP per Capita by Age Group')
plt.show()


# The above tell us following things
# *The main trend line represents the average relationship between population and GDP per capita across all age groups. It shows whether there's a general tendency for GDP per capita to increase or decrease with population size.
# *The different colors represent distinct age groups. By observing how the data points cluster around their respective trend lines, we can discern whether the relationship between population and GDP per capita varies significantly among different age groups.

# In[435]:


# cat plot
sns.catplot(data=df,x='sex',y='suicides_no')
plt.legend(['Male','Female'])


# In[436]:


sns.catplot(data=df,x='sex',y='suicides_no',jitter=False)
plt.legend(['Male','Female'])


# In[437]:


sns.catplot(x='age', y='suicides/100k pop', data=df, kind='boxen',linewidth=1)
plt.xlabel('Age')
plt.ylabel('Suicides per 100k Population')
plt.title('Suicides per 100k Population by Age Group')
plt.show()

Showing distribution of suicides /100k and age group
# In[438]:


# Facetgrid
g = sns.FacetGrid(df, col="age", row="sex", margin_titles=True, height=3)
g.map(sns.histplot, "suicides_no", kde=True, bins=20, color="skyblue")
g.set_axis_labels("Number of Suicides", "Density")
plt.show()

we can observe the distribution of suicides within each age group and sex category. This allows us to se
# In[439]:


# scatter plot
age_mapping = {'5-14 years': 1, '15-24 years': 2, '25-34 years': 3, '35-54 years': 4, '55-74 years': 5, '75+ years': 6}
df['age_numeric'] = df['age'].map(age_mapping)
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['gdp_per_capita ($)'], df['suicides/100k pop'], c=df['age_numeric'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Age Group')
plt.xlabel('GDP per Capita ($)')
plt.ylabel('Suicides per 100k Population')
plt.title('Suicides per 100k Population vs. GDP per Capita by Age Group')
plt.grid()
plt.show()

The scatter plot visualizes the relationship between the GDP per capita and the number of suicides per 100k population,
# In[440]:


# clustermap
columns_to_cluster = ['suicides_no', 'population', 'suicides/100k pop', 'gdp_per_capita ($)']
subset_df = df[columns_to_cluster]
plt.figure(figsize=(12, 8))
sns.clustermap(subset_df, cmap='coolwarm', standard_scale=1)  # Standardize the data
plt.title('Cluster Map of Suicide Statistics')
plt.show()

from above we can identify groups of countries that exhibit comparable levels of suicides, population sizes, suicide rates per 100k population, and GDP per capita.
# # Plotly

# In[441]:


fig = px.scatter(df, x="gdp_per_capita ($)", y="suicides_no", color="country",
                 labels={"gdp_per_capita ($)": "GDP per Capita ($)", "suicides_no": "Number of Suicides"},
                 title="Suicides vs. GDP per Capita by Country")
fig.update_traces(marker=dict(size=8))
fig.update_layout(
    xaxis_title="GDP per Capita ($)",
    yaxis_title="Number of Suicides",
    showlegend=False
)
fig.show()


# In[442]:


fig = px.box(df, x="age", y="suicides_no", color="sex",
             labels={"age": "Age Group", "suicides_no": "Number of Suicides", "sex": "Sex"},
             title="Suicides by Age Group and Sex")
fig.update_layout(
    xaxis_title="Age Group",
    yaxis_title="Number of Suicides"
)
fig.show()


# In[ ]:




