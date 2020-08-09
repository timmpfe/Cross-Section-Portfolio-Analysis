#!/usr/bin/env python
# coding: utf-8

# # Bachelorarbeit Timm Pfeil

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smp
import matplotlib.pyplot as plt
import statsmodels.regression.linear_model as rg
import statsmodels.tools.tools as ct
import statsmodels.stats.diagnostic as dg
import scipy

from scipy import stats
from statsmodels import regression
from functools import reduce

import warnings
warnings.filterwarnings('ignore')


# ### Load the DataSets

# In[ ]:


CRSP = pd.read_csv('C:/Users/timmp/Desktop/Bachelorarbeit_F/CRSP_Monthly_Stock.csv', delimiter = ',')


# In[ ]:


display('There are', len(CRSP.groupby('PERMNO')), 'different shares in this Dataset')


# In[ ]:


display('The sum of missing values per column:', CRSP.isna().sum())


# In[ ]:


display('The whole Dataset contains:', len(CRSP), 'rows of Data')


# - as one can see there are many cells with missing values
#
# #### Things that are important in this Dataset:
#
# - only the Sharecodes 10 & 11 are of interest
# - the Price multiplied with the Shares Outstanding is the Market Capitalization in Thousands because the Shares Outstanding are listed in Thousands
# - the company name is for the analysis not of interest

# In[ ]:


del CRSP['COMNAM']


# ### Clean up the record

# - since the Dataset is to big for my PC to handle in one, I need to do the clean up in chunks

# In[ ]:


CRSP1 = CRSP[:500000]
CRSP2 = CRSP[500000:1000000]
CRSP3 = CRSP[1000000:1500000]
CRSP4 = CRSP[1500000:2000000]
CRSP5 = CRSP[2000000:2500000]
CRSP6 = CRSP[2500000:3000000]
CRSP7 = CRSP[3000000:3500000]
CRSP8 = CRSP[3500000:]


# In[ ]:


CRSP1 = CRSP1[CRSP1['RET'].notna()]
CRSP2 = CRSP2[CRSP2['RET'].notna()]
CRSP3 = CRSP3[CRSP3['RET'].notna()]
CRSP4 = CRSP4[CRSP4['RET'].notna()]
CRSP5 = CRSP5[CRSP5['RET'].notna()]
CRSP6 = CRSP6[CRSP6['RET'].notna()]
CRSP7 = CRSP7[CRSP7['RET'].notna()]
CRSP8 = CRSP8[CRSP8['RET'].notna()]


# In[ ]:


CRSP1 = CRSP1[(CRSP1.SHRCD >= 10) & (CRSP1.SHRCD <= 11)]
CRSP2 = CRSP2[(CRSP2.SHRCD >= 10) & (CRSP2.SHRCD <= 11)]
CRSP3 = CRSP3[(CRSP3.SHRCD >= 10) & (CRSP3.SHRCD <= 11)]
CRSP4 = CRSP4[(CRSP4.SHRCD >= 10) & (CRSP4.SHRCD <= 11)]
CRSP5 = CRSP5[(CRSP5.SHRCD >= 10) & (CRSP5.SHRCD <= 11)]
CRSP6 = CRSP6[(CRSP6.SHRCD >= 10) & (CRSP6.SHRCD <= 11)]
CRSP7 = CRSP7[(CRSP7.SHRCD >= 10) & (CRSP7.SHRCD <= 11)]
CRSP8 = CRSP8[(CRSP8.SHRCD >= 10) & (CRSP8.SHRCD <= 11)]


# - as mentioned before only the Sharecodes 10 & 11 are of interest

# - some Price values are negative (the price in this cells is the difference between the bid and the ask) they will be multiplied by -1 to get a positive price

# In[ ]:


CRSP1.loc[CRSP1.PRC < 0, 'PRC'] *= (-1)
CRSP2.loc[CRSP2.PRC < 0, 'PRC'] *= (-1)
CRSP3.loc[CRSP3.PRC < 0, 'PRC'] *= (-1)
CRSP4.loc[CRSP4.PRC < 0, 'PRC'] *= (-1)
CRSP5.loc[CRSP5.PRC < 0, 'PRC'] *= (-1)
CRSP6.loc[CRSP6.PRC < 0, 'PRC'] *= (-1)
CRSP7.loc[CRSP7.PRC < 0, 'PRC'] *= (-1)
CRSP8.loc[CRSP8.PRC < 0, 'PRC'] *= (-1)


# In[ ]:


CRSP = pd.concat([CRSP1, CRSP2, CRSP3, CRSP4, CRSP5, CRSP6, CRSP7, CRSP8], ignore_index=True)


# In[ ]:


CRSP = CRSP[CRSP.RET != 'C']


# In[ ]:


CRSP = CRSP[CRSP.RET != 'B']


# In[ ]:


CRSP['RET'] = CRSP['RET'].astype('float32')
CRSP['PRC'] = CRSP['PRC'].astype('float32')
CRSP['PERMNO'] = CRSP['PERMNO'].astype('int32')


# - the record contains now only the Sharecodes 10&11, therefore the column SHRCD is not of interest anymore

# In[ ]:


del CRSP['SHRCD']


# In[ ]:


display('The sums of missing values per column:', CRSP.isna().sum())


# - the clean up was successful

# In[ ]:


display('There are', len(CRSP.groupby('PERMNO')), 'different shares in this Dataset')


# In[ ]:


display('The whole Dataset contains:', len(CRSP), 'rows of Data')


# - Due to the clean up ~26% of the whole dataset were lost

# - due to the clean up 8452 Permno are lost

# In[ ]:


CRSP['date'] = CRSP['date'].astype(str)

CRSP['date'] = CRSP['date'].str[0:4]+'-'+CRSP['date'].str[4:6]+'-'+CRSP['date'].str[6:]


# In[ ]:


CRSP.to_csv('CRSP_only_10&11.csv', sep=',')


# In[ ]:


CRSP['date'] = CRSP['date'].astype(str)

CRSP['year'] = CRSP['date'].str[0:4]
CRSP['month'] = CRSP['date'].str[4:6]

CRSP['month'] = CRSP['month'].astype('int32')

CRSP['year'] = CRSP['year'].astype('int32')


# - since the portfolios will be bought at the end of June, the holding Period will be from the first Juli till the last trading day in June
# - The return is calculated from the beginning of the month till the end of the month, therefore the start return of Juli is our fist return for each portfolio and the return of June is our last return for the portfolio

# In[ ]:


chunk1 = CRSP[CRSP.month < 7]

chunk2 = CRSP[CRSP.month > 6]

chunk1['year'] = chunk1['year']-1

CRSP = pd.concat([chunk1, chunk2], ignore_index=True)


# #### Calculate the Market Capitalisation

# - The Market Capitalisation will be calculated at the end of every year.
# - The price at the end of June will be the price and the SHROUT at the end of June will be the Shares Outstanding in thousands

# In[ ]:


CRSP['Mkt_Cap_in_Mio'] = (CRSP['SHROUT']*CRSP['PRC'])/1000


# ## Calulate the excess returns of each share

# In[ ]:


RF = pd.read_csv('C:/Users/timmp/Desktop/Bachelorarbeit_F/F-F_Research_Data_5_Factors_2x3.csv', delimiter=',')


# In[ ]:


CRSP['t'] = CRSP['date'].astype(str)
CRSP['t'] = CRSP['t'].str[:6]


# In[ ]:


CRSP['t'] = CRSP['t'].astype('int32')


# In[ ]:


CRSP['RET'] = CRSP['RET']*100


# In[ ]:


RF['t'] = RF['t'].astype('int32')


# In[ ]:


final_df = CRSP.merge(RF, on=['t'])


# In[ ]:


final_df['ExRET'] = final_df['RET'] - final_df['RF']


# ## Calculate the Market Capitalization for each share

# In[ ]:


final_df['Mkt_Cap_in_Mio'] = (final_df['SHROUT']*final_df['PRC'])/1000


# In[ ]:


final_df['PERMNO'] = final_df['PERMNO'].astype('int32')
final_df['year'] = final_df['year'].astype('int32')
final_df['Mkt-RF'] = final_df['Mkt-RF'].astype('float32')
final_df['RF'] = final_df['RF'].astype('float32')


# In[ ]:


del final_df['t']
del final_df['RET']
del final_df['PRC']
del final_df['SHROUT']


# In[ ]:


final_df.to_csv('Data_incl_ExRET.csv', sep=',', index=False)


# # Beta Calculation with the CRSP data

# In[ ]:


# Formula to calculate the Beta of each stock
def getbeta (group):
    group['Beta'] = group['ExRET'].rolling(60, min_periods=36).cov(group['Mkt-RF']) / group['Mkt-RF'].rolling(60,  min_periods=36).var()
    return group


# In[ ]:


# Load the Dataset
CRSP = pd.read_csv('Data_incl_ExRET.csv', delimiter=',')


# In[ ]:


CRSP['ExRET'] = CRSP['ExRET'] /100
CRSP['Mkt-RF'] = CRSP['Mkt-RF'] /100
CRSP['SMB'] = CRSP['SMB'] /100
CRSP['HML'] = CRSP['HML'] /100
CRSP['RMW'] = CRSP['RMW'] /100
CRSP['CMA'] = CRSP['CMA'] /100
CRSP['RF'] = CRSP['RF'] /100

# Splitting the Dataset to avoid memory errors
CRSP1 = CRSP[CRSP.PERMNO < 50000]
CRSP2 = CRSP[CRSP.PERMNO >= 50000]


# In[ ]:


# Run the function
data1 = CRSP1.groupby('PERMNO').apply(getbeta)

data2 = CRSP2.groupby('PERMNO').apply(getbeta)


# In[ ]:


# Save the Data with the new column in two new files, again to avoid memory errors
data1.to_csv('CRSP1.csv', sep=',', index=False)

data2.to_csv('CRSP2.csv', sep=',', index=False)


# ## Calculate the cumulative return of each stock for the holding periode

# In[ ]:


# Formula to calculate the cumulative return of the holding period
# The group variable represents one Stock for the holding period - Beginning of July until the end of June the following year
def getCumRet (group):
    group['CumRET'] = np.exp(np.log(group['ExRET']+1).rolling(len(group)).sum())-1
    return group


# In[ ]:


# Load the Datafiles
data1 = pd.read_csv('CRSP1.csv', delimiter=',')


# In[ ]:


data2 = pd.read_csv('CRSP2.csv', delimiter=',')


# In[ ]:


# Dataset 2 needs to be split, due to a memory error
data3 = data2[data2.PERMNO < 70000]
data4 = data2[data2.PERMNO >= 70000]


# In[ ]:


df1 = data1.groupby(['PERMNO','year']).apply(getCumRet)


# In[ ]:


df2 = data3.groupby(['PERMNO','year']).apply(getCumRet)


# In[ ]:


df3 = data4.groupby(['PERMNO','year']).apply(getCumRet)


# In[ ]:


# Concat the datasets
df2 = pd.concat([df2, df3], ignore_index=True)

# Save to files because of memory errors
df1.to_csv('CRSP1.csv', sep=',', index=False)
df2.to_csv('CRSP2.csv', sep=',', index=False)

#df2.to_csv('chunk1.csv', sep=',', index=False)
#df3.to_csv('chunk2.csv', sep=',', index=False)

#df1 = pd.read_csv('chunk1.csv', delimiter=',')
#df2 = pd.read_csv('chunk2.csv', delimiter=',')


# In[ ]:


# Load Datasets
df1 = pd.read_csv('CRSP1.csv', delimiter=',')
df2 = pd.read_csv('CRSP2.csv', delimiter=',')

# Concat the datasets
df = pd.concat([df1, df2], ignore_index=True)

# Save to the final Dataset
df.to_csv('CRSP_DATA_INCL_STOCK_BETA.csv', sep=',', index=False)


# # Breakpoint Analyse

# In[ ]:


FinR = pd.read_csv('C:/Users/timmp/Desktop/Bachelorarbeit_F/Financial_Ratios_Firm_Level.csv', delimiter = ',')


# In[ ]:


display('There are', len(FinR.groupby('permno')), 'different Stocks in this Dataset')


# In[ ]:


display('The sums of missing values per column:', FinR.isna().sum())


# #### The rows with missing values in the column pe_exi will be deleted

# In[ ]:


FinR = FinR[FinR['pe_exi'].notna()]


# In[ ]:


display('There are', len(FinR.groupby('permno')), 'different Stocks in this Dataset')


# In[ ]:


display('The sums of missing values per column are now:', FinR.isna().sum())


# - The difference between the Dataframe with missing values and the Dataframe after dropping them is 1193 Shares

# In[ ]:


FinR['adate'] = FinR['adate'].astype(str)

FinR['fiscal_month'] = FinR['adate'].str[4:6]

FinR['fiscal_year'] = FinR['adate'].str[0:4]


# In[ ]:


c = FinR.groupby('fiscal_month')['permno'].count().to_frame()
c = c.reset_index()
c = c[1:]
c.rename(columns = {'fiscal_month': 'Monat', 'permno': 'Stammaktien'}, inplace = True)
c.set_index('Monat', inplace=True)


# In[ ]:


c.T.to_csv('Aktien_JA_pro_Monat.csv', sep=',')


# ### Filtering the Data for the values that are needed
# - The fiscal year end should be in December
# - The calculated PE-Ratio should be at the end of March, since the companies with a fiscal year end in december have three months time to publish the results of the fiscal year

# In[ ]:


FinR['adate'] = FinR['adate'].astype(str)

FinR['qdate'] = FinR['qdate'].astype(str)

FinR['public_date'] = FinR['public_date'].astype(str)

FinR['fiscal_year'] = FinR['adate'].str[0:4]

FinR['fiscal_month'] = FinR['adate'].str[4:6]

FinR['q_month'] = FinR['qdate'].str[4:6]

FinR['p_month'] = FinR['public_date'].str[4:6]


# - as one can see the fiscal_months differ per share, that is because not all companies end their fiscal year at the end of december
# - most of the companies do end their fiscal year at the end of december, that is why I will focus on those companies

# In[ ]:


FinR = FinR[FinR.fiscal_month == '12']


# - since we are only intereste in the fiscal year end in december we filter the Data for the specific month

# In[ ]:


display('There are', len(FinR.groupby('permno')), 'different Stocks in this Dataset')


# - the number above indicates, that the majority of the companies ends their fiscal year at the end of december

# - as expected the leads to a major drop in the different permno in this Dataset
# - there are now 6761 Permno less in this set | or only 66.58% of the Permnos left

# In[ ]:


FinR = FinR[FinR.p_month == '03']


# - a company who ends their fiscal year at the end of december has to publish their balance sheet until the end of march
# - to be consistent with the calculation of the PE-Ratio we take the PE-Ratio which is calculated at the end of march

# In[ ]:


FinR = FinR[['permno','fiscal_year', 'fiscal_month', 'public_date', 'bm', 'pe_exi']]


# ## Get the Breakpoints for the PE-sorted Portfolios incl. negative PE-Ratios

# In[ ]:


def get_percentile(group):
    #Sort the groups(years) into percentiles
    qs, bins = pd.qcut(group['pe_exi'], [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1] , retbins=True)
    group['x_percentile'] = 1 + pd.qcut(group['pe_exi'], q=10, labels=False)

    return pd.concat([pd.Series(bins, index=['P_10', 'P_20', 'P_30', 'P_40', 'P_50', 'P_60', 'P_70', 'P_80', 'P_90', 'P_100'])])

percentile_df = FinR.groupby(['fiscal_year']).apply(get_percentile)
percentile_df.head()


# In[ ]:


percentile_df.to_csv('NEG_Breakpoints.csv', sep=',')


# #### Which Permno is in which Percentile

# In[ ]:


def get_perc(group):
    #Sort the groups(years) into percentiles
    qs, bins = pd.qcut(group['pe_exi'], [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1] , retbins=True)
    group['percentile'] = 1 + pd.qcut(group['pe_exi'], q=10, labels=False)

    return group

perc_df = FinR.groupby(['fiscal_year']).apply(get_perc)
perc_df.set_index('fiscal_year', inplace=True)
perc_df.head(2)


# In[ ]:


perc_df.to_csv('NEG_Permno_per_percentile.csv', sep=',')


# ## Get the Breakpoints for the PE-sorted Portfolios | only positive PE-Ratios

# - maybe there is a difference, when the focus is only on the positive PE-Ratios

# In[ ]:


FinR_pos = FinR[FinR.pe_exi > 0]


# In[ ]:


display('There are', len(FinR_pos.groupby('permno')), 'different Stocks in this Dataset')


# - as expected some data is lost

# In[ ]:


def get_percentile(group):
    #Sort the groups(years) into quantiles
    qs, bins = pd.qcut(group['pe_exi'], [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1] , retbins=True)
    group['x_percentile'] = 1 + pd.qcut(group['pe_exi'], q=10, labels=False)

    return pd.concat([pd.Series(bins, index=['P_10', 'P_20', 'P_30', 'P_40', 'P_50', 'P_60', 'P_70', 'P_80', 'P_90', 'P_100'])])

percentile_df2 = FinR_pos.groupby(['fiscal_year']).apply(get_percentile)
percentile_df2.head()


# In[ ]:


percentile_df2.to_csv('POS_Breakpoints.csv', sep=',', index=False)


# In[ ]:


def get_perc(group):
    #Sort the groups(years) into percentiles
    qs, bins = pd.qcut(group['pe_exi'], [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1] , retbins=True)
    group['percentile'] = 1 + pd.qcut(group['pe_exi'], q=10, labels=False)

    return group

perc_df2 = FinR_pos.groupby(['fiscal_year']).apply(get_perc)
perc_df2.set_index('fiscal_year', inplace=True)
perc_df2.head()


# In[ ]:


perc_df2.to_csv('POS_Permno_per_percentile.csv', sep=',', index=False)


# # Combine the Breakpoints with the CRSP Data

# In[ ]:


Breakp_neg = pd.read_csv('C:/Users/timmp/Desktop/Bachelorarbeit_F/NEG_Permno_per_percentile.csv', delimiter = ',')


# In[ ]:


CRSP = pd.read_csv('C:/Users/timmp/Desktop/Bachelorarbeit_F/CRSP_DATA_INCL_STOCK_BETA.csv', delimiter = ',')


# In[ ]:


column_names = {'permno':'PERMNO', 'fiscal_year':'year'}
Breakp_neg.rename(columns = column_names, inplace=True)


# In[ ]:


del Breakp_neg['public_date']
del Breakp_neg['fiscal_month']


# In[ ]:


final_df_neg = CRSP.merge(Breakp_neg, on=['PERMNO', 'year'])


# In[ ]:


final_df_neg['Size'] = np.log(final_df_neg['Mkt_Cap_in_Mio'])

final_df_neg['EP'] = 1 / final_df_neg['pe_exi']


# In[ ]:


final_df_neg.to_csv('NEG_final_df.csv', sep=',', index=False)


# In[ ]:


final_df_neg = final_df_neg[final_df_neg.year == 1986]


# In[ ]:


final_df_neg = final_df_neg[final_df_neg['Beta'].notna()]


# In[ ]:


final_df_neg.groupby('month')['Beta'].count()


# ### Combine the Breakpoints with the CRSP Data only positive PE-Ratios

# In[ ]:


Breakp_pos = pd.read_csv('C:/Users/timmp/Desktop/Bachelorarbeit_F/POS_Permno_per_percentile.csv', delimiter = ',')


# In[ ]:


CRSP = pd.read_csv('C:/Users/timmp/Desktop/Bachelorarbeit_F/CRSP_DATA_INCL_STOCK_BETA.csv', delimiter = ',')


# In[ ]:


column_names = {'permno':'PERMNO', 'fiscal_year':'year'}
Breakp_pos.rename(columns = column_names, inplace=True)


# In[ ]:


del Breakp_pos['public_date']
del Breakp_pos['fiscal_month']


# In[ ]:


final_df_pos = CRSP.merge(Breakp_pos, on=['PERMNO', 'year'])


# In[ ]:


final_df_pos['Size'] = np.log(final_df_pos['Mkt_Cap_in_Mio'])

final_df_pos['EP'] = 1 / final_df_pos['pe_exi']


# In[ ]:


final_df_pos.to_csv('POS_final_df.csv', sep=',', index=False)


# # Plot the CRSP Data

# In[ ]:


CRSP = pd.read_csv('C:/Users/timmp/Desktop/Bachelorarbeit_F/CRSP_Monthly_Stock.csv', delimiter = ',')

CRSP['date'] = CRSP['date'].astype(str)

CRSP['date'] = CRSP['date'].str[0:4]+'-'+CRSP['date'].str[4:6]+'-'+CRSP['date'].str[6:]

CRSP.index = pd.to_datetime(CRSP['date'])


# In[ ]:


CRSP1 = pd.read_csv('CRSP_only_10&11.csv', delimiter=',')

CRSP1.index = pd.to_datetime(CRSP1['date'])

Securities = CRSP.groupby(CRSP.index)['PERMNO'].count()

Stocks = CRSP1.groupby(CRSP1.index)['PERMNO'].count()


# In[ ]:


plt.figure(figsize=(9,6))
plt.gca().set_facecolor('whitesmoke')
plt.grid(True)
plt.plot(Securities, label='Anzahl der CRSP Wertpapiere ('+ r'$\bar{x}$' + ': 6707)')
plt.plot(Stocks, label='Anzahl der Stammaktien ('+ r'$\bar{x}$' + ': 4948)')
plt.legend(loc='best')
plt.ylabel('Anzahl der Wertpapiere', fontsize=14)
plt.xlabel('Jahr', fontsize=14)
plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.savefig('graph.png', dpi=600)
plt.show()


# In[ ]:


display(CRSP1.groupby(CRSP1.index)['PERMNO'].count().mean().round())
display(CRSP.groupby(CRSP.index)['PERMNO'].count().mean().round())


# # Plot the Data with PE-Ratios and fiscal year end in december

# In[2]:


df = pd.read_csv('NEG_final_df.csv', delimiter=',')

df2 = pd.read_csv('POS_final_df.csv', delimiter=',')


# In[3]:


df['date'] = df['date'].astype(str)
df['date'] = df['date'].str[:4] + '-' + df['date'].str[4:6] + '-' + df['date'].str[6:]
df.index = pd.to_datetime(df['date'])

df2['date'] = df2['date'].astype(str)
df2['date'] = df2['date'].str[:4] + '-' + df2['date'].str[4:6] + '-' + df2['date'].str[6:]
df2.index = pd.to_datetime(df2['date'])


# In[4]:


Incl_neg = df.groupby(df.index)['PERMNO'].count()

Only_pos = df2.groupby( df2.index)['PERMNO'].count()


# In[5]:


display('Average number of shares for all shares with fiscal year end in december:', Incl_neg.mean().round())
display('Average number of shares for all shares with positive PE-Ratio:', Only_pos.mean().round())


# In[6]:


plt.figure(figsize=(9,6))
plt.gca().set_facecolor('whitesmoke')
plt.grid(True)
plt.plot(Incl_neg, label='Aktien mit Jahresabschluss im Dezember ('+ r'$\bar{x}$' + ': 2591)')
plt.plot(Only_pos, label='Aktien mit Jahresabschluss im Dezember & positivem KGV ('+ r'$\bar{x}$' + ': 1847)')
plt.legend(loc='best')
plt.ylabel('Anzahl der Wertpapiere', fontsize=14)
plt.xlabel('Jahr', fontsize=14)
plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.savefig('graph2.png', dpi=600)
plt.show()


df_p = pd.read_csv('POS_final_df.csv', delimiter=',')


# In[3]:


df = df_p[df_p.month == 6]


# ## PE-Ratios

# In[6]:


a = df.groupby(['year', 'date'])['pe_exi'].describe(percentiles=[.05, .25, .75, .95]).shift(1)

b = df.groupby(['year', 'date'])['pe_exi'].skew().shift(1)

c = df.groupby(['year', 'date'])['pe_exi'].apply(pd.DataFrame.kurt).shift(1)

a = a.merge(b, on='year')

column_names = {'pe_exi': 'Skew'}
a.rename(columns = column_names, inplace=True)

a = a.merge(c, on='year')

column_names = {'pe_exi': 'Kurt'}
a.rename(columns = column_names, inplace=True)

a = a[['mean', 'std', 'Skew', 'Kurt', 'min', '5%', '25%', '50%', '75%', '95%', 'max', 'count']]

column_names = {'mean':'Mean', 'std':'SD', 'min':'Min', '50%':'Median', 'max':'Max', 'count':'n'}
a.rename(columns=column_names, inplace=True)

a

#a.to_csv('POS_Annual_Summary_Statistics_for_PE.csv', sep=',')


# ##  Mkt_Cap

# - The Market Capitalisation will be calculated at the end of every year.
# - The price at the end of June will be the price and the SHROUT at the end of June will be the Shares Outstanding in thousands

# In[ ]:


a = df.groupby(['year', 'date'])['Mkt_Cap_in_Mio'].describe(percentiles=[.05, .25, .75, .95]).shift(1)

b = df.groupby(['year', 'date'])['Mkt_Cap_in_Mio'].skew().shift(1)

c = df.groupby(['year', 'date'])['Mkt_Cap_in_Mio'].apply(pd.DataFrame.kurt).shift(1)

a = a.merge(b, on='year')

column_names = {'Mkt_Cap_in_Mio': 'Skew'}
a.rename(columns = column_names, inplace=True)

a = a.merge(c, on='year')

column_names = {'Mkt_Cap_in_Mio': 'Kurt'}
a.rename(columns = column_names, inplace=True)

a = a[['mean', 'std', 'Skew', 'Kurt', 'min', '5%', '25%', '50%', '75%', '95%', 'max', 'count']]

column_names = {'mean':'Mean', 'std':'SD', 'min':'Min', '50%':'Median', 'max':'Max', 'count':'n'}
a.rename(columns=column_names, inplace=True)

a.to_csv('POS_Annual_Summary_Statistics_for_Mkt_Cap.csv', sep=',')


# ## Book-to-Market

# In[ ]:


a = df.groupby(['year', 'date'])['bm'].describe(percentiles=[.05, .25, .75, .95]).shift(1)

b = df.groupby(['year', 'date'])['bm'].skew().shift(1)

c = df.groupby(['year', 'date'])['bm'].apply(pd.DataFrame.kurt).shift(1)

a = a.merge(b, on='year')

column_names = {'bm': 'Skew'}
a.rename(columns = column_names, inplace=True)

a = a.merge(c, on='year')

column_names = {'bm': 'Kurt'}
a.rename(columns = column_names, inplace=True)

a = a[['mean', 'std', 'Skew', 'Kurt', 'min', '5%', '25%', '50%', '75%', '95%', 'max', 'count']]

column_names = {'mean':'Mean', 'std':'SD', 'min':'Min', '50%':'Median', 'max':'Max', 'count':'n'}
a.rename(columns=column_names, inplace=True)

a.to_csv('POS_Annual_Summary_Statistics_for_BM.csv', sep=',')


# ## Size

# In[ ]:


a = df.groupby(['year', 'date'])['Size'].describe(percentiles=[.05, .25, .75, .95]).shift(1)

b = df.groupby(['year', 'date'])['Size'].skew().shift(1)

c = df.groupby(['year', 'date'])['Size'].apply(pd.DataFrame.kurt).shift(1)

a = a.merge(b, on='year')

column_names = {'Size': 'Skew'}
a.rename(columns = column_names, inplace=True)

a = a.merge(c, on='year')

column_names = {'Size': 'Kurt'}
a.rename(columns = column_names, inplace=True)

a = a[['mean', 'std', 'Skew', 'Kurt', 'min', '5%', '25%', '50%', '75%', '95%', 'max', 'count']]

column_names = {'mean':'Mean', 'std':'SD', 'min':'Min', '50%':'Median', 'max':'Max', 'count':'n'}
a.rename(columns=column_names, inplace=True)

a.to_csv('POS_Annual_Summary_Statistics_for_Size.csv', sep=',')


# # Beta
#
# Due to the calculation of beta the month for the comparision is July not end of June

# In[ ]:


df = df_p[df_p.month == 7]


# In[ ]:


a = df.groupby(['year', 'date'])['Beta'].describe(percentiles=[.05, .25, .75, .95])

b = df.groupby(['year', 'date'])['Beta'].skew()

c = df.groupby(['year', 'date'])['Beta'].apply(pd.DataFrame.kurt)

a = a.merge(b, on='year')

column_names = {'Beta': 'Skew'}
a.rename(columns = column_names, inplace=True)

a = a.merge(c, on='year')

column_names = {'Beta': 'Kurt'}
a.rename(columns = column_names, inplace=True)

a = a[['mean', 'std', 'Skew', 'Kurt', 'min', '5%', '25%', '50%', '75%', '95%', 'max', 'count']]

column_names = {'mean':'Mean', 'std':'SD', 'min':'Min', '50%':'Median', 'max':'Max', 'count':'n'}
a.rename(columns=column_names, inplace=True)

a.to_csv('POS_Annual_Summary_Statistics_for_Beta.csv', sep=',')


# -------

# # Information about the Portfolios

# In[ ]:


df = pd.read_csv('POS_final_df.csv', delimiter=',')

df1 = df[df.month == 6]

df2 = df[df.month == 7]

df3 = df[df['CumRET'].notna()]

df1 = df1[['year', 'PERMNO', 'Mkt_Cap_in_Mio', 'Size', 'bm', 'pe_exi']]

df2 = df2[['year', 'PERMNO', 'Beta']]

df3 = df3[['year', 'PERMNO', 'percentile', 'CumRET']]

df1['year'] = df1['year'] + 1

df3 = df3[df3.year > 1969]

df1 = df1[df1.year < 2019]

final_df = df3.merge(df1, how='left', on=['year', 'PERMNO'])

final_df = final_df.merge(df2, how='left', on=['year', 'PERMNO'])


# In[ ]:


a = final_df.groupby(['percentile', 'year'])['CumRET'].mean().to_frame()
b = final_df.groupby(['percentile', 'year'])['Mkt_Cap_in_Mio'].mean().to_frame()
c = final_df.groupby(['percentile', 'year'])['Size'].mean().to_frame()
d = final_df.groupby(['percentile', 'year'])['bm'].mean().to_frame()
e = final_df.groupby(['percentile', 'year'])['Beta'].mean().to_frame()

table1 = pd.pivot_table(a, columns='percentile', index='year', values='CumRET')
table2 = pd.pivot_table(b, columns='percentile', index='year', values='Mkt_Cap_in_Mio')
table3 = pd.pivot_table(c, columns='percentile', index='year', values='Size')
table4 = pd.pivot_table(d, columns='percentile', index='year', values='bm')
table5 = pd.pivot_table(e, columns='percentile', index='year', values='Beta')


# In[ ]:


table2.to_csv('POS_MktCap_in_Mio_per_port.csv', sep=',')
table3.to_csv('POS_Size_per_port.csv', sep=',')
table4.to_csv('POS_BM_per_port.csv', sep=',')
table5.to_csv('POS_Beta_per_port.csv', sep=',')


# #  $ r_{t+1} $

# In[ ]:


df = pd.read_csv('POS_final_df.csv', delimiter=',')

df2 = df[df['CumRET'].notna()]

df2['CumRET'] = df2['CumRET']*100


# ### Univariate Portfolio Returns
#
# The portfolios are listed one below the other and not side by side, so I used Excel to sort them side by side. I must admit that this is not the most elegant way.

# In[ ]:


a = df2.groupby(['percentile', 'year'])['CumRET'].mean().to_frame()

table = pd.pivot_table(a, columns='percentile', index='year', values='CumRET')
table = table[table.index > 1969]
table['10-1'] = table[10] - table[1]


# In[ ]:


table.mean()


# In[ ]:


table.to_csv('POS_Annual_rt+1_NotWin.csv', sep=',')


# ### Summary Statistic

# In[ ]:


a = df2.groupby(['year'])['CumRET'].describe(percentiles=[.05, .25, .75, .95])

b = df2.groupby(['year'])['CumRET'].skew()

c = df2.groupby(['year'])['CumRET'].apply(pd.DataFrame.kurt)

a = a.merge(b, on='year')

column_names = {'CumRET': 'Skew'}
a.rename(columns = column_names, inplace=True)

a = a.merge(c, on='year')

column_names = {'CumRET': 'Kurt'}
a.rename(columns = column_names, inplace=True)

a = a[['mean', 'std', 'Skew', 'Kurt', 'min', '5%', '25%', '50%', '75%', '95%', 'max', 'count']]

column_names = {'mean':'Mean', 'std':'SD', 'min':'Min', '50%':'Median', 'max':'Max', 'count':'n'}
a.rename(columns=column_names, inplace=True)


# In[ ]:


a.to_csv('POS_Annual_Summary_Statistics_for_ExRET_NotWin.csv', sep=',')


# ## Winsorize the returns to reduce the outliers

# In[ ]:


df2['CumRET'] = stats.mstats.winsorize(df2['CumRET'], limits=0.005)


# In[ ]:


df2['date'] = df2['date'].astype(str)

df2['date'] = df2['date'].str[:4] + '-' + df2['date'].str[4:6] + '-' + df2['date'].str[6:]

df2.index = pd.to_datetime(df2['date'])


# ### Univariate Portfolio Returns

# In[ ]:


a = df2.groupby(['percentile', 'year'])['CumRET'].mean().to_frame()

table = pd.pivot_table(a, columns='percentile', index='year', values='CumRET')
table = table[table.index > 1969]

# Calculate the differnce portfolio
table['10-1'] = table[10] - table[1]


# In[ ]:


table.to_csv('POS_Annual_rt+1_Win.csv', sep=',')


# ### Annual Statistics Summary with winzorised rt+1

# In[ ]:


a = df2.groupby(['year'])['CumRET'].describe(percentiles=[.05, .25, .75, .95])

b = df2.groupby(['year'])['CumRET'].skew()

c = df2.groupby(['year'])['CumRET'].apply(pd.DataFrame.kurt)

a = a.merge(b, on='year')

column_names = {'CumRET': 'Skew'}
a.rename(columns = column_names, inplace=True)

a = a.merge(c, on='year')

column_names = {'CumRET': 'Kurt'}
a.rename(columns = column_names, inplace=True)

a = a[['mean', 'std', 'Skew', 'Kurt', 'min', '5%', '25%', '50%', '75%', '95%', 'max', 'count']]

column_names = {'mean':'Mean', 'std':'SD', 'min':'Min', '50%':'Median', 'max':'Max', 'count':'n'}
a.rename(columns=column_names, inplace=True)


# In[ ]:


a.to_csv('POS_Annual_Summary_Statistics_for_ExRET_Win.csv', sep=',')


# -----

# -----

# # Correlation

# In[ ]:


df = pd.read_csv('POS_final_df.csv', delimiter=',')

df1 = df[df.month == 6]

df2 = df[df.month == 7]

df3 = df[df['CumRET'].notna()]

df1 = df1[['year', 'PERMNO', 'Mkt_Cap_in_Mio', 'Size', 'bm', 'pe_exi']]

df2 = df2[['year', 'PERMNO', 'Beta']]

df3 = df3[['year', 'PERMNO', 'CumRET']]

df1['year'] = df1['year'] + 1

df3 = df3[df3.year > 1969]

df1 = df1[df1.year < 2019]

final_df = df3.merge(df1, how='left', on=['year', 'PERMNO'])

final_df = final_df.merge(df2, how='left', on=['year', 'PERMNO'])


# ### Pearson Correlation
# Pearson correlation quantifies the linear relationship between two variables. Pearson correlation coefficient can lie between -1 and +1, like other correlation measures.  A positive Pearson corelation mean that one variable’s value increases with the others. And a negative Pearson coefficient  means one variable decreases as other variable decreases.  Correlations coefficients of -1 or +1 mean the relationship is exactly linear.
#
# #### Winsorize the before calculating the Pearson Correlation
# - Winsorization is performed on a period-by-period basis using only entities for which validd values of both X and Y are available

# In[ ]:


# Drop the rows with missing values
df_Pearson = final_df.dropna(how='any')

df_Pearson = df_Pearson[['year', 'CumRET', 'Size', 'bm', 'pe_exi', 'Beta']]


# In[ ]:


def winsorize(group):
    group['Beta'] = stats.mstats.winsorize(group['Beta'], limits=[.005,.005])
    group['pe_exi'] = stats.mstats.winsorize(group['pe_exi'], limits=[.005,.005])
    group['Size'] = stats.mstats.winsorize(group['Size'], limits=[.005,.005])
    group['bm'] = stats.mstats.winsorize(group['bm'], limits=[.005,.005])
    group['CumRET'] = stats.mstats.winsorize(group['CumRET'], limits=[.005,.005])
    return group


# In[ ]:


df = df_Pearson.groupby('year').apply(winsorize)


# In[ ]:


PE_B = df.groupby('year')['pe_exi'].corr(df['Beta'], method='pearson')
PE_bm = df.groupby('year')['pe_exi'].corr(df['bm'], method='pearson')
PE_S = df.groupby('year')['pe_exi'].corr(df['Size'], method='pearson')
PE_r = df.groupby('year')['pe_exi'].corr(df['CumRET'], method='pearson')
B_bm = df.groupby('year')['Beta'].corr(df['bm'], method='pearson')
B_S = df.groupby('year')['Beta'].corr(df['Size'], method='pearson')
B_r = df.groupby('year')['Beta'].corr(df['CumRET'], method='pearson')
bm_S = df.groupby('year')['bm'].corr(df['Size'], method='pearson')
bm_r = df.groupby('year')['bm'].corr(df['CumRET'], method='pearson')
S_r = df.groupby('year')['Size'].corr(df['CumRET'], method='pearson')


# In[ ]:


Pearson = pd.concat([PE_B, PE_bm, PE_S, PE_r, B_bm, B_S, B_r, bm_S, bm_r, S_r], axis=1)

Pearson.columns = ['pt(PE, b)', 'pt(PE, BM)', 'pt(PE, Size)', 'pt(PE, rt+1)', 'pt(b, BM)', 'pt(b, Size)', 'pt(b, rt+1)', 'pt(BM, Size)', 'pt(BM, rt+1)', 'pt(Size, rt+1)']


# ### Spearman Rank Correlation
# Pearson correlation assumes that the data we are comparing is normally distributed. When that assumption is not true, the correlation value is reflecting the true association. Spearman correlation does not assume that data is from a specific distribution, so it is a non-parametric correlation measure. Spearman correlation is also known as Spearman’s rank correlation as it computes correlation coefficient on rank values of the data.

# In[ ]:


df = final_df

df = df[['year', 'CumRET', 'Size', 'bm', 'pe_exi', 'Beta']]


# In[ ]:


PE_B = df.groupby('year')['pe_exi'].corr(df['Beta'], method='spearman')
PE_bm = df.groupby('year')['pe_exi'].corr(df['bm'], method='spearman')
PE_S = df.groupby('year')['pe_exi'].corr(df['Size'], method='spearman')
PE_r = df.groupby('year')['pe_exi'].corr(df['CumRET'], method='spearman')
B_bm = df.groupby('year')['Beta'].corr(df['bm'], method='spearman')
B_S = df.groupby('year')['Beta'].corr(df['Size'], method='spearman')
B_r = df.groupby('year')['Beta'].corr(df['CumRET'], method='spearman')
bm_S = df.groupby('year')['bm'].corr(df['Size'], method='spearman')
bm_r = df.groupby('year')['bm'].corr(df['CumRET'], method='spearman')
S_r = df.groupby('year')['Size'].corr(df['CumRET'], method='spearman')


# In[ ]:


Spearman = pd.concat([PE_B, PE_bm, PE_S, PE_r, B_bm, B_S, B_r, bm_S, bm_r, S_r], axis=1)

Spearman.columns = ['pSt(PE, b)', 'pSt(PE, BM)', 'pSt(PE, Size)', 'pSt(PE, rt+1)', 'pSt(b, BM)', 'pSt(b, Size)', 'pSt(b, rt+1)', 'pSt(BM, Size)', 'pSt(BM, rt+1)', 'pSt(Size, rt+1)']


# ## Combine both datasets

# In[ ]:


df = Pearson.merge(Spearman, on='year')

df = df[['pt(PE, b)', 'pSt(PE, b)', 'pt(PE, BM)', 'pSt(PE, BM)', 'pt(PE, Size)', 'pSt(PE, Size)', 'pt(PE, rt+1)', 'pSt(PE, rt+1)',
         'pt(b, BM)', 'pSt(b, BM)', 'pt(b, Size)', 'pSt(b, Size)', 'pt(b, rt+1)', 'pSt(b, rt+1)', 'pt(BM, Size)', 'pSt(BM, Size)',
         'pt(BM, rt+1)', 'pSt(BM, rt+1)', 'pt(Size, rt+1)', 'pSt(Size, rt+1)']]


# In[ ]:


df.to_csv('POS_Corr.csv', sep=',')


# ### Time Series Average

# In[ ]:


a = df.mean()

df2 = a.to_frame().T


# In[ ]:


df2.to_csv('POS_Corr_Av.csv', sep=',', index=False)


# ------

# --------

# # Statistical tests for

# ### Test for heteroscedasticity

# In[ ]:


df2 = pd.read_csv('POS_final_df.csv', delimiter=',')

df2 = df2[df2['CumRET'].notna()]

df2 = df2[['year', 'CumRET', 'Mkt-RF']]

df2.loc[:, 'const'] = ct.add_constant(df2)
ivar = ['Mkt-RF', 'const']
reg = rg.OLS(df2['CumRET'], df2[ivar], hasconst=bool).fit()
res = reg.resid
display('Breusch-Pagan LM Test Statistic', np.round(dg.het_breuschpagan(res, exog_het=df2[ivar])[0],6))
display('Breusch-Pagan LM Test P-Value', np.round(dg.het_breuschpagan(res, exog_het=df2[ivar])[1],6))


# ## t statistic of mean

# In[ ]:


df = pd.read_csv('POS_Annual_rt+1_NotWin.csv', delimiter=',')

df = df.set_index('year')

column_names = {'1':'P1', '2':'P2', '3':'P3', '4':'P4', '5':'P5', '6':'P6', '7':'P7', '8':'P8', '9':'P9', '10':'P10', '10-1':'PDif'}
df.rename(columns=column_names, inplace=True)

#Creating a dummy variable
df['dummy'] = 0.0


# In[ ]:


# T_test without the Newey and West adjusted standard error
t_test = (df.mean()-0)/(df.std()/(len(df)-1)**0.5)
t_test


# In[ ]:


# Calculating the NW standard error
P1_Se = smp.ols('P1 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).bse[0]
P2_Se = smp.ols('P2 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).bse[0]
P3_Se = smp.ols('P3 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).bse[0]
P4_Se = smp.ols('P4 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).bse[0]
P5_Se = smp.ols('P5 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).bse[0]
P6_Se = smp.ols('P6 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).bse[0]
P7_Se = smp.ols('P7 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).bse[0]
P8_Se = smp.ols('P8 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).bse[0]
P9_Se = smp.ols('P9 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).bse[0]
P10_Se = smp.ols('P10 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).bse[0]
PDif_Se = smp.ols('PDif ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).bse[0]


# In[ ]:


# Calculating the T statistics
P1 = smp.ols('P1 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).tvalues[0]
P2 = smp.ols('P2 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).tvalues[0]
P3 = smp.ols('P3 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).tvalues[0]
P4 = smp.ols('P4 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).tvalues[0]
P5 = smp.ols('P5 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).tvalues[0]
P6 = smp.ols('P6 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).tvalues[0]
P7 = smp.ols('P7 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).tvalues[0]
P8 = smp.ols('P8 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).tvalues[0]
P9 = smp.ols('P9 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).tvalues[0]
P10 = smp.ols('P10 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).tvalues[0]
PDif = smp.ols('PDif ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).tvalues[0]


# In[ ]:


# Calculating the P Values
P1_p = smp.ols('P1 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).pvalues[0]
P2_p = smp.ols('P2 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).pvalues[0]
P3_p = smp.ols('P3 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).pvalues[0]
P4_p = smp.ols('P4 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).pvalues[0]
P5_p = smp.ols('P5 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).pvalues[0]
P6_p = smp.ols('P6 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).pvalues[0]
P7_p = smp.ols('P7 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).pvalues[0]
P8_p = smp.ols('P8 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).pvalues[0]
P9_p = smp.ols('P9 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).pvalues[0]
P10_p = smp.ols('P10 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).pvalues[0]
PDif_p = smp.ols('PDif ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).pvalues[0]


# In[ ]:


Av = df.mean()[:-1].to_frame()

NWSe = [P1_Se, P2_Se, P3_Se, P4_Se, P5_Se, P6_Se, P7_Se, P8_Se, P9_Se, P10_Se, PDif_Se]

T_values = [P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, PDif]

P_values = [P1_p, P2_p, P3_p, P4_p, P5_p, P6_p, P7_p, P8_p, P9_p, P10_p, PDif_p]


# In[ ]:


d = {'Standard error': NWSe, 't-statistic': T_values, 'p-value': P_values}
statistic = pd.DataFrame(data=d)


# In[ ]:


final = Av.merge(statistic, on=Av.index)

column_names = {'key_0':'Portfolio', 0:'Average'}
final.rename(columns=column_names, inplace=True)

final.set_index('Portfolio', inplace=True)

final = final.T

final = final.applymap('{:,.2f}'.format)

final.to_csv('POS_Return_Summary_NotWin.csv', sep=',')


# ## Same calculation with winsorized Data

# In[ ]:


df = pd.read_csv('POS_Annual_rt+1_Win.csv', delimiter=',')

df = df.set_index('year')

column_names = {'1':'P1', '2':'P2', '3':'P3', '4':'P4', '5':'P5', '6':'P6', '7':'P7', '8':'P8', '9':'P9', '10':'P10', '10-1':'PDif'}
df.rename(columns=column_names, inplace=True)

#Creating a dummy variable
df['dummy'] = 0.0


# In[ ]:


# T_test without the Newey and West adjusted standard error
t_test = (df.mean()-0)/(df.std()/(len(df)-1)**0.5)
t_test


# In[ ]:


# Calculating the NW standard error
P1_Se = smp.ols('P1 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).bse[0]
P2_Se = smp.ols('P2 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).bse[0]
P3_Se = smp.ols('P3 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).bse[0]
P4_Se = smp.ols('P4 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).bse[0]
P5_Se = smp.ols('P5 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).bse[0]
P6_Se = smp.ols('P6 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).bse[0]
P7_Se = smp.ols('P7 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).bse[0]
P8_Se = smp.ols('P8 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).bse[0]
P9_Se = smp.ols('P9 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).bse[0]
P10_Se = smp.ols('P10 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).bse[0]
PDif_Se = smp.ols('PDif ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).bse[0]


# In[ ]:


# Calculating the T statistics
P1 = smp.ols('P1 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).tvalues[0]
P2 = smp.ols('P2 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).tvalues[0]
P3 = smp.ols('P3 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).tvalues[0]
P4 = smp.ols('P4 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).tvalues[0]
P5 = smp.ols('P5 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).tvalues[0]
P6 = smp.ols('P6 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).tvalues[0]
P7 = smp.ols('P7 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).tvalues[0]
P8 = smp.ols('P8 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).tvalues[0]
P9 = smp.ols('P9 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).tvalues[0]
P10 = smp.ols('P10 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).tvalues[0]
PDif = smp.ols('PDif ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).tvalues[0]


# In[ ]:


# Calculating the P Values
P1_p = smp.ols('P1 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).pvalues[0]
P2_p = smp.ols('P2 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).pvalues[0]
P3_p = smp.ols('P3 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).pvalues[0]
P4_p = smp.ols('P4 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).pvalues[0]
P5_p = smp.ols('P5 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).pvalues[0]
P6_p = smp.ols('P6 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).pvalues[0]
P7_p = smp.ols('P7 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).pvalues[0]
P8_p = smp.ols('P8 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).pvalues[0]
P9_p = smp.ols('P9 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).pvalues[0]
P10_p = smp.ols('P10 ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).pvalues[0]
PDif_p = smp.ols('PDif ~ 1 + dummy',data=df).fit().get_robustcov_results(cov_type='HAC',maxlags=6).pvalues[0]


# In[ ]:


Av = df.mean()[:-1].to_frame()

NWSe = [P1_Se, P2_Se, P3_Se, P4_Se, P5_Se, P6_Se, P7_Se, P8_Se, P9_Se, P10_Se, PDif_Se]

T_values = [P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, PDif]

P_values = [P1_p, P2_p, P3_p, P4_p, P5_p, P6_p, P7_p, P8_p, P9_p, P10_p, PDif_p]


# In[ ]:


d = {'Standard error': NWSe, 't-statistic': T_values, 'p-value': P_values}
statistic = pd.DataFrame(data=d)


# In[ ]:


final = Av.merge(statistic, on=Av.index)

column_names = {'key_0':'Portfolio', 0:'Average'}
final.rename(columns=column_names, inplace=True)

final.set_index('Portfolio', inplace=True)

final = final.T

final = final.applymap('{:,.2f}'.format)

final.to_csv('POS_Return_Summary_Win.csv', sep=',')


# -----

# # Calculate the Risk adjusted Returns
#
# ## not winzorised data

# In[ ]:


Rf = pd.read_csv('F-F_Research_Data_5_Factors_2x3.csv', delimiter=',')

Rf['t'] = Rf['t'].astype(str)
Rf['year'] = Rf['t'].str[:4]
Rf['month'] = Rf['t'].str[4:6]

Rf['month'] = Rf['month'].astype('int32')
Rf['year'] = Rf['year'].astype('int32')

Rf1 = Rf[Rf.month < 7]
Rf2 = Rf[Rf.month > 6]

Rf1['year'] = Rf1['year']-1

Rf = pd.concat([Rf1, Rf2], ignore_index=True)

Rf[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']] = Rf[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']]/100

def getCumRet (group):
    group['Mkt_Rf'] = np.exp(np.log(group['Mkt-RF']+1).rolling(len(group)).sum())-1
    group['SMBc'] = np.exp(np.log(group['SMB']+1).rolling(len(group)).sum())-1
    group['HMLc'] = np.exp(np.log(group['HML']+1).rolling(len(group)).sum())-1
    group['RMWc'] = np.exp(np.log(group['RMW']+1).rolling(len(group)).sum())-1
    group['CMAc'] = np.exp(np.log(group['CMA']+1).rolling(len(group)).sum())-1
    group['RFc'] = np.exp(np.log(group['RF']+1).rolling(len(group)).sum())-1
    return group

Rf = Rf.groupby(['year']).apply(getCumRet)

Rf = Rf[Rf['Mkt_Rf'].notna()]

Rf[['Mkt_Rf', 'SMBc', 'HMLc', 'RMWc', 'CMAc', 'RFc']] = Rf[['Mkt_Rf', 'SMBc', 'HMLc', 'RMWc', 'CMAc', 'RFc']]*100


# In[ ]:


final_df = pd.read_csv('C:/Users/timmp/Desktop/Bachelorarbeit_F/POS_Annual_rt+1_NotWin.csv', delimiter=',')

final_df = final_df.merge(Rf, on='year')

final_df = final_df[['year', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '10-1', 'Mkt_Rf', 'SMBc',
       'HMLc', 'RMWc', 'CMAc', 'RFc']]

column_names = {'1':'P1', '2':'P2', '3':'P3', '4':'P4', '5':'P5', '6':'P6', '7':'P7', '8':'P8', '9':'P9', '10':'P10', '10-1':'PDif', 'Mkt_Rf':'MKT'}
final_df.rename(columns=column_names, inplace=True)

final_df.set_index('year', inplace=True)


# In[ ]:


CAPM_1 = smp.ols(formula = 'P1 ~ 1 + MKT', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF3_1 = smp.ols(formula = 'P1 ~ 1 + MKT + SMBc + HMLc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF5_1 = smp.ols(formula = 'P1 ~ 1 + MKT + SMBc + HMLc + RMWc + CMAc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
CAPM_2 = smp.ols(formula = 'P2 ~ 1 + MKT', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF3_2 = smp.ols(formula = 'P2 ~ 1 + MKT + SMBc + HMLc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF5_2 = smp.ols(formula = 'P2 ~ 1 + MKT + SMBc + HMLc + RMWc + CMAc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
CAPM_3 = smp.ols(formula = 'P3 ~ 1 + MKT', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF3_3 = smp.ols(formula = 'P3 ~ 1 + MKT + SMBc + HMLc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF5_3 = smp.ols(formula = 'P3 ~ 1 + MKT + SMBc + HMLc + RMWc + CMAc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
CAPM_4 = smp.ols(formula = 'P4 ~ 1 + MKT', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF3_4 = smp.ols(formula = 'P4 ~ 1 + MKT + SMBc + HMLc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF5_4 = smp.ols(formula = 'P4 ~ 1 + MKT + SMBc + HMLc + RMWc + CMAc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
CAPM_5 = smp.ols(formula = 'P5 ~ 1 + MKT', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF3_5 = smp.ols(formula = 'P5 ~ 1 + MKT + SMBc + HMLc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF5_5 = smp.ols(formula = 'P5 ~ 1 + MKT + SMBc + HMLc + RMWc + CMAc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
CAPM_6 = smp.ols(formula = 'P6 ~ 1 + MKT', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF3_6 = smp.ols(formula = 'P6 ~ 1 +  MKT + SMBc + HMLc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF5_6 = smp.ols(formula = 'P6 ~ 1 + MKT + SMBc + HMLc + RMWc + CMAc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
CAPM_7 = smp.ols(formula = 'P7 ~ 1 + MKT', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF3_7 = smp.ols(formula = 'P7 ~ 1 + MKT + SMBc + HMLc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF5_7 = smp.ols(formula = 'P7 ~ 1 + MKT + SMBc + HMLc + RMWc + CMAc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
CAPM_8 = smp.ols(formula = 'P8 ~ 1 + MKT', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF3_8 = smp.ols(formula = 'P8 ~ 1 + MKT + SMBc + HMLc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF5_8 = smp.ols(formula = 'P8 ~ 1 + MKT + SMBc + HMLc + RMWc + CMAc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
CAPM_9 = smp.ols(formula = 'P9 ~ 1 + MKT', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF3_9 = smp.ols(formula = 'P9 ~ 1 + MKT + SMBc + HMLc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF5_9 = smp.ols(formula = 'P9 ~ 1 + MKT + SMBc + HMLc + RMWc + CMAc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
CAPM_10 = smp.ols(formula = 'P10 ~ 1 + MKT', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF3_10 = smp.ols(formula = 'P10 ~ 1 + MKT + SMBc + HMLc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF5_10 = smp.ols(formula = 'P10 ~ 1 + MKT + SMBc + HMLc + RMWc + CMAc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
CAPM_11 = smp.ols(formula = 'PDif ~ 1 + MKT', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF3_11 = smp.ols(formula = 'PDif ~ 1 + MKT + SMBc + HMLc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF5_11 = smp.ols(formula = 'PDif ~ 1 + MKT + SMBc + HMLc + RMWc + CMAc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})


# In[ ]:


Cc1 = CAPM_1.params
F3c1 = FF3_1.params
F5c1 = FF5_1.params
Cc2 = CAPM_2.params
F3c2 = FF3_2.params
F5c2 = FF5_2.params
Cc3 = CAPM_3.params
F3c3 = FF3_3.params
F5c3 = FF5_3.params
Cc4 = CAPM_4.params
F3c4 = FF3_4.params
F5c4 = FF5_4.params
Cc5 = CAPM_5.params
F3c5 = FF3_5.params
F5c5 = FF5_5.params
Cc6 = CAPM_6.params
F3c6 = FF3_6.params
F5c6 = FF5_6.params
Cc7 = CAPM_7.params
F3c7 = FF3_7.params
F5c7 = FF5_7.params
Cc8 = CAPM_8.params
F3c8 = FF3_8.params
F5c8 = FF5_8.params
Cc9 = CAPM_9.params
F3c9 = FF3_9.params
F5c9 = FF5_9.params
Cc10 = CAPM_10.params
F3c10 = FF3_10.params
F5c10 = FF5_10.params
Cc11 = CAPM_11.params
F3c11 = FF3_11.params
F5c11 = FF5_11.params


# In[ ]:


Ct1 = CAPM_1.tvalues
F3t1 = FF3_1.tvalues
F5t1 = FF5_1.tvalues
Ct2 = CAPM_2.tvalues
F3t2 = FF3_2.tvalues
F5t2 = FF5_2.tvalues
Ct3 = CAPM_3.tvalues
F3t3 = FF3_3.tvalues
F5t3 = FF5_3.tvalues
Ct4 = CAPM_4.tvalues
F3t4 = FF3_4.tvalues
F5t4 = FF5_4.tvalues
Ct5 = CAPM_5.tvalues
F3t5 = FF3_5.tvalues
F5t5 = FF5_5.tvalues
Ct6 = CAPM_6.tvalues
F3t6 = FF3_6.tvalues
F5t6 = FF5_6.tvalues
Ct7 = CAPM_7.tvalues
F3t7 = FF3_7.tvalues
F5t7 = FF5_7.tvalues
Ct8 = CAPM_8.tvalues
F3t8 = FF3_8.tvalues
F5t8 = FF5_8.tvalues
Ct9 = CAPM_9.tvalues
F3t9 = FF3_9.tvalues
F5t9 = FF5_9.tvalues
Ct10 = CAPM_10.tvalues
F3t10 = FF3_10.tvalues
F5t10 = FF5_10.tvalues
Ct11 = CAPM_11.tvalues
F3t11 = FF3_11.tvalues
F5t11 = FF5_11.tvalues


# In[ ]:


Cp1 = CAPM_1.pvalues
F3p1 = FF3_1.pvalues
F5p1 = FF5_1.pvalues
Cp2 = CAPM_2.pvalues
F3p2 = FF3_2.pvalues
F5p2 = FF5_2.pvalues
Cp3 = CAPM_3.pvalues
F3p3 = FF3_3.pvalues
F5p3 = FF5_3.pvalues
Cp4 = CAPM_4.pvalues
F3p4 = FF3_4.pvalues
F5p4 = FF5_4.pvalues
Cp5 = CAPM_5.pvalues
F3p5 = FF3_5.pvalues
F5p5 = FF5_5.pvalues
Cp6 = CAPM_6.pvalues
F3p6 = FF3_6.pvalues
F5p6 = FF5_6.pvalues
Cp7 = CAPM_7.pvalues
F3p7 = FF3_7.pvalues
F5p7 = FF5_7.pvalues
Cp8 = CAPM_8.pvalues
F3p8 = FF3_8.pvalues
F5p8 = FF5_8.pvalues
Cp9 = CAPM_9.pvalues
F3p9 = FF3_9.pvalues
F5p9 = FF5_9.pvalues
Cp10 = CAPM_10.pvalues
F3p10 = FF3_10.pvalues
F5p10 = FF5_10.pvalues
Cp11 = CAPM_11.pvalues
F3p11 = FF3_11.pvalues
F5p11 = FF5_11.pvalues


# In[ ]:


val1=[Cc1[0], Ct1[0], Cp1[0], Cc1[1], Ct1[1], Cp1[1], F3c1[0], F3t1[0], F3p1[0], F3c1[1], F3t1[1], F3p1[1], F3c1[2],
        F3t1[2], F3p1[2], F3c1[3], F3t1[3], F3p1[3], F5c1[0], F5t1[0], F5p1[0], F5c1[1], F5t1[1], F5p1[1], F5c1[2],
        F5t1[2], F5p1[2], F5c1[3], F5t1[3], F5p1[3], F5c1[4], F5t1[4], F5p1[4], F5c1[5], F5t1[5], F5p1[5]]
val2=[Cc2[0], Ct2[0], Cp2[0], Cc2[1], Ct2[1], Cp2[1], F3c2[0], F3t2[0], F3p2[0], F3c2[1], F3t2[1], F3p2[1], F3c2[2],
        F3t2[2], F3p2[2], F3c2[3], F3t2[3], F3p2[3], F5c2[0], F5t2[0], F5p2[0], F5c2[1], F5t2[1], F5p2[1], F5c2[2],
        F5t2[2], F5p2[2], F5c2[3], F5t2[3], F5p2[3], F5c2[4], F5t2[4], F5p2[4], F5c2[5], F5t2[5], F5p2[5]]
val3=[Cc3[0], Ct3[0], Cp3[0], Cc3[1], Ct3[1], Cp3[1], F3c3[0], F3t3[0], F3p3[0], F3c3[1], F3t3[1], F3p3[1], F3c3[2],
        F3t3[2], F3p3[2], F3c3[3], F3t3[3], F3p3[3], F5c3[0], F5t3[0], F5p3[0], F5c3[1], F5t3[1], F5p3[1], F5c3[2],
        F5t3[2], F5p3[2], F5c3[3], F5t3[3], F5p3[3], F5c3[4], F5t3[4], F5p3[4], F5c3[5], F5t3[5], F5p3[5]]
val4=[Cc4[0], Ct4[0], Cp4[0], Cc4[1], Ct4[1], Cp4[1], F3c4[0], F3t4[0], F3p4[0], F3c4[1], F3t4[1], F3p4[1], F3c4[2],
        F3t4[2], F3p4[2], F3c4[3], F3t4[3], F3p4[3], F5c4[0], F5t4[0], F5p4[0], F5c4[1], F5t4[1], F5p4[1], F5c4[2],
        F5t4[2], F5p4[2], F5c4[3], F5t4[3], F5p4[3], F5c4[4], F5t4[4], F5p4[4], F5c4[5], F5t4[5], F5p4[5]]
val5=[Cc5[0], Ct5[0], Cp5[0], Cc5[1], Ct5[1], Cp5[1], F3c5[0], F3t5[0], F3p5[0], F3c5[1], F3t5[1], F3p5[1], F3c5[2],
        F3t5[2], F3p5[2], F3c5[3], F3t5[3], F3p5[3], F5c5[0], F5t5[0], F5p5[0], F5c5[1], F5t5[1], F5p5[1], F5c5[2],
        F5t5[2], F5p5[2], F5c5[3], F5t5[3], F5p5[3], F5c5[4], F5t5[4], F5p5[4], F5c5[5], F5t5[5], F5p5[5]]
val6=[Cc6[0], Ct6[0], Cp6[0], Cc6[1], Ct6[1], Cp6[1], F3c6[0], F3t6[0], F3p6[0], F3c6[1], F3t6[1], F3p6[1], F3c6[2],
        F3t6[2], F3p6[2], F3c6[3], F3t6[3], F3p6[3], F5c6[0], F5t6[0], F5p6[0], F5c6[1], F5t6[1], F5p6[1], F5c6[2],
        F5t6[2], F5p6[2], F5c6[3], F5t6[3], F5p6[3], F5c6[4], F5t6[4], F5p6[4], F5c6[5], F5t6[5], F5p6[5]]
val7=[Cc7[0], Ct7[0], Cp7[0], Cc7[1], Ct7[1], Cp7[1], F3c7[0], F3t7[0], F3p7[0], F3c7[1], F3t7[1], F3p7[1], F3c7[2],
        F3t7[2], F3p7[2], F3c7[3], F3t7[3], F3p7[3], F5c7[0], F5t7[0], F5p7[0], F5c7[1], F5t7[1], F5p7[1], F5c7[2],
        F5t7[2], F5p7[2], F5c7[3], F5t7[3], F5p7[3], F5c7[4], F5t7[4], F5p7[4], F5c7[5], F5t7[5], F5p7[5]]
val8=[Cc8[0], Ct8[0], Cp8[0], Cc8[1], Ct8[1], Cp8[1], F3c8[0], F3t8[0], F3p8[0], F3c8[1], F3t8[1], F3p8[1], F3c8[2],
        F3t8[2], F3p8[2], F3c8[3], F3t8[3], F3p8[3], F5c8[0], F5t8[0], F5p8[0], F5c8[1], F5t8[1], F5p8[1], F5c8[2],
        F5t8[2], F5p8[2], F5c8[3], F5t8[3], F5p8[3], F5c8[4], F5t8[4], F5p8[4], F5c8[5], F5t8[5], F5p8[5]]
val9=[Cc9[0], Ct9[0], Cp9[0], Cc9[1], Ct9[1], Cp9[1], F3c9[0], F3t9[0], F3p9[0], F3c9[1], F3t9[1], F3p9[1], F3c9[2],
        F3t9[2], F3p9[2], F3c9[3], F3t9[3], F3p9[3], F5c9[0], F5t9[0], F5p9[0], F5c9[1], F5t9[1], F5p9[1], F5c9[2],
        F5t9[2], F5p9[2], F5c9[3], F5t9[3], F5p9[3], F5c9[4], F5t9[4], F5p9[4], F5c9[5], F5t9[5], F5p9[5]]
val10=[Cc10[0], Ct10[0], Cp10[0], Cc10[1], Ct10[1], Cp10[1], F3c10[0], F3t10[0], F3p10[0], F3c10[1], F3t10[1], F3p10[1], F3c10[2],
        F3t10[2], F3p10[2], F3c10[3], F3t10[3], F3p10[3], F5c10[0], F5t10[0], F5p10[0], F5c10[1], F5t10[1], F5p10[1], F5c10[2],
        F5t10[2], F5p10[2], F5c10[3], F5t10[3], F5p10[3], F5c10[4], F5t10[4], F5p10[4], F5c10[5], F5t10[5], F5p10[5]]
val11=[Cc11[0], Ct11[0], Cp11[0], Cc11[1], Ct11[1], Cp11[1], F3c11[0], F3t11[0], F3p11[0], F3c11[1], F3t11[1], F3p11[1], F3c11[2],
        F3t11[2], F3p11[2], F3c11[3], F3t11[3], F3p11[3], F5c11[0], F5t11[0], F5p11[0], F5c11[1], F5t11[1], F5p11[1], F5c11[2],
        F5t11[2], F5p11[2], F5c11[3], F5t11[3], F5p11[3], F5c11[4], F5t11[4], F5p11[4], F5c11[5], F5t11[5], F5p11[5]]
ind=['Cc[0]', 'Ct[0]', 'Cp[0]', 'Cc[1]', 'Ct[1]','Cp[1]', 'F3c[0]', 'F3t[0]', 'F3p[0]', 'F3c[1]', 'F3t[1]', 'F3p[1]',
       'F3c[2]', 'F3t[2]', 'F3p[2]', 'F3c[3]', 'F3t[3]', 'F3p[3]', 'F5c[0]', 'F5t[0]', 'F5p[0]', 'F5c[1]', 'F5t[1]', 'F5p[1]',
       'F5c[2]', 'F5t[2]', 'F5p[2]', 'F5c[3]', 'F5t[3]', 'F5p[3]', 'F5c[4]', 'F5t[4]', 'F5p[4]', 'F5c[5]', 'F5t[5]', 'F5p[5]']


# In[ ]:


a = pd.DataFrame({'Coefficient':ind,'1':val1,'2':val2,'3':val3,'4':val4,'5':val5,'6':val6,
               '7':val7,'8':val8,'9':val9,'10':val10,'10-1':val11})

a.set_index('Coefficient', inplace=True)

a = a.applymap('{:,.2f}'.format)

#a.to_csv('POS_PE-Sorted_Risk-Adjusted_Results_NotWIN.csv', sep=',')
a


# ## Winzorised Data

# In[ ]:


Rf = pd.read_csv('F-F_Research_Data_5_Factors_2x3.csv', delimiter=',')

Rf['t'] = Rf['t'].astype(str)
Rf['year'] = Rf['t'].str[:4]
Rf['month'] = Rf['t'].str[4:6]

Rf['month'] = Rf['month'].astype('int32')
Rf['year'] = Rf['year'].astype('int32')

Rf1 = Rf[Rf.month < 7]
Rf2 = Rf[Rf.month > 6]

Rf1['year'] = Rf1['year']-1

Rf = pd.concat([Rf1, Rf2], ignore_index=True)

Rf[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']] = Rf[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']]/100

def getCumRet (group):
    group['Mkt_Rf'] = np.exp(np.log(group['Mkt-RF']+1).rolling(len(group)).sum())-1
    group['SMBc'] = np.exp(np.log(group['SMB']+1).rolling(len(group)).sum())-1
    group['HMLc'] = np.exp(np.log(group['HML']+1).rolling(len(group)).sum())-1
    group['RMWc'] = np.exp(np.log(group['RMW']+1).rolling(len(group)).sum())-1
    group['CMAc'] = np.exp(np.log(group['CMA']+1).rolling(len(group)).sum())-1
    group['RFc'] = np.exp(np.log(group['RF']+1).rolling(len(group)).sum())-1
    return group

Rf = Rf.groupby(['year']).apply(getCumRet)

Rf = Rf[Rf['Mkt_Rf'].notna()]

Rf[['Mkt_Rf', 'SMBc', 'HMLc', 'RMWc', 'CMAc', 'RFc']] = Rf[['Mkt_Rf', 'SMBc', 'HMLc', 'RMWc', 'CMAc', 'RFc']]*100


# In[ ]:


final_df = pd.read_csv('C:/Users/timmp/Desktop/Bachelorarbeit_F/POS_Annual_rt+1_Win.csv', delimiter=',')

final_df = final_df.merge(Rf, on='year')

final_df = final_df[['year', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '10-1', 'Mkt_Rf', 'SMBc',
       'HMLc', 'RMWc', 'CMAc', 'RFc']]

column_names = {'1':'P1', '2':'P2', '3':'P3', '4':'P4', '5':'P5', '6':'P6', '7':'P7', '8':'P8', '9':'P9', '10':'P10', '10-1':'PDif', 'Mkt_Rf':'MKT'}
final_df.rename(columns=column_names, inplace=True)

final_df.set_index('year', inplace=True)


# In[ ]:


CAPM_1 = smp.ols(formula = 'P1 ~ MKT', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF3_1 = smp.ols(formula = 'P1 ~ MKT + SMBc + HMLc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF5_1 = smp.ols(formula = 'P1 ~ MKT + SMBc + HMLc + RMWc + CMAc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
CAPM_2 = smp.ols(formula = 'P2 ~ MKT', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF3_2 = smp.ols(formula = 'P2 ~ MKT + SMBc + HMLc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF5_2 = smp.ols(formula = 'P2 ~ MKT + SMBc + HMLc + RMWc + CMAc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
CAPM_3 = smp.ols(formula = 'P3 ~ MKT', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF3_3 = smp.ols(formula = 'P3 ~ MKT + SMBc + HMLc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF5_3 = smp.ols(formula = 'P3 ~ MKT + SMBc + HMLc + RMWc + CMAc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
CAPM_4 = smp.ols(formula = 'P4 ~ MKT', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF3_4 = smp.ols(formula = 'P4 ~ MKT + SMBc + HMLc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF5_4 = smp.ols(formula = 'P4 ~ MKT + SMBc + HMLc + RMWc + CMAc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
CAPM_5 = smp.ols(formula = 'P5 ~ MKT', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF3_5 = smp.ols(formula = 'P5 ~ MKT + SMBc + HMLc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF5_5 = smp.ols(formula = 'P5 ~ MKT + SMBc + HMLc + RMWc + CMAc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
CAPM_6 = smp.ols(formula = 'P6 ~ MKT', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF3_6 = smp.ols(formula = 'P6 ~ MKT + SMBc + HMLc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF5_6 = smp.ols(formula = 'P6 ~ MKT + SMBc + HMLc + RMWc + CMAc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
CAPM_7 = smp.ols(formula = 'P7 ~ MKT', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF3_7 = smp.ols(formula = 'P7 ~ MKT + SMBc + HMLc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF5_7 = smp.ols(formula = 'P7 ~ MKT + SMBc + HMLc + RMWc + CMAc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
CAPM_8 = smp.ols(formula = 'P8 ~ MKT', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF3_8 = smp.ols(formula = 'P8 ~ MKT + SMBc + HMLc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF5_8 = smp.ols(formula = 'P8 ~ MKT + SMBc + HMLc + RMWc + CMAc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
CAPM_9 = smp.ols(formula = 'P9 ~ MKT', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF3_9 = smp.ols(formula = 'P9 ~ MKT + SMBc + HMLc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF5_9 = smp.ols(formula = 'P9 ~ MKT + SMBc + HMLc + RMWc + CMAc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
CAPM_10 = smp.ols(formula = 'P10 ~ MKT', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF3_10 = smp.ols(formula = 'P10 ~ MKT + SMBc + HMLc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF5_10 = smp.ols(formula = 'P10 ~ MKT + SMBc + HMLc + RMWc + CMAc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
CAPM_11 = smp.ols(formula = 'PDif ~ MKT', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF3_11 = smp.ols(formula = 'PDif ~ MKT + SMBc + HMLc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})
FF5_11 = smp.ols(formula = 'PDif ~ MKT + SMBc + HMLc + RMWc + CMAc', data=final_df).fit(cov_type='HAC',cov_kwds={'maxlags':6})


# In[ ]:


Cc1 = CAPM_1.params
F3c1 = FF3_1.params
F5c1 = FF5_1.params
Cc2 = CAPM_2.params
F3c2 = FF3_2.params
F5c2 = FF5_2.params
Cc3 = CAPM_3.params
F3c3 = FF3_3.params
F5c3 = FF5_3.params
Cc4 = CAPM_4.params
F3c4 = FF3_4.params
F5c4 = FF5_4.params
Cc5 = CAPM_5.params
F3c5 = FF3_5.params
F5c5 = FF5_5.params
Cc6 = CAPM_6.params
F3c6 = FF3_6.params
F5c6 = FF5_6.params
Cc7 = CAPM_7.params
F3c7 = FF3_7.params
F5c7 = FF5_7.params
Cc8 = CAPM_8.params
F3c8 = FF3_8.params
F5c8 = FF5_8.params
Cc9 = CAPM_9.params
F3c9 = FF3_9.params
F5c9 = FF5_9.params
Cc10 = CAPM_10.params
F3c10 = FF3_10.params
F5c10 = FF5_10.params
Cc11 = CAPM_11.params
F3c11 = FF3_11.params
F5c11 = FF5_11.params


# In[ ]:


Ct1 = CAPM_1.tvalues
F3t1 = FF3_1.tvalues
F5t1 = FF5_1.tvalues
Ct2 = CAPM_2.tvalues
F3t2 = FF3_2.tvalues
F5t2 = FF5_2.tvalues
Ct3 = CAPM_3.tvalues
F3t3 = FF3_3.tvalues
F5t3 = FF5_3.tvalues
Ct4 = CAPM_4.tvalues
F3t4 = FF3_4.tvalues
F5t4 = FF5_4.tvalues
Ct5 = CAPM_5.tvalues
F3t5 = FF3_5.tvalues
F5t5 = FF5_5.tvalues
Ct6 = CAPM_6.tvalues
F3t6 = FF3_6.tvalues
F5t6 = FF5_6.tvalues
Ct7 = CAPM_7.tvalues
F3t7 = FF3_7.tvalues
F5t7 = FF5_7.tvalues
Ct8 = CAPM_8.tvalues
F3t8 = FF3_8.tvalues
F5t8 = FF5_8.tvalues
Ct9 = CAPM_9.tvalues
F3t9 = FF3_9.tvalues
F5t9 = FF5_9.tvalues
Ct10 = CAPM_10.tvalues
F3t10 = FF3_10.tvalues
F5t10 = FF5_10.tvalues
Ct11 = CAPM_11.tvalues
F3t11 = FF3_11.tvalues
F5t11 = FF5_11.tvalues


# In[ ]:


Cp1 = CAPM_1.pvalues
F3p1 = FF3_1.pvalues
F5p1 = FF5_1.pvalues
Cp2 = CAPM_2.pvalues
F3p2 = FF3_2.pvalues
F5p2 = FF5_2.pvalues
Cp3 = CAPM_3.pvalues
F3p3 = FF3_3.pvalues
F5p3 = FF5_3.pvalues
Cp4 = CAPM_4.pvalues
F3p4 = FF3_4.pvalues
F5p4 = FF5_4.pvalues
Cp5 = CAPM_5.pvalues
F3p5 = FF3_5.pvalues
F5p5 = FF5_5.pvalues
Cp6 = CAPM_6.pvalues
F3p6 = FF3_6.pvalues
F5p6 = FF5_6.pvalues
Cp7 = CAPM_7.pvalues
F3p7 = FF3_7.pvalues
F5p7 = FF5_7.pvalues
Cp8 = CAPM_8.pvalues
F3p8 = FF3_8.pvalues
F5p8 = FF5_8.pvalues
Cp9 = CAPM_9.pvalues
F3p9 = FF3_9.pvalues
F5p9 = FF5_9.pvalues
Cp10 = CAPM_10.pvalues
F3p10 = FF3_10.pvalues
F5p10 = FF5_10.pvalues
Cp11 = CAPM_11.pvalues
F3p11 = FF3_11.pvalues
F5p11 = FF5_11.pvalues


# In[ ]:


val1=[Cc1[0], Ct1[0], Cp1[0], Cc1[1], Ct1[1], Cp1[1], F3c1[0], F3t1[0], F3p1[0], F3c1[1], F3t1[1], F3p1[1], F3c1[2],
        F3t1[2], F3p1[2], F3c1[3], F3t1[3], F3p1[3], F5c1[0], F5t1[0], F5p1[0], F5c1[1], F5t1[1], F5p1[1], F5c1[2],
        F5t1[2], F5p1[2], F5c1[3], F5t1[3], F5p1[3], F5c1[4], F5t1[4], F5p1[4], F5c1[5], F5t1[5], F5p1[5]]
val2=[Cc2[0], Ct2[0], Cp2[0], Cc2[1], Ct2[1], Cp2[1], F3c2[0], F3t2[0], F3p2[0], F3c2[1], F3t2[1], F3p2[1], F3c2[2],
        F3t2[2], F3p2[2], F3c2[3], F3t2[3], F3p2[3], F5c2[0], F5t2[0], F5p2[0], F5c2[1], F5t2[1], F5p2[1], F5c2[2],
        F5t2[2], F5p2[2], F5c2[3], F5t2[3], F5p2[3], F5c2[4], F5t2[4], F5p2[4], F5c2[5], F5t2[5], F5p2[5]]
val3=[Cc3[0], Ct3[0], Cp3[0], Cc3[1], Ct3[1], Cp3[1], F3c3[0], F3t3[0], F3p3[0], F3c3[1], F3t3[1], F3p3[1], F3c3[2],
        F3t3[2], F3p3[2], F3c3[3], F3t3[3], F3p3[3], F5c3[0], F5t3[0], F5p3[0], F5c3[1], F5t3[1], F5p3[1], F5c3[2],
        F5t3[2], F5p3[2], F5c3[3], F5t3[3], F5p3[3], F5c3[4], F5t3[4], F5p3[4], F5c3[5], F5t3[5], F5p3[5]]
val4=[Cc4[0], Ct4[0], Cp4[0], Cc4[1], Ct4[1], Cp4[1], F3c4[0], F3t4[0], F3p4[0], F3c4[1], F3t4[1], F3p4[1], F3c4[2],
        F3t4[2], F3p4[2], F3c4[3], F3t4[3], F3p4[3], F5c4[0], F5t4[0], F5p4[0], F5c4[1], F5t4[1], F5p4[1], F5c4[2],
        F5t4[2], F5p4[2], F5c4[3], F5t4[3], F5p4[3], F5c4[4], F5t4[4], F5p4[4], F5c4[5], F5t4[5], F5p4[5]]
val5=[Cc5[0], Ct5[0], Cp5[0], Cc5[1], Ct5[1], Cp5[1], F3c5[0], F3t5[0], F3p5[0], F3c5[1], F3t5[1], F3p5[1], F3c5[2],
        F3t5[2], F3p5[2], F3c5[3], F3t5[3], F3p5[3], F5c5[0], F5t5[0], F5p5[0], F5c5[1], F5t5[1], F5p5[1], F5c5[2],
        F5t5[2], F5p5[2], F5c5[3], F5t5[3], F5p5[3], F5c5[4], F5t5[4], F5p5[4], F5c5[5], F5t5[5], F5p5[5]]
val6=[Cc6[0], Ct6[0], Cp6[0], Cc6[1], Ct6[1], Cp6[1], F3c6[0], F3t6[0], F3p6[0], F3c6[1], F3t6[1], F3p6[1], F3c6[2],
        F3t6[2], F3p6[2], F3c6[3], F3t6[3], F3p6[3], F5c6[0], F5t6[0], F5p6[0], F5c6[1], F5t6[1], F5p6[1], F5c6[2],
        F5t6[2], F5p6[2], F5c6[3], F5t6[3], F5p6[3], F5c6[4], F5t6[4], F5p6[4], F5c6[5], F5t6[5], F5p6[5]]
val7=[Cc7[0], Ct7[0], Cp7[0], Cc7[1], Ct7[1], Cp7[1], F3c7[0], F3t7[0], F3p7[0], F3c7[1], F3t7[1], F3p7[1], F3c7[2],
        F3t7[2], F3p7[2], F3c7[3], F3t7[3], F3p7[3], F5c7[0], F5t7[0], F5p7[0], F5c7[1], F5t7[1], F5p7[1], F5c7[2],
        F5t7[2], F5p7[2], F5c7[3], F5t7[3], F5p7[3], F5c7[4], F5t7[4], F5p7[4], F5c7[5], F5t7[5], F5p7[5]]
val8=[Cc8[0], Ct8[0], Cp8[0], Cc8[1], Ct8[1], Cp8[1], F3c8[0], F3t8[0], F3p8[0], F3c8[1], F3t8[1], F3p8[1], F3c8[2],
        F3t8[2], F3p8[2], F3c8[3], F3t8[3], F3p8[3], F5c8[0], F5t8[0], F5p8[0], F5c8[1], F5t8[1], F5p8[1], F5c8[2],
        F5t8[2], F5p8[2], F5c8[3], F5t8[3], F5p8[3], F5c8[4], F5t8[4], F5p8[4], F5c8[5], F5t8[5], F5p8[5]]
val9=[Cc9[0], Ct9[0], Cp9[0], Cc9[1], Ct9[1], Cp9[1], F3c9[0], F3t9[0], F3p9[0], F3c9[1], F3t9[1], F3p9[1], F3c9[2],
        F3t9[2], F3p9[2], F3c9[3], F3t9[3], F3p9[3], F5c9[0], F5t9[0], F5p9[0], F5c9[1], F5t9[1], F5p9[1], F5c9[2],
        F5t9[2], F5p9[2], F5c9[3], F5t9[3], F5p9[3], F5c9[4], F5t9[4], F5p9[4], F5c9[5], F5t9[5], F5p9[5]]
val10=[Cc10[0], Ct10[0], Cp10[0], Cc10[1], Ct10[1], Cp10[1], F3c10[0], F3t10[0], F3p10[0], F3c10[1], F3t10[1], F3p10[1], F3c10[2],
        F3t10[2], F3p10[2], F3c10[3], F3t10[3], F3p10[3], F5c10[0], F5t10[0], F5p10[0], F5c10[1], F5t10[1], F5p10[1], F5c10[2],
        F5t10[2], F5p10[2], F5c10[3], F5t10[3], F5p10[3], F5c10[4], F5t10[4], F5p10[4], F5c10[5], F5t10[5], F5p10[5]]
val11=[Cc11[0], Ct11[0], Cp11[0], Cc11[1], Ct11[1], Cp11[1], F3c11[0], F3t11[0], F3p11[0], F3c11[1], F3t11[1], F3p11[1], F3c11[2],
        F3t11[2], F3p11[2], F3c11[3], F3t11[3], F3p11[3], F5c11[0], F5t11[0], F5p11[0], F5c11[1], F5t11[1], F5p11[1], F5c11[2],
        F5t11[2], F5p11[2], F5c11[3], F5t11[3], F5p11[3], F5c11[4], F5t11[4], F5p11[4], F5c11[5], F5t11[5], F5p11[5]]
ind=['Cc[0]', 'Ct[0]', 'Cp[0]', 'Cc[1]', 'Ct[1]','Cp[1]', 'F3c[0]', 'F3t[0]', 'F3p[0]', 'F3c[1]', 'F3t[1]', 'F3p[1]',
       'F3c[2]', 'F3t[2]', 'F3p[2]', 'F3c[3]', 'F3t[3]', 'F3p[3]', 'F5c[0]', 'F5t[0]', 'F5p[0]', 'F5c[1]', 'F5t[1]', 'F5p[1]',
       'F5c[2]', 'F5t[2]', 'F5p[2]', 'F5c[3]', 'F5t[3]', 'F5p[3]', 'F5c[4]', 'F5t[4]', 'F5p[4]', 'F5c[5]', 'F5t[5]', 'F5p[5]']


# In[ ]:


a = pd.DataFrame({'Coefficient':ind,'1':val1,'2':val2,'3':val3,'4':val4,'5':val5,'6':val6,
               '7':val7,'8':val8,'9':val9,'10':val10,'10-1':val11})

a.set_index('Coefficient', inplace=True)

a = a.applymap('{:,.2f}'.format)

a.to_csv('POS_PE-Sorted_Risk-Adjusted_Results_WIN.csv', sep=',')


# -------

# # Fama MacBeth Regression Analyse
#
# The model is given by
#        $ y_{it}=\beta^{\prime}x_{it}+\epsilon_{it} $
#     The Fama-MacBeth estimator is computed by performing T regressions, one
#     for each time period using all available entity observations.  Denote the
#     estimate of the model parameters as $ `\hat{\beta}_t` $.  The reported
#     estimator is then
#        $ \hat{\beta} = T^{-1}\sum_{t=1}^T \hat{\beta}_t $
#     While the model does not explicitly include time-effects, the
#     implementation based on regressing all observation in a single
#     time period is "as-if" time effects are included.
#     Parameter inference is made using the set of T parameter estimates with
#     either the standard covariance estimator or a kernel-based covariance,
#     depending on ``cov_type``.

# In[ ]:


df = pd.read_csv('POS_final_df.csv', delimiter=',')

df1 = df[df.month == 6]

df2 = df[df.month == 7]

df3 = df[df['CumRET'].notna()]

df1 = df1[['year', 'PERMNO', 'Size', 'bm', 'pe_exi']]

df2 = df2[['year', 'PERMNO', 'Beta']]

df3 = df3[['year', 'PERMNO', 'CumRET']]

df1['year'] = df1['year'] + 1

df3 = df3[df3.year > 1969]

df1 = df1[df1.year < 2019]

final_df = df3.merge(df1, how='left', on=['year', 'PERMNO'])

df = final_df.merge(df2, how='left', on=['year', 'PERMNO'])


# In[ ]:


df['Beta'] = stats.mstats.winsorize(df['Beta'], limits=[.005,.005])
df['pe_exi'] = stats.mstats.winsorize(df['pe_exi'], limits=[.005,.005])
df['Size'] = stats.mstats.winsorize(df['Size'], limits=[.005,.005])
df['bm'] = stats.mstats.winsorize(df['bm'], limits=[.005,.005])
df['Beta'] = stats.mstats.winsorize(df['Beta'], limits=[.005,.005])


# In[ ]:


df['CumRET'] = df['CumRET']*100


# In[ ]:


def FamaMacBeth(group, formula):
    FM = smp.ols(formula, data=group).fit(cov_type='HAC',cov_kwds={'maxlags':6})
    FMparams = FM.params[:]
    FMparams['R2t'] = FM.rsquared
    FMparams['Adj.R2t'] = FM.rsquared_adj
    FMparams['nt'] = FM.nobs
    return FMparams


# In[ ]:


Panel_A = df.groupby('year').apply(FamaMacBeth, 'CumRET~ 1 + pe_exi')

Panel_B = df.groupby('year').apply(FamaMacBeth, 'CumRET~ 1 + Size')

Panel_C = df.groupby('year').apply(FamaMacBeth, 'CumRET~ 1 + bm')

Panel_D = df.groupby('year').apply(FamaMacBeth, 'CumRET~ 1 + Beta')

Panel_E = df.groupby('year').apply(FamaMacBeth, 'CumRET~ 1 + pe_exi + Size + bm + Beta')


# In[ ]:


Panel_A['dummy'] = 0
Panel_B['dummy'] = 0
Panel_C['dummy'] = 0
Panel_D['dummy'] = 0
Panel_E['dummy'] = 0


# In[ ]:


V2 = smp.ols('Intercept ~ 1 + dummy', data=Panel_A).fit().get_robustcov_results(cov_type='HAC',maxlags=6)
x = pd.DataFrame(columns=['Average', 'Standard error', 't-statistic', 'p-value'], index=['1'])
x['Average'] = V2.params[0]
x['Standard error'] = V2.bse[0]
x['t-statistic'] = V2.tvalues[0]
x['p-value'] = V2.pvalues[0]

V3 = smp.ols('pe_exi ~ 1 + dummy', data=Panel_A).fit().get_robustcov_results(cov_type='HAC',maxlags=6)
x['Average1'] = V3.params[0]
x['Standard error1'] = V3.bse[0]
x['t-statistic1'] = V3.tvalues[0]
x['p-value1'] = V3.pvalues[0]

x['R2t'] = Panel_A['R2t'].mean()
x['Adj.R2t'] = Panel_A['Adj.R2t'].mean()
x['nt'] = Panel_A['nt'].mean().round()

#------
V4 = smp.ols('Intercept ~ 1 + dummy', data=Panel_B).fit().get_robustcov_results(cov_type='HAC',maxlags=6)
y = pd.DataFrame(columns=['Average', 'Standard error', 't-statistic', 'p-value'], index=['2'])
y['Average'] = V4.params[0]
y['Standard error'] = V4.bse[0]
y['t-statistic'] = V4.tvalues[0]
y['p-value'] = V4.pvalues[0]

V5 = smp.ols('Size ~ 1 + dummy', data=Panel_B).fit().get_robustcov_results(cov_type='HAC',maxlags=6)
y['Average2'] = V5.params[0]
y['Standard error2'] = V5.bse[0]
y['t-statistic2'] = V5.tvalues[0]
y['p-value2'] = V5.pvalues[0]

y['R2t'] = Panel_B['R2t'].mean()
y['Adj.R2t'] = Panel_B['Adj.R2t'].mean()
y['nt'] = Panel_B['nt'].mean().round()

#------
V6 = smp.ols('Intercept ~ 1 + dummy', data=Panel_C).fit().get_robustcov_results(cov_type='HAC',maxlags=6)
z = pd.DataFrame(columns=['Average', 'Standard error', 't-statistic', 'p-value'], index=['3'])
z['Average'] = V6.params[0]
z['Standard error'] = V6.bse[0]
z['t-statistic'] = V6.tvalues[0]
z['p-value'] = V6.pvalues[0]

V7 = smp.ols('bm ~ 1 + dummy', data=Panel_C).fit().get_robustcov_results(cov_type='HAC',maxlags=6)
z['Average3'] = V7.params[0]
z['Standard error3'] = V7.bse[0]
z['t-statistic3'] = V7.tvalues[0]
z['p-value3'] = V7.pvalues[0]

z['R2t'] = Panel_C['R2t'].mean()
z['Adj.R2t'] = Panel_C['Adj.R2t'].mean()
z['nt'] = Panel_C['nt'].mean().round()

#------
V8 = smp.ols('Intercept ~ 1 + dummy', data=Panel_D).fit().get_robustcov_results(cov_type='HAC',maxlags=6)
v = pd.DataFrame(columns=['Average', 'Standard error', 't-statistic', 'p-value'], index=['4'])
v['Average'] = V8.params[0]
v['Standard error'] = V8.bse[0]
v['t-statistic'] = V8.tvalues[0]
v['p-value'] = V8.pvalues[0]

V9 = smp.ols('Beta ~ 1 + dummy', data=Panel_D).fit().get_robustcov_results(cov_type='HAC',maxlags=6)
v['Average4'] = V9.params[0]
v['Standard error4'] = V9.bse[0]
v['t-statistic4'] = V9.tvalues[0]
v['p-value4'] = V9.pvalues[0]

v['R2t'] = Panel_D['R2t'].mean()
v['Adj.R2t'] = Panel_D['Adj.R2t'].mean()
v['nt'] = Panel_D['nt'].mean().round()

#------
V10 = smp.ols('Intercept ~ 1 + dummy', data=Panel_E).fit().get_robustcov_results(cov_type='HAC',maxlags=6)
w = pd.DataFrame(columns=['Average', 'Standard error', 't-statistic', 'p-value'], index=['5'])
w['Average'] = V10.params[0]
w['Standard error'] = V10.bse[0]
w['t-statistic'] = V10.tvalues[0]
w['p-value'] = V10.pvalues[0]

V11 = smp.ols('pe_exi ~ 1 + dummy', data=Panel_E).fit().get_robustcov_results(cov_type='HAC',maxlags=6)
w['Average1'] = V11.params[0]
w['Standard error1'] = V11.bse[0]
w['t-statistic1'] = V11.tvalues[0]
w['p-value1'] = V11.pvalues[0]

V12 = smp.ols('Size ~ 1 + dummy', data=Panel_E).fit().get_robustcov_results(cov_type='HAC',maxlags=6)
w['Average2'] = V12.params[0]
w['Standard error2'] = V12.bse[0]
w['t-statistic2'] = V12.tvalues[0]
w['p-value2'] = V12.pvalues[0]

V13 = smp.ols('bm ~ 1 + dummy', data=Panel_E).fit().get_robustcov_results(cov_type='HAC',maxlags=6)
w['Average3'] = V13.params[0]
w['Standard error3'] = V13.bse[0]
w['t-statistic3'] = V13.tvalues[0]
w['p-value3'] = V13.pvalues[0]

V14 = smp.ols('Beta ~ 1 + dummy', data=Panel_E).fit().get_robustcov_results(cov_type='HAC',maxlags=6)
w['Average4'] = V14.params[0]
w['Standard error4'] = V14.bse[0]
w['t-statistic4'] = V14.tvalues[0]
w['p-value4'] = V14.pvalues[0]

w['R2t'] = Panel_E['R2t'].mean()
w['Adj.R2t'] = Panel_E['Adj.R2t'].mean()
w['nt'] = Panel_E['nt'].mean().round()


# In[ ]:


FM = pd.concat([x,y,z,v,w],axis=0).applymap('{:,.3f}'.format).T


# In[ ]:


FM.to_csv('POS_FM_Regression_Results.csv', sep=',')
