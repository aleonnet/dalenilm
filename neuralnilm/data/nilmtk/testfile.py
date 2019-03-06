from sample_data import safe_resample
from good_sections import find_good_sections
from power_series import power_series_all_data
from open_dataset import load_appliance_dataset,load_mains_dataset
from set_time_windows import set_window
from get_activations import get_app_activations

import pandas as pd
import numpy as np

df=load_appliance_dataset(1)
df=set_window(df,'2011-04-19', '2011-05-21')
sample_period=6
resample_kwargs={}
resample_kwargs['rule'] = '{:d}S'.format(sample_period)
df=df['power']['apparent']
df=safe_resample(df,resample_kwargs)
df=df.agg(np.mean)
df=df.interpolate()
df=get_app_activations(df)

#my_interval=find_good_sections(df,30)


#columns=[('power','apparent')]
#LEVEL_NAMES=['physical_quantity', 'type']
#df1 = pd.read_csv('../data/redd/house_1/channel_1.dat', sep=' ', names=columns, dtype={m:np.float32 for m in columns})
#df1.columns.set_names(LEVEL_NAMES, inplace=True)
#df1.index = pd.to_datetime(df1.index.values, unit='s', utc=True)
#df1 = df1.tz_convert('US/Eastern')

#df2 = pd.read_csv('../data/redd/house_1/channel_2.dat', sep=' ', names=columns, dtype={m:np.float32 for m in columns})
#df2.columns.set_names(LEVEL_NAMES, inplace=True)
#df2.index = pd.to_datetime(df2.index.values, unit='s', utc=True)
#df2 = df2.tz_convert('US/Eastern')

#df=df1+df2
#sample_period=6
#df=df[(df.index>='2011-04-19 00:30:36-04:00')&(df.index<='2011-05-21')]

#df=df.interpolate()
#my_interval=find_good_sections(df,30)
#resample_kwargs={}
#resample_kwargs['rule'] = '{:d}S'.format(sample_period)
#df=df['power']['apparent']
#df=safe_resample(df,resample_kwargs)
#df=df.agg(np.mean)
#df=df.interpolate()

#ans=power_series_all_data(df,my_interval).dropna()
