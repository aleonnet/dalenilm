import numpy as np
import pandas as pd
from nilmtk.consts import columns, LEVEL_NAMES

def load_appliance_dataset(house_no):
    #my_channel={1:'channel_5',2: 'channel_9',3:'channel_7', 4:None, 5:'channel_18', 6:'channel_8'}
    uk_dale_channel={1:'channel_12', 2:'channel_14', 3:None, 4:'channel_5', 5:'channel_19'}
    
    #channel_no=my_channel[house_no]
    channel_no=uk_dale_channel[house_no]
    house_no=str(house_no)
    filename='ukdale/house_'+house_no+'/'+channel_no+'.dat'
    df = pd.read_csv(filename, sep=' ', names=columns, dtype={m:np.float32 for m in columns})
    df.columns.set_names(LEVEL_NAMES, inplace=True)
    df.index = pd.to_datetime(df.index.values, unit='s', utc=True)
    df = df.tz_convert('US/Eastern')
    return df

def load_mains_dataset(house_no):
    house_no=str(house_no)
    df1 = pd.read_csv('ukdale/house_'+house_no+'/channel_1.dat', sep=' ', names=columns, dtype={m:np.float32 for m in columns})
    df1.columns.set_names(LEVEL_NAMES, inplace=True)
    df1.index = pd.to_datetime(df1.index.values, unit='s', utc=True)
    df1 = df1.tz_convert('US/Eastern')
    return df1