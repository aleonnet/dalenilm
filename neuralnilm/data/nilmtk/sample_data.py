import warnings
import pandas as pd
import numpy as np

def safe_resample(data,resample_kwargs):
    if data.empty:
        return data
    
    def _resample_chain(data, all_resample_kwargs):
        rule=all_resample_kwargs.pop('rule')
        axis = all_resample_kwargs.pop('axis', None)
        on = all_resample_kwargs.pop('on', None)
        level = all_resample_kwargs.pop('level', None)
        
        resample_kwargs = {}
        if axis is not None: resample_kwargs['axis'] = axis
        if on is not None: resample_kwargs['on'] = on
        if level is not None: resample_kwargs['level'] = level
        
        fill_method_str=all_resample_kwargs.pop('fill_method', None)
        
        if fill_method_str:
            fill_method = lambda df: getattr(df, fill_method_str)()
            
        else:
            fill_method = lambda df: df
            
        how_str = all_resample_kwargs.pop('how', None)
        if how_str:
            how=lambda df: getattr(df,how_str)()
        else:
            how=lambda df: df
            
        if resample_kwargs:
            warnings.warn("Not all resample_kwargs were consumed: {}".format(repr(resample_kwargs))) 
        return fill_method(how(data.resample(rule, **resample_kwargs)))
    
    try:
        dups_in_index=data.index.duplicated(keep='first')
        if dups_in_index.any():
            warnings.warn("Found duplicate index. Keeping first value")
            data = data[~dups_in_index]
            
        data=_resample_chain(data,resample_kwargs)
        #print(data)
    except pytz.AmbiguousTimeError:
        tz = data.index.tz.zone
        data = data.tz_convert('UTC')
        data = _resample_chain(data, resample_kwargs)
        data = data.tz_convert(tz)
        
    return data


