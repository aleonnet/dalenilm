import pandas as pd
import numpy as np

def power_series_all_data(df, good_sections):
    chunks=[]
    
    for duration in good_sections:
        point_start=duration.start
        point_end=duration.end
        my_data=df[(df.index>=point_start)&(df.index<=point_end)]
        chunks.append(my_data)
    if chunks:
        prev_end = None
        for i, chunk in enumerate(chunks):
            if i > 0:
                if chunk.index[0] <= prev_end:
                    chunks[i] = chunk.iloc[1:]
            prev_end = chunk.index[-1]
            
        all_data = pd.concat(chunks)
    else:
        all_data = None
    return all_data

