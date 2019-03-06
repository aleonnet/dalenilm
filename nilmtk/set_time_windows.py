def set_window(df,start,end):
    if end==None:
        return df[(df.index>=start)]
    else:
        return df[(df.index>=start)&(df.index<=end)]
    