import pandas as pd
import numpy as np
from numpy import diff, concatenate
import gc
from nilmtk.timeframe import TimeFrame

def timedelta64_to_secs(timedelta):
    """Convert `timedelta` to seconds.
    Parameters
    ----------
    timedelta : np.timedelta64
    Returns
    -------
    float : seconds
    """
    if len(timedelta) == 0:
        return np.array([])
    else:
        return timedelta / np.timedelta64(1, 's')
 
def find_good_sections(df,max_sample_period,look_ahead=None,previous_chunk_ended_with_open_ended_good_section=False):
    index = df.dropna().sort_index().index
    del df

    if len(index) < 2:
        return []

    timedeltas_sec = timedelta64_to_secs(diff(index.values))
    timedeltas_check = timedeltas_sec <= max_sample_period

    # Memory management
    del timedeltas_sec
    gc.collect()

    timedeltas_check = concatenate(
        [[previous_chunk_ended_with_open_ended_good_section],
         timedeltas_check])
    transitions = diff(timedeltas_check.astype(np.int))

    # Memory management
    last_timedeltas_check = timedeltas_check[-1]
    del timedeltas_check
    gc.collect()

    good_sect_starts = list(index[:-1][transitions ==  1])
    good_sect_ends   = list(index[:-1][transitions == -1])

    # Memory management
    last_index = index[-1]
    del index
    gc.collect()

    # Use look_ahead to see if we need to append a 
    # good sect start or good sect end.
    look_ahead_valid = look_ahead is not None and not look_ahead.empty
    if look_ahead_valid:
        look_ahead_timedelta = look_ahead.dropna().index[0] - last_index
        look_ahead_gap = look_ahead_timedelta.total_seconds()
    if last_timedeltas_check: # current chunk ends with a good section
        if not look_ahead_valid or look_ahead_gap > max_sample_period:
            # current chunk ends with a good section which needs to 
            # be closed because next chunk either does not exist
            # or starts with a sample which is more than max_sample_period
            # away from df.index[-1]
            good_sect_ends += [last_index]
    elif look_ahead_valid and look_ahead_gap <= max_sample_period:
        # Current chunk appears to end with a bad section
        # but last sample is the start of a good section
        good_sect_starts += [last_index]

    # Work out if this chunk ends with an open ended good section
    if len(good_sect_ends) == 0:
        ends_with_open_ended_good_section = (
            len(good_sect_starts) > 0 or 
            previous_chunk_ended_with_open_ended_good_section)
    elif len(good_sect_starts) > 0:
        # We have good_sect_ends and good_sect_starts
        ends_with_open_ended_good_section = (
            good_sect_ends[-1] < good_sect_starts[-1])
    else:
        # We have good_sect_ends but no good_sect_starts
        ends_with_open_ended_good_section = False

    # If this chunk starts or ends with an open-ended
    # good section then the relevant TimeFrame needs to have
    # a None as the start or end.
    if previous_chunk_ended_with_open_ended_good_section:
        good_sect_starts = [None] + good_sect_starts
    if ends_with_open_ended_good_section:
        good_sect_ends += [None]

    assert len(good_sect_starts) == len(good_sect_ends)

    sections = [TimeFrame(start, end)
                for start, end in zip(good_sect_starts, good_sect_ends)
                if not (start == end and start is not None)]

    # Memory management
    del good_sect_starts
    del good_sect_ends
    gc.collect()
    return sections

