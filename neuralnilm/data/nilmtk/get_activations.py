import numpy as np
import pandas as pd
from nilmtk.consts import timedelta64_to_secs

def get_app_activations(chunk, min_off_duration=0, min_on_duration=0,
                    border=1, on_power_threshold=10):
    
    when_on = chunk >= on_power_threshold
    #print(when_on)

    # Find state changes
    state_changes = when_on.astype(np.int8).diff()
    #print(state_changes)
    del when_on
    switch_on_events = np.where(state_changes == 1)[0]
    switch_off_events = np.where(state_changes == -1)[0]
    del state_changes
    
    #print(switch_on_events)
    #print(len(switch_on_events))
    #print("***************************************************")
    #print(switch_off_events)

    if len(switch_on_events) == 0 or len(switch_off_events) == 0:
        return []

    # Make sure events align
    if switch_off_events[0] < switch_on_events[0]:
        switch_off_events = switch_off_events[1:]
        if len(switch_off_events) == 0:
            return []
    if switch_on_events[-1] > switch_off_events[-1]:
        switch_on_events = switch_on_events[:-1]
        if len(switch_on_events) == 0:
            return []
    print(len(switch_off_events))
    print(len(switch_on_events))
    assert len(switch_on_events) == len(switch_off_events)

    # Smooth over off-durations less than min_off_duration
    if min_off_duration > 0:
        off_durations = (chunk.index[switch_on_events[1:]].values -
                         chunk.index[switch_off_events[:-1]].values)

        off_durations = timedelta64_to_secs(off_durations)

        above_threshold_off_durations = np.where(
            off_durations >= min_off_duration)[0]

        # Now remove off_events and on_events
        switch_off_events = switch_off_events[
            np.concatenate([above_threshold_off_durations,
                            [len(switch_off_events)-1]])]
        switch_on_events = switch_on_events[
            np.concatenate([[0], above_threshold_off_durations+1])]
    assert len(switch_on_events) == len(switch_off_events)

    activations = []
    for on, off in zip(switch_on_events, switch_off_events):
        duration = (chunk.index[off] - chunk.index[on]).total_seconds()
        if duration < min_on_duration:
            continue
        on -= 1 + border
        if on < 0:
            on = 0
        off += border
        activation = chunk.iloc[on:off]
        # throw away any activation with any NaN values
        if not activation.isnull().values.any() and len(activation)>5:
            activations.append(activation)
            
    return activations