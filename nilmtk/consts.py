import numpy as np

JOULES_PER_KWH = 3600000
SECS_PER_DAY = 86400
columns=[('power','apparent')]
LEVEL_NAMES=['physical_quantity', 'type']

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
