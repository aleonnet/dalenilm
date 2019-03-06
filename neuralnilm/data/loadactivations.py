from __future__ import print_function, division
from copy import copy
import numpy as np

#import nilmtk
from neuralnilm.utils import check_windows
from nilmtk.open_dataset import load_appliance_dataset
from nilmtk.set_time_windows import set_window
from nilmtk.get_activations import get_app_activations
from nilmtk.sample_data import safe_resample
import logging
logger = logging.getLogger(__name__)


def load_nilmtk_activations(appliances, filename, sample_period, windows):
    """
    Parameters
    ----------
    appliances : list of strings
    filename : string
    sample_period : int
    windows : dict
        Structure example:
        {
            'train': {<building_i>: <window>},
            'unseen_activations_of_seen_appliances': {<building_i>: <window>},
            'unseen_appliances': {<building_i>: <window>}
        }

    Returns
    -------
    all_activations : dict
        Structure example:
        {<train | unseen_appliances | unseen_activations_of_seen_appliances>: {
             <appliance>: {
                 <building_name>: [<activations>]
        }}}
        Each activation is a pd.Series with DatetimeIndex and the following
        metadata attributes: building, appliance, fold.
    """
    logger.info("Loading NILMTK activations...")

    # Sanity check
    check_windows(windows)

    # Load dataset
    #dataset = nilmtk.DataSet(filename)

    all_activations = {}
    for fold, buildings_and_windows in windows.items():
        activations_for_fold = {}
        for building_i, window in buildings_and_windows.items():
            dataset=load_appliance_dataset(int(building_i))
            dataset=set_window(dataset,window[0],window[1])
            #elec = dataset.buildings[building_i].elec
            building_name = (
                'REDD' + '_building_{}'.format(building_i))
            for appliance in appliances:
                logger.info(
                    "Loading {} for {}...".format(appliance, building_name))

                # Get meter for appliance
                #try:
                meter = dataset
                resample_kwargs={}
                resample_kwargs['rule'] = '{:d}S'.format(sample_period)
                meter=meter['power']['apparent']
                meter=safe_resample(meter,resample_kwargs)
                meter=meter.agg(np.mean)
                meter=meter.fillna(method='ffill')
                #except KeyError as exception:
                 #   logger.info(building_name + " has no " + appliance +
                  #              ". Full exception: {}".format(exception))
                   # continue

                # Get activations_for_fold and process them
                meter_activations = get_app_activations(meter)
                meter_activations = [activation.astype(np.float32)
                                     for activation in meter_activations]

                # Attach metadata
                for activation in meter_activations:
                    activation._metadata = copy(activation._metadata)
                    activation._metadata.extend(
                        ["building", "appliance", "fold"])
                    activation.building = building_name
                    activation.appliance = appliance
                    activation.fold = fold

                # Save
                if meter_activations:
                    activations_for_fold.setdefault(
                        appliance, {})[building_name] = meter_activations
                logger.info(
                    "Loaded {} {} activations from {}."
                    .format(len(meter_activations), appliance, building_name))
        all_activations[fold] = activations_for_fold

    #dataset.store.close()
    logger.info("Done loading NILMTK activations.")
    return all_activations
