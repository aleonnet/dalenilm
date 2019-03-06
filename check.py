from __future__ import print_function
#import nilmtk
#from nilmtk.utils import print_dict
#from nilmtk import DataSet

#from nilmtk.dataset_converters import convert_redd
#convert_redd('./data/redd', './data/redd.h5')

#dataset = DataSet('./data/redd.h5')
#dataset.set_window("2011-04-01", "2011-05-01")

#import nilmtk
#from nilmtk.utils import print_dict
#from nilmtk import DataSet
from neuralnilm.data.loadactivations import load_nilmtk_activations
from neuralnilm.data.syntheticaggregatesource import SyntheticAggregateSource
from neuralnilm.data.realaggregatesource import RealAggregateSource
#from neuralnilm.data.stridesource import StrideSource
from neuralnilm.data.datapipeline import DataPipeline
from neuralnilm.data.processing import DivideBy, IndependentlyCenter

# ------------
def select_windows(train_buildings, unseen_buildings):
    windows = {fold: {} for fold in DATA_FOLD_NAMES}

    def copy_window(fold, i):
        windows[fold][i] = WINDOWS[fold][i]

    for i in train_buildings:
        copy_window('train', i)
        copy_window('unseen_activations_of_seen_appliances', i)
    for i in unseen_buildings:
        copy_window('unseen_appliances', i)
    return windows

def filter_activations(windows, activations):
    new_activations = {
        fold: {appliance: {} for appliance in APPLIANCES}
        for fold in DATA_FOLD_NAMES}
    for fold, appliances in activations.items():
        for appliance, buildings in appliances.items():
            required_building_ids = windows[fold].keys()
            required_building_names = [
                'UK-DALE_building_{}'.format(i) for i in required_building_ids]
            for building_name in required_building_names:
                try:
                    new_activations[fold][appliance][building_name] = (
                        activations[fold][appliance][building_name])
                except KeyError:
                    pass
    return activations
    
NILMTK_FILENAME = 'redd'
SAMPLE_PERIOD = 6
STRIDE = None
APPLIANCES = ['fridge']
WINDOWS = {
    'train': {
        #1: ("2012-12-14", "2013-03-14"),
        2: ("2013-05-20", "2013-08-20"),
        4: ("2013-03-09", "2013-06-09"),
        #6: ("2011-05-22", "2011-06-14"),
    },
    'unseen_activations_of_seen_appliances': {
        #1: ("2012-12-14", None),
        2: ("2013-05-20", None),
        4: ("2013-03-09", None),
        #6: ("2011-05-22", None),
    },
    'unseen_appliances': {
        5: ("2014-06-29", None)
    }
}

activations = load_nilmtk_activations(
    appliances=APPLIANCES,
    filename=NILMTK_FILENAME,
    sample_period=SAMPLE_PERIOD,
    windows=WINDOWS
)

num_seq_per_batch = 16
target_appliance = 'fridge'
seq_length = 512
train_buildings = [2,4]
unseen_buildings = [5]
DATA_FOLD_NAMES = (
    'train', 'unseen_appliances', 'unseen_activations_of_seen_appliances')



filtered_windows = select_windows(train_buildings, unseen_buildings)
filtered_activations = filter_activations(filtered_windows, activations)

synthetic_agg_source = SyntheticAggregateSource(
    activations=filtered_activations,
    target_appliance=target_appliance,
    seq_length=seq_length,
    sample_period=SAMPLE_PERIOD
)

real_agg_source = RealAggregateSource(
    activations=filtered_activations,
    target_appliance=target_appliance,
    seq_length=seq_length,
    filename=NILMTK_FILENAME,
    windows=filtered_windows,
    sample_period=SAMPLE_PERIOD
)

# rescaling is done using the a first batch of num_seq_per_batch sequences
sample = real_agg_source.get_batch(num_seq_per_batch=1024).__next__()
sample = sample.before_processing
input_std = sample.input.flatten().std()
target_std = sample.target.flatten().std()


pipeline = DataPipeline(
    [synthetic_agg_source, real_agg_source],
    num_seq_per_batch=num_seq_per_batch,
    input_processing=[DivideBy(input_std), IndependentlyCenter()],
    target_processing=[DivideBy(target_std)]
)

import numpy as np
num_test_seq = 101

X_valid = np.empty((num_test_seq*num_seq_per_batch, seq_length))
Y_valid = np.empty((num_test_seq*num_seq_per_batch, 3))

for i in range(num_test_seq):
    (x_valid,y_valid) = pipeline.train_generator(fold = 'unseen_appliances', source_id = 1).__next__()
    X_valid[i*num_seq_per_batch: (i+1)*num_seq_per_batch,:] = x_valid[:,:,0]
    Y_valid[i*num_seq_per_batch:  (i+1)*num_seq_per_batch,:] = y_valid
X_valid = np.reshape(X_valid, [X_valid.shape[0],X_valid.shape[1],1])

# needed to rescale the input aggregated data
# rescaling is done using the a first batch of num_seq_per_batch sequences
sample = real_agg_source.get_batch(num_seq_per_batch=1024).__next__()
sample = sample.before_processing
input_std = sample.input.flatten().std()
target_std = sample.target.flatten().std()

def scores(Y_pred, Y_test, activation_threshold = 0.1 ,plot_results= True,  print_results = False):

    """
    a function that computes the classification scores with various metrics
    return: dictionary with the various scores

    """

    # post process the data

    np.putmask(Y_pred[:,0], Y_pred[:,0] <=0, 0)
    np.putmask(Y_pred[:,1], Y_pred[:,1] >=1, 1)
    np.putmask(Y_pred[:,0],Y_pred[:,1] < Y_pred[:,0],0)
    np.putmask(Y_pred[:,1],Y_pred[:,1] < Y_pred[:,0],0)
    np.putmask(Y_pred[:,1],Y_pred[:,2] < activation_threshold,0)
    np.putmask(Y_pred[:,0],Y_pred[:,2] < activation_threshold,0)    

    # find negative in prediction
    pred_negatives = (Y_pred[:,0] ==0) &(Y_pred[:,1] ==0)
    pred_positives = ~pred_negatives
    obs_negatives = (Y_test[:,0] ==0) &(Y_test[:,1] ==0)
    obs_positives = ~obs_negatives
    TP = obs_positives[pred_positives].sum()
    FN = obs_positives[pred_negatives].sum()
    TN = obs_negatives[pred_negatives].sum()
    FP = obs_negatives[pred_positives].sum()

    recall = TP / float(TP + FN)
    precision = TP / float(TP+ FP)
    f1 = 2* precision*recall / (precision + recall)
    accuracy = (TP + TN)/ float(obs_negatives.sum() +obs_positives.sum() )
    if print_results:
        print('number of Predicted negatives:',pred_negatives.sum() )
        print('number of Predicted positives:',pred_positives.sum() )
        print('number of Observed negatives:', obs_negatives.sum() )
        print('number of Observed positives:', obs_positives.sum() )
        print('f1:',  f1)
        print('precision :' ,precision)
        print('recall : ', recall)
        print('accuracy:', accuracy)

    results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall_score': recall}
    if plot_results:
        pd_results = pd.DataFrame.from_dict(results, orient = 'index')
        pd_results = pd_results.transpose()    
        sns.barplot(data = pd_results)

    return results

def _get_output_neurons(self, new_batch):
    batch_size = new_batch.target.shape[0]
    neural_net_output = np.empty((batch_size, 3))
    for b in range(batch_size):
        seq =  new_batch.target[b]
        if seq[0] > 0:
            start = 0
            stop_array = np.where(seq > 0)[0]
            if len(stop_array) == 0:
                stop = seq[-1]
            else:
                stop = stop_array[-1]  
                # calculate avg power
                avg_power =  np.mean(seq[start:stop + 1])

            # case 3: signal starts after 0 and before 1
        else:
            start_array = np.where(seq > 0)[0]
            if len(start_array) == 0:
                start = 0
                stop = 0
                avg_power = 0
            else:
                start = start_array[0]
                stop_array = np.where(seq > 0)[0]
                if len(stop_array) == 0:
                    stop = seq[-1]
                else:
                    stop = stop_array[-1]        
                avg_power =  np.mean(seq[start:stop + 1])
                    
        start = start / float(new_batch.target.shape[1] - 1)
        stop = stop  / float(new_batch.target.shape[1] - 1)
        if stop < start:
            raise ValueError("start must be before stop in sequence {}".format(b))

            neural_net_output[b, :] = np.array([start, stop, avg_power])
        
        
    return neural_net_output




# import Keras related libraries
from keras.layers import Input, Dense, Flatten, MaxPooling1D, AveragePooling1D, Convolution1D
from keras.models import Model
import keras.callbacks
from keras.callbacks import ModelCheckpoint
import time
from keras.models import model_from_json
import pickle

# ------------
exp_number = 13
output_architecture = './tmpdata/convnet_architecture_exp' + str(exp_number) + '.json'
best_weights_during_run = './tmpdata/weights_exp' + str(exp_number) + '.h5'
final_weights = './tmpdata/weights_exp' + str(exp_number) + '_final.h5'
loss_history = './tmpdata/history_exp' + str(exp_number) + '.pickle'
# ------------


# ------------
# a class used to record the training and validation loss 
# at the end of each epoch

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.train_losses = [] 
        self.valid_losses = []

    def on_epoch_end(self, epoch, logs = {}):
        self.train_losses.append(logs.get('loss'))
        self.valid_losses.append(logs.get('val_loss'))
        
# ------------        
# input sequence
input_seq = Input(shape = (seq_length, 1))
# first convolutional layer
conv1_layer =  Convolution1D(nb_filter = 16, filter_length = 3, border_mode='valid',
                      init = 'normal', activation =  'relu')
conv1 = conv1_layer(input_seq)
# flatten the weights
flat = Flatten()(conv1)
# first dense layer
dense1 = Dense(1024, activation = 'relu')(flat)
# second dense layer
dense2 = Dense(512, activation = 'relu', init= 'normal')(dense1)
# output layer
predictions = Dense(3, activation = 'linear')(dense2)   
# create the model
model = Model(input=input_seq, output=predictions)
# compile the model -- define the loss and the optimizer
model.compile(loss='mean_squared_error', optimizer='Adam')
# record the loss history
history = LossHistory()
# save the weigths when the vlaidation lost decreases only
checkpointer = ModelCheckpoint(filepath=best_weights_during_run, save_best_only=True, verbose =1 )
# fit the network using the generator of mini-batches.
model.fit_generator(pipeline.train_generator(fold = 'train'), \
                    samples_per_epoch = 30000, \
                    nb_epoch = 20, verbose = 1, callbacks=[history, checkpointer],
                   validation_data = (x_valid,y_valid), max_q_size = 50)
losses_dic = {'train_loss': history.train_losses, 'valid_loss':history.valid_losses}
# save history
losses_dic = {'train_loss': history.train_losses, 'valid_loss':history.valid_losses}
with open(loss_history, 'wb') as handle:
  pickle.dump(losses_dic, handle)

print('\n saving the architecture of the model \n')
json_string = model.to_json()
open(output_architecture, 'w').write(json_string)

print('\n saving the final weights ... \n')
model.save_weights(final_weights, overwrite = True)
print('done saving the weights')

print('\n saving the training and validation losses')

print('This was the model trained')
print(model.summary())




