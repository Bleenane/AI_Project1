#
# kerasMNIST_CNN_sweep3.py
#
# Third search, to cut the training resources of the net in kerasMNIST_CNN.py.
#  --
#
# The project favours nets needing the least training and testing resources,
# suggests investigating Dropout, and recommends validation accuracy rather
# than validation loss as the early stopping metric.
#
# The timings from the second search showed that training time barely depends
# on the size of the net: 6, 8 and 12 slices all took about 8 s per epoch.
# Augmentation doubled it, from about 4 s per epoch.  So SLICES stays at 12, and
# this search tests shorter runs, Dropout in place of augmentation, and whether
# Batch Normalisation is needed.
#
# Step 1: run every candidate once, on the first seed.
# Step 2: taking candidates in order, the first with every seed at TARGET or
# above wins.  If there is none, the current net stays.  These rules were
# chosen in advance.
#
# Only validation data is used.  The testing set is thrown away on loading.
#

CANDIDATES = [(True,  0.1,  0.0,  0.0,  20),  # (Batch Normalisation, Dropout, ROTATION, SHIFT, EPOCHS)
              (True,  0.25, 0.0,  0.0,  20),  # Listed with the cheapest training first.
              (True,  0.25, 0.0,  0.0,  30),
              (False, 0.0,  0.02, 0.05, 20),
              (True,  0.0,  0.02, 0.05, 20),
              (True,  0.0,  0.02, 0.05, 30)]
SLICES   = 12
SEEDS    = [1, 2, 3]         # Every seed must reach TARGET.
TARGET   = 99.30             # Validation accuracy (%) required, as in the earlier searches.

RATE     = 0.002             # Starting learning rate, falls to 0 along a cosine curve.
SPLIT    = 0.2
SHUFFLE  = True
BATCH    = 64

import time

import numpy as np
import tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Activation
from tensorflow.keras.layers import RandomRotation, RandomTranslation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.utils import to_categorical, set_random_seed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend

print(tensorflow.__version__)
print(tensorflow.config.list_physical_devices('GPU')) # An empty list means no GPU.


# The net from kerasMNIST_CNN.py, with the options under test.

def build_model(batch_norm, dropout, rotation, shift, steps):
    model = Sequential()
    model.add(InputLayer(shape=(28,28,1)))
    if rotation > 0:
        model.add(RandomRotation(rotation, fill_mode='constant'))
    if shift > 0:
        model.add(RandomTranslation(shift, shift, fill_mode='constant'))

    for size, pool in [(SLICES, False), (SLICES, True),
                       (2*SLICES, False), (2*SLICES, True),
                       (4*SLICES, False)]:
        model.add(Conv2D(size, kernel_size=3, padding='same'))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(Activation('relu'))
        if pool:
            model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
            if dropout > 0:
                model.add(Dropout(dropout))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(10))
    model.add(Activation('softmax'))

    schedule = CosineDecay(RATE, decay_steps=steps)
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=schedule),
                  metrics=['accuracy'])
    return model


# Train one net, and return its best validation accuracy (%).  Patience equal
# to the epoch limit means that the run never stops early, and EarlyStopping is
# only used to restore the best weights at the end.

def run(candidate, seed):
    batch_norm, dropout, rotation, shift, epochs = candidate

    backend.clear_session()
    set_random_seed(seed)

    steps = epochs*int(np.ceil(len(training_images)*(1-SPLIT)/BATCH))
    model = build_model(batch_norm, dropout, rotation, shift, steps)
    best_weights = EarlyStopping(monitor='val_accuracy', mode='max', patience=epochs,
                                 restore_best_weights=True)

    start = time.time()
    history = model.fit(training_images, categorical_training_outputs,
                        epochs=epochs,
                        batch_size=BATCH,
                        shuffle=SHUFFLE,
                        validation_split=SPLIT,
                        verbose=0,
                        callbacks=[best_weights])
    elapsed = time.time()-start

    best = int(np.argmax(history.history['val_accuracy']))
    accuracy = history.history['val_accuracy'][best]*100
    print("BN={:5}  dropout={:.2f}  rotation={:.2f}  shift={:.2f}  epochs={:2d}  seed={}  weights={:6d}  best epoch={:2d}  validation={:6.2f}%  time={:4.0f}s  ({:.1f} s/epoch)".format(
          str(batch_norm), dropout, rotation, shift, epochs, seed, model.count_params(), best+1, accuracy, elapsed, elapsed/epochs),
          flush=True)
    return accuracy


(training_inputs, training_labels), _ = mnist.load_data()   # The testing set is thrown away here.

training_images = (training_inputs.astype('float32')/255)[:,:,:,np.newaxis]
categorical_training_outputs = to_categorical(training_labels)


# Step 1: every candidate gets one run, so that the report has a figure for each.

print("Step 1: every candidate, seed {}".format(SEEDS[0]))

scores = [run(candidate, SEEDS[0]) for candidate in CANDIDATES]


# Step 2: confirm on the other seeds, cheapest training first.

print("Step 2: confirming on the other seeds")

winner = None

for candidate, score in zip(CANDIDATES, scores):
    accuracies = [score]
    for seed in SEEDS[1:]:
        if accuracies[-1] >= TARGET:
            accuracies.append(run(candidate, seed))

    if len(accuracies) == len(SEEDS) and min(accuracies) >= TARGET:
        print("Passed: worst={:.2f}%  mean={:.2f}%".format(min(accuracies), np.mean(accuracies)), flush=True)
        winner = candidate
        break

if winner is None:
    print("RESULT: no candidate reached {:.2f}% on every seed.  Keep the current net.".format(TARGET))
else:
    print("RESULT: use Batch Normalisation = {}, Dropout = {}, ROTATION = {}, SHIFT = {}, EPOCHS = {}".format(*winner))
