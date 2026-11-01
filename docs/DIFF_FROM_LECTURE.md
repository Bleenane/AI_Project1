# What Changed From the Lecture Example

This compares the lecturer's example with our final net ([code/Final_Model.ipynb](../code/Final_Model.ipynb)), and gives the reason for every change.

**In short:** we followed the lecturer's own suggestions to get past 99%, then replaced the large Dense layer to cut the weights, and cut the training time after the project hints defined "lightweight".

Of the lecturer's 82 lines of code, 54 are unchanged, 28 were removed and we added 56. Most of the comments were also rewritten.

## The two nets

**Lecture example:**

```text
Input --> Conv(10) --> Maxpool --> Flatten --> Dense(256) --> Dense(10, softmax)
```

**Final net:**

```text
Input --> Conv(12) --> Conv(12) --> Maxpool --> Dropout
      --> Conv(24) --> Conv(24) --> Maxpool --> Dropout
      --> Conv(48) --> Global Average Pooling --> Dense(10, softmax)
```

In the final net, each Conv is followed by Batch Normalisation and ReLU.

| Measure | Lecture example | Final net | Change |
|---|---:|---:|---|
| Test accuracy | About 98% | 99.34% | Now above the 99% requirement |
| Test errors | About 200 | 66 | About 3 times fewer |
| Weights | 504,686 | 20,638 | 24.5 times fewer |
| Multiplications per image | 0.57 million | 3.13 million | About 5.5 times more |
| Layers, not counting the input | 8 | 22 | 14 more |
| Convolutional layers | 1 | 5 | 4 more |
| Optimiser | Adam | RMSprop | Changed |
| Batch size | 32 | 64 | Doubled |
| Epochs | Up to 30 | Always 20 | Shorter limit, and no early halt |

The example's 98% comes from its own comments, as we never ran it. The multiplication counts are calculated from the layer sizes, not measured.

## How to read the reasons

Each change below says where it came from:

- **Lecturer:** the example's comments or the project hints told us to do it.
- **Tested:** one of our searches picked it, using validation data only.
- **Reasoned:** we chose it from theory, and did not test it on its own.
- **Project need:** the report or Colab required it.

## Changes that affect the net

### 1. More convolutional layers and slices

```diff
-model.add(Conv2D(10, kernel_size=3, padding='same'))
+model.add(Conv2D(SLICES, kernel_size=3, padding='same'))
+model.add(BatchNormalization())
+model.add(Activation('relu'))
+model.add(Conv2D(SLICES, kernel_size=3, padding='same'))
...
+model.add(Conv2D(2*SLICES, kernel_size=3, padding='same'))
+model.add(Conv2D(2*SLICES, kernel_size=3, padding='same'))
+model.add(Conv2D(4*SLICES, kernel_size=3, padding='same'))
```

**Why:** the example says it is "not good enough to achieve the full 99%+ accuracy", and it ends with this suggestion: "Try increasing the number of convolutional layers and the number of slices in the deeper convolutional layers (with maxpooling between the layers for dimensionality reduction)." Each maxpool halves the image, so the slices double to compensate.

**Source:** Lecturer for the idea. Tested for `SLICES` = 12, in the first and second searches. Reasoned for the fifth layer, which was added when the Dense layer went. The fourth search then Tested it: without that layer, no seed reached the 99.30% validation target, so it stayed.

### 2. Global Average Pooling replaces Flatten and Dense(256)

```diff
-model.add(Flatten())
-model.add(Dense(256))
-model.add(Activation('relu'))
+model.add(GlobalAveragePooling2D())
 model.add(Dense(10))
 model.add(Activation('softmax'))
```

**Why:** this one layer was the problem. Dense(256) holds 502,016 of the example's 504,686 weights, which is 99.5%. Global Average Pooling has no weights, and it leaves the output layer with 48 inputs and 490 weights.

**Source:** Reasoned. The saving is calculated, and the net still reached 99.34%, but we never compared the same net with and without it.

### 3. Batch Normalisation after each convolutional layer

```diff
+model.add(BatchNormalization())
```

**Why:** it rescales each layer's outputs, which helps a small net train reliably. It adds 5 layers and 480 weights.

**Source:** Reasoned. The third search then cast doubt on it: without Batch Normalisation the net scored 99.38%, against 99.31% with it. That is one seed, with augmentation on, so it isn't proof. We kept it because the search stopped at its first winner.

### 4. Dropout after each maxpool

```diff
+DROPOUT    = 0.1
...
 model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
+model.add(Dropout(DROPOUT))
```

**Why:** the example suggests "adding (judiciously) one or more Dropout layers", and the project hints say to "investigate using techniques like dropout". It adds no weights and is off during testing.

**Source:** Lecturer for the idea. Tested for the rate: the third search tried 0.1 and 0.25, and 0.1 won. Dropout also replaced random rotations and shifts, at half the time per epoch.

### 5. RMSprop replaces Adam

```diff
-OPT     = 'Adam'  # Adam optimizer.
...
-model.compile(loss='categorical_crossentropy', optimizer=OPT, metrics=['accuracy'])
+model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=schedule), metrics=['accuracy'])
```

**Why:** the example's notes warn that Adam "can converge to suboptimal solutions" and say "I have found RMSprop to be slightly better in practice". The example's code still uses Adam, even though a comment further down says RMSprop.

**Source:** Lecturer. We did not compare the two ourselves.

### 6. A cosine learning rate schedule

```diff
+RATE       = 0.002      # Starting learning rate, falls to 0 along a cosine curve.
...
+steps = EPOCHS*int(np.ceil(len(training_images)*(1-SPLIT)/BATCH))  # Training batches in the full run.
+schedule = CosineDecay(RATE, decay_steps=steps)
```

**Why:** the example uses a fixed rate, Adam's default of 0.001. A high rate makes quick progress but then bounces around the minimum, so a falling rate lets the net settle. The curve is fixed in advance, so a noisy validation score can't cut the rate too early.

**Source:** Tested, but not alone. The second search lifted `SLICES` = 12 from 99.24% to 99.42% on seed 1, but it changed the schedule and augmentation together. The starting value of 0.002 is Reasoned.

### 7. Batch size 32 to 64

```diff
-BATCH   = 32      # Minibatch size (note Keras default is 32).
+BATCH      = 64         # Minibatch size (note Keras default is 32).
```

**Why:** this gives 750 updates per epoch instead of 1,500, which should make each epoch quicker.

**Source:** Reasoned. It was set in v1 and never tested. No reason was logged at the time, so this is our best explanation.

### 8. Epochs 30 to 20

```diff
-EPOCHS  = 30      # Training run parameters.  30 epoch run limit.
+EPOCHS     = 20         # Training run parameters.  Every run takes all 20 epochs.
```

**Why:** the project favours "the least training and testing resources". Going from 20 to 30 epochs added only 0.04 points with Dropout, for 50% more time.

**Source:** Tested, in the third search.

### 9. Early stopping now picks the best weights, and never halts the run

```diff
-stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3,
-                     verbose=2, mode='auto',
-                     restore_best_weights=True)
+best_weights = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=EPOCHS,
+                             verbose=2, mode='max',
+                             restore_best_weights=True)
...
-                    callbacks=[stop])
+                    callbacks=[best_weights])
```

| Setting | Why it changed |
|---|---|
| `monitor='val_accuracy'` | The hints say: "I recommend using "validation accuracy" rather than "validation loss" as a metric for early-stopping" |
| `mode='max'` | Higher accuracy is better, the opposite of loss |
| `patience=EPOCHS` | Halting early would cut off the cosine schedule before the rate falls, and the best epochs come late (epoch 16 of 20) |
| `best_weights` | A new name, because it no longer stops anything |

**Source:** Lecturer for the metric. Reasoned for the patience.

**Be clear about this in the report:** we use the lecturer's metric, but the run never stops early. Epochs 17 to 20 gave no improvement, so that time was spent for nothing.

### 10. Random seeds

```diff
-np.random.seed(1)                    # Initialise system RNG.
-
 import tensorflow
-tensorflow.random.set_seed(2)        # and the seed of the Tensorflow backend.
+from tensorflow.keras.utils import set_random_seed
+
+set_random_seed(1)                   # Seeds Python, NumPy and the Tensorflow backend, as in the search.
```

**Why:** one call seeds Python, NumPy and TensorFlow, and it matches the searches.

**Source:** Reasoned. It did not make runs identical: seed 1 gave 99.38% in the search and 99.42% in the final run.

## Changes to what gets reported or saved

None of these change the net.

### 11. Report the best epoch, not the last

```diff
-print("Performance of network:")
-print("Accuracy on training data:   {:6.2f}%".format(history.history['accuracy'][-1]*100))
-print("Accuracy on validation data: {:6.2f}%".format(history.history['val_accuracy'][-1]*100))
+best = int(np.argmax(history.history['val_accuracy']))
+
+print("Performance of network (best epoch was {}):".format(best+1))
+print("Accuracy on training data:   {:6.2f}%".format(history.history['accuracy'][best]*100))
+print("Accuracy on validation data: {:6.2f}%".format(history.history['val_accuracy'][best]*100))
```

**Why:** the net keeps the weights from the best epoch, but the example prints the last epoch's figures. Those wouldn't describe the net being tested. In our run the last epoch shows 99.40% validation accuracy, while the kept weights have 99.42%.

**Source:** Project need.

### 12. An accuracy plot

```diff
+plt.figure('Training and Validation Accuracy per epoch', figsize=(8,8))
+
+plt.plot(history.history['accuracy'],label='training') # Training data accuracy per epoch.
+
+plt.plot(history.history['val_accuracy'],label='validation') # Validation accuracy per ep.
...
+plt.savefig('kerasMNIST_CNN_accuracy.png')
```

**Why:** section 3.1 of the report template needs accuracy per epoch, and the example only plots loss.

**Source:** Project need.

### 13. Plots saved to files

```diff
+plt.savefig('kerasMNIST_CNN_loss.png')
```

**Why:** the report needs the plots as image files.

**Source:** Project need.

### 14. The trained net saved to a file

```diff
+model.save('kerasMNIST_CNN.keras')
```

**Why:** it lets us check the result with no retraining, because runs aren't exactly repeatable. The confusion matrix and the errors notebook both depend on it.

**Source:** Project need.

### 15. A GPU printout

```diff
+print(tensorflow.config.list_physical_devices('GPU')) # An empty list means no GPU.
```

**Why:** the lecturer wants a T4 GPU, and the first Colab log didn't show the hardware. An empty list means training has fallen back to the CPU.

**Source:** Project need.

## Tidying, with no effect on the result

### 16. Sample images removed

```diff
-plt.figure(figsize=(14,4))
-for i in range(20):
-    plt.subplot(2,10,i+1)
-    plt.imshow(training_images[i,:,:,0],cmap='gray')
-    plt.title(str(training_labels[i]))
-    plt.axis('off')
```

This displayed the first 20 training images. It only affects the display. We removed it in v1 and didn't log the reason, and it didn't need to go.

### 17. Imports moved

`EarlyStopping` and `mnist` moved up to join the other imports. `Flatten` went, as the net no longer uses it. `GlobalAveragePooling2D`, `BatchNormalization`, `Dropout`, `RMSprop` and `CosineDecay` are new, for the changes above.

### 18. Comments rewritten

- **Header and settings:** rewritten to describe our net and name the searches that chose each value.
- **Teaching comments:** the lecturer's long explanations were shortened. These cover the 4-tensor shape, `compile`, `fit` and early stopping.
- **Model comment:** replaced with reasons for our layers, as the original described a net with one convolutional layer.
- **Out-of-date lines removed:** "We get about 98% accuracy" and the closing suggestions, which we followed.
- **Errors fixed:** `Returns s 2-tuple (row,cols)` now says it returns a 3-tuple, and "both the testing and the validation data" now says training, which is what the plot shows.

## What did not change

- **Data:** loading MNIST, scaling to 0-1, the `(N,28,28,1)` shape and `to_categorical`.
- **Split:** `SPLIT = 0.2` and `SHUFFLE = True`.
- **Loss:** `categorical_crossentropy`.
- **Layer settings:** 3 × 3 filters, `padding='same'`, 2 × 2 maxpooling, ReLU and softmax.
- **Structure:** `Sequential`, the `NEW_KERAS` check, and the `model.fit` and `model.evaluate` calls.
- **Output:** the loss plot code and the test accuracy printouts.

## Tried and removed, so not in the diff

The diff only compares the first and last versions. These were part of the work, but aren't in the final code:

| Tried | When | Why it went |
|---|---|---|
| `ReduceLROnPlateau` | v1 and the first search | The cosine schedule replaced it |
| Random rotation and shift | v2 | They doubled the training time, and Dropout matched them at half the time |
| Removing Dropout altogether | v2 | This was never tested, and the third search put Dropout back |
| 40 epochs | v2 | The third search showed that 20 were enough |

v1 and v2 are not in this repo, but `CHANGES.md` describes both.

## Where the final net is not lighter

- **More arithmetic per image:** our net does about 5.5 times more multiplications than the lecture example, despite having 24.5 times fewer weights. A convolutional filter reuses its weights at every position in the image, so fewer weights doesn't always mean less arithmetic. Calculated, not measured.
- **More layers:** 22 against 8. The hints warn against "unnecessary extra layers". The fourth search showed that our net needs the fifth convolutional layer to reach its validation target, but Batch Normalisation was never shown to be needed.
- **What this means:** the example doesn't reach 99%, so it isn't a valid alternative. The fair lightweight comparison is against other nets above 99%, such as our v1, which needs about 6 times more multiplications per image and 42 times more weights.
