# What Changed From v1

This compares our first net, v1, with our final net ([code/Final_Model.ipynb](../code/Final_Model.ipynb)), and gives the reason for every change. v1's code is not in this repo, so the lines that changed are quoted below. The last section covers what v1 changed from the lecture example.

**In short:** v1 proved that 99% was reachable, but it was far too big. We replaced its large Dense layer, narrowed its convolutional layers, then cut the training time. That cost 22 or 23 test errors out of 10,000.

Of v1's 103 lines of code, 75 are unchanged, 28 were removed and we added 35.

## The two nets

**v1:**

```text
Input --> Conv(32) --> Conv(32) --> Maxpool --> Dropout(0.25)
      --> Conv(64) --> Conv(64) --> Maxpool --> Dropout(0.25)
      --> Flatten --> Dense(256) --> Dropout(0.5) --> Dense(10, softmax)
```

**Final net:**

```text
Input --> Conv(12) --> Conv(12) --> Maxpool --> Dropout(0.1)
      --> Conv(24) --> Conv(24) --> Maxpool --> Dropout(0.1)
      --> Conv(48) --> Global Average Pooling --> Dense(10, softmax)
```

In the final net, each Conv is followed by Batch Normalisation and ReLU. In v1, each Conv is followed by ReLU only.

| Measure | v1 | Final net | Change |
|---|---:|---:|---|
| Test accuracy | 99.56%, 99.57%, 99.57% | 99.34% | 0.22 to 0.23 points lower |
| Test errors | 44, 43, 43 | 66 | 22 or 23 more |
| Weights | 870,634 | 20,638 | 42 times fewer |
| Weight storage | 3.32 MB | 80.62 KB | 42 times smaller |
| Multiplications per image | 19.1 million | 3.13 million | 6 times fewer |
| Training time | About 2 minutes | About 82 s | About 1.5 times faster |
| Epochs run | 22, 26, 22 | 20 | Fewer, and always the same |
| Time per epoch | About 5 s | About 4 s | Quicker |
| Layers, not counting the input | 18 | 22 | 4 more |
| Convolutional layers | 4 | 5 | 1 more |

The multiplication counts are calculated from the layer sizes, not measured. v1 has three test results because it had three full Colab runs. The third was a rerun for context, made after the final net was chosen, and it played no part in any choice.

## Where the savings came from

Two different changes gave the two main savings:

| Saving | From removing Dense(256) | From the narrower convolutional layers |
|---|---:|---:|
| Weights | 94% of the saving | 6% of the saving |
| Multiplications | 5% of the saving | 95% of the saving |

v1's weights sat in its Dense layer, but its arithmetic sat in its convolutional layers. So removing the Dense layer alone would have left the net almost as slow per image.

## How to read the reasons

Each change below says where it came from:

- **Lecturer:** the example's comments or the project hints told us to do it.
- **Tested:** one of our searches picked it, using validation data only.
- **Reasoned:** we chose it from theory, and did not test it on its own.
- **Project need:** the report or Colab required it.

## Changes that affect the net

### 1. Global Average Pooling replaces Flatten and Dense(256)

```diff
-model.add(Flatten())
-model.add(Dense(256))
+model.add(Conv2D(4*SLICES, kernel_size=3, padding='same'))
+model.add(BatchNormalization())
 model.add(Activation('relu'))
-model.add(Dropout(0.5))
+
+model.add(GlobalAveragePooling2D())
 model.add(Dense(10))
```

**Why:** the project favours lightweight nets, and this was the obvious first target. Dense(256) held 803,072 of v1's 870,634 weights, which is 92%. Global Average Pooling has no weights, and the output layer falls from 2,570 weights to 490.

**Source:** Reasoned. The saving is calculated, and the net still passed 99%, but we never compared the same net with and without it.

### 2. Narrower convolutional layers, set by one number

```diff
-    model.add(Conv2D(32, kernel_size=3, padding='same'))
+    model.add(Conv2D(SLICES, kernel_size=3, padding='same'))
...
-model.add(Conv2D(64, kernel_size=3, padding='same'))
+model.add(Conv2D(2*SLICES, kernel_size=3, padding='same'))
```

**Why:** this gave most of the saving in arithmetic. Slices fell from 32, 32, 64, 64 to 12, 12, 24, 24. A single `SLICES` setting scales the whole net, so a search only has to try one number.

**Source:** Tested. The first search found no size that passed. The second search picked `SLICES` = 12, with 99.42%, 99.42% and 99.37% on 3 seeds. Sizes 6 and 8 failed, and sizes 9 to 11 were not tried.

### 3. A fifth convolutional layer

```diff
+model.add(Conv2D(4*SLICES, kernel_size=3, padding='same'))
```

**Why:** it replaces some of the capability lost with the Dense layer. It adds 10,608 weights including its Batch Normalisation, which is about half of the final net. It also lets Global Average Pooling start from 48 slices rather than 24.

**Source:** Reasoned at first, then Tested. The hints warn against "unnecessary extra layers", so the fourth search removed this layer and kept everything else. Validation accuracy fell from 99.38%, 99.37% and 99.33% to 99.17%, 98.96% and 99.02%, so no seed reached the 99.30% target and the layer stayed. Training time barely changed.

### 4. Batch Normalisation added

```diff
+model.add(BatchNormalization())
 model.add(Activation('relu'))
```

**Why:** it rescales each layer's outputs, which helps a small net train reliably. v1 was large enough to train without it. It adds 5 layers and 480 weights.

**Source:** Reasoned. The third search cast doubt on it: without it the net scored 99.38%, against 99.31% with it. That is one seed with augmentation on, so it isn't proof.

### 5. Dropout reduced

```diff
-model.add(Dropout(0.25))
+model.add(Dropout(DROPOUT))
...
-model.add(Dropout(0.25))
+model.add(Dropout(DROPOUT))
...
-model.add(Dropout(0.5))
```

**Why:** the 0.5 went with Dense(256), because it was there to protect that layer. The route for the other two wasn't direct: the first lightweight net removed Dropout without testing it, and the project hints then made us test it, which brought it back at 0.1.

**Source:** Tested. The third search tried 0.1 and 0.25 without augmentation, and 0.1 won because it came first and passed. Both scored about the same.

### 6. A cosine schedule replaces ReduceLROnPlateau

```diff
-OPT        = 'RMSprop'  # RMSprop optimizer.
+RATE       = 0.002      # Starting learning rate, falls to 0 along a cosine curve.
...
-LR_FACTOR   = 0.5       # Multiply learning rate by this when val_loss stalls...
-LR_PATIENCE = 2         # ...for this many epochs.
...
-reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=LR_FACTOR,
-                              patience=LR_PATIENCE, verbose=2)
...
-model.compile(loss='categorical_crossentropy', optimizer=OPT, metrics=['accuracy'])
+steps = EPOCHS*int(np.ceil(len(training_images)*(1-SPLIT)/BATCH))  # Training batches in the full run.
+schedule = CosineDecay(RATE, decay_steps=steps)
+
+model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=schedule), metrics=['accuracy'])
```

**Why:** v1 started at RMSprop's default of 0.001, and halved the rate whenever validation loss stalled for 2 epochs. A small net's validation loss is noisy, so a stall can cut the rate too early. A cosine curve is fixed in advance, so noise can't affect it. The optimiser is still RMSprop, and `OPT` went because a schedule has to be passed in as an object.

**Source:** Tested, but not alone. The second search lifted `SLICES` = 12 from 99.24% to 99.42% on seed 1, but it changed the schedule and augmentation together. The starting value of 0.002 is Reasoned.

### 7. Fewer epochs, and every run takes all of them

```diff
-EPOCHS     = 50         # Training run parameters.  50 epoch run limit.
+EPOCHS     = 20         # Training run parameters.  Every run takes all 20 epochs.
```

**Why:** the project favours "the least training and testing resources". v1's limit of 50 was only an upper bound, and it stopped at epochs 22, 26 and 22. The cosine schedule needs a known length, so the run now has a fixed 20 epochs.

**Source:** Tested, in the third search.

### 8. Early stopping picks the best weights, and never halts the run

```diff
-STOP_PATIENCE = 7       # Stop after this many epochs without improvement.  Must
-                        # exceed LR_PATIENCE, or the lower rate never gets a chance.
...
-stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=STOP_PATIENCE,
-                     verbose=2, mode='auto',
-                     restore_best_weights=True)
+best_weights = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=EPOCHS,
+                             verbose=2, mode='max',
+                             restore_best_weights=True)
...
-                    callbacks=[reduce_lr, stop])
+                    callbacks=[best_weights])
```

**Why:** the metric follows the hint: "I recommend using "validation accuracy" rather than "validation loss" as a metric for early-stopping". The patience now equals the epoch limit, because halting early would cut the cosine schedule off before the rate falls.

**Source:** Lecturer for the metric. Reasoned for the patience.

**Be clear about this in the report:** v1 really did stop early. The final net doesn't, and epochs 17 to 20 gave no improvement.

### 9. Random seeds

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

**Source:** Reasoned. It did not make runs identical.

## Changes to what gets reported

None of these change the net.

### 10. The best epoch is found by accuracy

```diff
-best = int(np.argmin(history.history['val_loss']))
+best = int(np.argmax(history.history['val_accuracy']))
```

**Why:** it has to match the metric in change 8, or the printed figures wouldn't describe the kept weights.

**Source:** Project need.

### 11. A GPU printout

```diff
+print(tensorflow.config.list_physical_devices('GPU')) # An empty list means no GPU.
```

**Why:** the lecturer wants a T4 GPU, and v1's log didn't show the hardware.

**Source:** Project need.

## Tidying, with no effect on the result

- **Imports:** `Flatten` and `ReduceLROnPlateau` went. `GlobalAveragePooling2D`, `BatchNormalization`, `RMSprop` and `CosineDecay` are new.
- **Header:** rewritten to describe the lightweight net, with the project's definition of lightweight.
- **Model comment:** now explains Global Average Pooling and Batch Normalisation, and no longer explains the 0.5 Dropout.
- **Spelling:** "dropout" became "Dropout" in one comment.

## What did not change

- **Training:** RMSprop, batch size 64, `SPLIT = 0.2`, `SHUFFLE = True` and `categorical_crossentropy`.
- **Data:** loading, scaling, the `(N,28,28,1)` shape and `to_categorical`.
- **Layer settings:** 3 × 3 filters, `padding='same'`, 2 × 2 maxpooling, ReLU and softmax.
- **Output:** both plots, saving them to files, the test evaluation and `model.save`.

## Where v1 is still better

- **Accuracy:** v1 scored higher on all three of its runs, with 22 or 23 fewer test errors. That is the real cost of going lightweight.
- **Simplicity:** v1 has 18 layers against 22, with no Batch Normalisation and no fifth convolutional layer.
- **Stopping:** v1 stops when it has stopped improving. The final net always takes all 20 epochs.

We chose the final net because both nets pass the 99% requirement, and the lecturer says lightweight solutions will be favoured. The hints don't say how much accuracy is worth trading for that, so this is a judgement, and the report should say so.

## What v1 changed from the lecture example

`DIFF_FROM_LECTURE.md` compares the lecture example with the final net. This table covers the step before that.

| Setting | Lecture example | v1 | Why | Source |
|---|---|---|---|---|
| Convolutional layers | 1 | 4 | The example's closing suggestion | Lecturer |
| Slices | 10 | 32, 32, 64, 64 | More slices in deeper layers, as suggested | Lecturer, with sizes Reasoned |
| Maxpool layers | 1 | 2 | "Maxpooling between the layers" | Lecturer |
| Dropout | None | 0.25, 0.25, 0.5 | "Adding (judiciously) one or more Dropout layers" | Lecturer, with rates Reasoned |
| Optimiser | Adam | RMSprop | "I have found RMSprop to be slightly better in practice" | Lecturer |
| Learning rate | Fixed, 0.001 | Halved when validation loss stalls | Lets the net settle near the minimum | Reasoned |
| Batch size | 32 | 64 | Half as many updates per epoch. No reason was logged at the time, so this is our best explanation | Reasoned |
| Epoch limit | 30 | 50 | Leaves room to keep training at a lower rate. No reason was logged at the time, so this is our best explanation | Reasoned |
| Stopping patience | 3 | 7 | Has to exceed the learning rate patience of 2 | Reasoned |
| Printed figures | Last epoch | Best epoch | Must describe the kept weights | Project need |
| Plots and saved net | Loss plot only | Accuracy plot added, plots and net saved | The report needs them | Project need |
| Sample images | Shown | Removed | No effect, and it didn't need to go | Tidying |

| Measure | Lecture example | v1 |
|---|---:|---:|
| Test accuracy | About 98% (its own comment, not measured by us) | 99.56%, 99.57%, 99.57% |
| Weights | 504,686 | 870,634 |
| Multiplications per image | 0.57 million | 19.1 million |

v1 reached the target by following the lecturer's suggestions, but it grew by 1.7 times in weights to do it. That is what made the lightweight rebuild necessary.
