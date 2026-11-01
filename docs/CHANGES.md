# Project 1: Code Changes

> **In this repo:** this log covers the whole project, so it names files that are not here, such as v1, v2, the other searches and `Colab Download/`. The final script, confusion matrix and new digits test are now together in `code/Final_Model.ipynb`.

**Status at 2026-11-01: finished.** The final net scored 99.34% on the test set with 20,638 weights and about 82 s of training. It matches the first lightweight net's test score, with training about 4 times faster. The report can now be filled in from `REPORT_NOTES.md`. The fourth search then tested removing the fifth convolutional layer. No seed reached the target, so the final net did not change. On 50,000 new digits from QMNIST, the saved net averaged 99.23%.

## Results so far

| Version | Weights | Saved size | Test accuracy | Test errors | Status |
|---|---:|---:|---:|---:|---|
| v1 (larger net) | 870,634 | 3.32 MB | 99.56%, 99.57% and 99.57% | 44, 43 and 43 of 10,000 | Three full Colab logs, same code and seeds |
| Lightweight v2 (`SLICES` = 12) | 20,638 | 80.62 KB | 99.34% | 66 of 10,000 | One Colab run, 319 s of training |
| Final (Dropout, 20 epochs) | 20,638 | 80.62 KB | 99.34% | 66 of 10,000 | One Colab run, about 82 s of training |

Saved size here is the size of the weights as reported by Keras, not the size of the `.keras` file.

## Files

| File | Purpose | Status |
|---|---|---|
| `kerasMNIST_CNN.py` | Final script | Done, scored 99.34%. Do not rerun |
| `Colab Download/` | Saved final net and the raw Colab plots | The saved net reproduces the confusion matrix exactly |
| `Colab Download/v1/` | Saved v1 net and plots from a later rerun, renamed so they cannot replace the final net | For context only, not used to choose anything |
| `kerasMNIST_CNN_v2.py` | Copy of the first lightweight net, which also scored 99.34% | Kept for comparison |
| `DIFF_FROM_LECTURE.md` | Every change from the lecture example, with the reason | Done |
| `kerasMNIST_CNN_vs_lecture.diff` | Raw line-by-line diff against the lecture example | Done |
| `DIFF_FROM_V1.md` | Every change from v1, with the reason, and what v1 changed from the lecture example | Done |
| `kerasMNIST_CNN_vs_v1.diff`, `kerasMNIST_CNN_v1_vs_lecture.diff` | Raw line-by-line diffs for v1 | Done |
| `kerasMNIST_CNN_confusion.py` | Confusion matrix for the saved net, no training | Done |
| `kerasMNIST_CNN_newdata.py` | Mean accuracy of the saved net on 50,000 new digits from QMNIST, no training | Done, mean 99.23% |
| `Model_Summary.ipynb` | Lists the net's 22 layers, and shows the 66 misclassified images and the net's confidence, no training | Done, saved with its outputs |
| `kerasMNIST_CNN_sweep4.py` | Fourth search, to test the net without its fifth convolutional layer | Done, no seed passed, so the final net stays |
| `kerasMNIST_CNN_sweep3.py` | Third search, to cut training resources | Done, picked candidate 1 |
| `REPORT_NOTES.md` | Notes for filling in the report, in template order | Up to date with the final net |
| `figures/3.1_accuracy.png`, `figures/3.2_loss.png`, `figures/3.3_confusion.png` | Report figures drawn from the final log | Use these in the report |
| `figures/3.3_errors.png` | All 66 misclassified images, from the notebook | Use this in the report |
| `kerasMNIST_CNN_sweep.py` | First search for the smallest net | Done, no size passed |
| `kerasMNIST_CNN_sweep2.py` | Second search, with new training settings | Done, picked `SLICES` = 12 |
| `kerasMNIST_CNN_v1.py` | Copy of the larger net | Kept for comparison |
| `keras-examples/` | Supplied module examples | Not changed |

The four out-of-date report figures from the v1 and v2 logs were removed on 2026-11-01.

## Changes

### 1. Built the first net (v1)

Started from `keras-examples/cnnMNIST/kerasMNIST_ex_CNN.py`, which has one convolutional layer and reaches about 98% according to its own comments. Followed the suggestions at the end of that file.

- Used four convolutional layers instead of one: 32, 32, 64 and 64 slices.
- Added a maxpool after each pair of layers.
- Added Dropout of 0.25 after each maxpool and 0.5 before the output layer.
- Switched from Adam to RMSprop, following the note in the example.
- Added a learning rate reduction when validation loss stalls.
- Raised the batch size to 64, the epoch limit to 50 and the early stopping patience to 7.
- Reported training and validation accuracy from the best epoch, not the last one.
- Saved the loss plot and the trained model to files.

### 2. Added an accuracy plot

The report template needs training and validation accuracy per epoch, and v1 only plotted loss. Added a second plot, saved as `kerasMNIST_CNN_accuracy.png`. Training was not affected.

### 3. Made the net lightweight

v1 kept 803,072 of its 870,634 weights (92%) in one Dense layer, so that layer was the first thing to remove.

| Change | Reason | Effect on weights |
|---|---|---|
| Replaced Flatten and Dense(256) with Global Average Pooling | Removes the largest layer | Removes 803,072 |
| Narrowed the convolutional layers to S, S, 2S, 2S, 4S slices | Set by one number, `SLICES` | Large reduction, depends on S |
| Added a fifth convolutional layer | Replaces some of the capability lost with the Dense layer | Small increase |
| Added Batch Normalisation after each convolutional layer | Helps a small net train quickly and reliably | Adds 4 per slice |
| Added random rotation and shift | Gives extra training variety for free | None |
| Removed Dropout | A net this small has little capacity to overfit | None |
| Added a printout showing whether a GPU is in use | The first Colab log did not show the hardware | None |

**Not changed:** optimiser, loss function, batch size, epoch limit, train/validation split, learning rate schedule, early stopping and random seeds.

### 4. Size options

| `SLICES` | Weights | Reduction from v1 |
|---:|---:|---:|
| 6 | 5,464 | 99.4% |
| 8 | 9,442 | 98.9% |
| 12 | 20,638 | 97.6% |
| 16 | 36,154 | 95.8% |

### 5. Applied the second search's settings

| Setting | Before | Now |
|---|---|---|
| `SLICES` | 8 | 12 |
| Rotation | 0.03 (about 11 degrees) | 0.02 (about 7 degrees) |
| Shift | 0.1 (about 3 pixels) | 0.05 (about 1.4 pixels) |
| Learning rate | 0.001, halved when validation loss stalled | Starts at 0.002, falls to 0 along a cosine curve |
| Epochs | Up to 50 | Always 40 |
| Early stopping | Stopped after 7 epochs with no improvement | Never stops the run, only restores the best weights |
| Random seed | `np.random.seed(1)` and `tensorflow.random.set_seed(2)` | `set_random_seed(1)`, as in the search |

**Tested:** a short run on 3,000 training images, with no real test images. The settings matched the search, the best weights were restored, the learning rate reached 0, the model had 20,638 weights, and the saved model reloaded with the same predictions.

### 6. Applied the third search's settings

`kerasMNIST_CNN_v2.py` keeps the previous script. The net keeps 12 slices, 5 convolutional layers and 20,638 weights.

| Setting | Before | Now |
|---|---|---|
| Rotation and shift | 0.02 and 0.05 | Removed |
| Dropout | None | 0.1 after each maxpool |
| Epochs | 40 | 20 |
| Cosine schedule | 30,000 steps | 15,000 steps |
| Best weights chosen on | Lowest validation loss | Highest validation accuracy |

**Tested:** a 3-epoch run on 3,000 training images, with no real test images. The settings and every layer matched sweep candidate 1, the best weights were restored, the model had 20,638 weights, the saved model reloaded with the same predictions, and the confusion script ran on it.

## How the final size gets chosen

- Both searches use validation data only. They throw the test set away when loading.
- Each search's rule was fixed before its results existed.
- The final script is then run once with the chosen settings. That run gives the test result for the report.

This keeps the test score a fair measure, because the test set plays no part in choosing the design.

### First search: no size passed

**Rule:** pick the smallest size with at least 99.30% validation accuracy on all 3 seeds. A size was dropped at its first failed seed, so each size ran once.

| `SLICES` | Weights | Best epoch | Epochs run | Validation accuracy |
|---:|---:|---:|---:|---:|
| 6 | 5,464 | 23 | 30 | 98.69% |
| 8 | 9,442 | 24 | 31 | 99.13% |
| 12 | 20,638 | 16 | 23 | 99.24% |
| 16 | 36,154 | 36 | 43 | 99.27% |

Going from 12 to 16 slices gained only 0.03 points, which is within normal variation. This suggests the limit is the training settings, not the size of the net. That is a reasoned guess, and the second search tests it.

### Second search: `SLICES` = 12 passed

**Changes tested:**

- A fixed cosine learning rate schedule, starting at 0.002 and running for 40 epochs, replaces the reduction on stalled validation loss.
- Three augmentation strengths, as (rotation, shift): (0.03, 0.1), (0.02, 0.05) and none.

**Rule:**

1. With `SLICES` at 8, keep the augmentation strength with the highest validation accuracy.
2. Try sizes 6, 8 and 12, smallest first, on 3 seeds. A size is dropped at the first seed below 99.20%.
3. The smallest size with every seed at 99.30% or above wins.
4. If there is none, the smallest size with every seed at 99.20% or above wins.

The 99.20% floor was new in this search. It was lower than the first search's 99.30%, and it still left room above the 99% requirement for test set variation. The clear win at 99.30% meant that the floor was not needed as a fallback, though it still dropped sizes 6 and 8.

**Step 1 results, with `SLICES` at 8 on seed 1:**

| Rotation | Shift | Best epoch | Validation accuracy | Outcome |
|---:|---:|---:|---:|---|
| 0.03 | 0.1 | 28 | 99.17% | Not kept |
| 0.02 | 0.05 | 38 | 99.18% | Kept |
| None | None | 15 | 98.89% | Not kept |

The first two strengths differ by about one validation image, so they are effectively tied. The rule took the higher one. Using no augmentation was clearly worse.

**Step 2 results, with rotation 0.02 and shift 0.05:**

| `SLICES` | Seed | Weights | Best epoch | Validation accuracy | Outcome |
|---:|---:|---:|---:|---:|---|
| 6 | 1 | 5,464 | 35 | 99.09% | Dropped, below 99.20% |
| 8 | 1 | 9,442 | 38 | 99.18% | Dropped, below 99.20% (reused from step 1) |
| 12 | 1 | 20,638 | 33 | 99.42% | Passed |
| 12 | 2 | 20,638 | 37 | 99.42% | Passed |
| 12 | 3 | 20,638 | 40 | 99.37% | Passed |

**Picked:** `SLICES` = 12, worst seed 99.37%, mean 99.40%.

`SLICES` = 8 missed the floor by 0.02 points, about two validation images, on its only seed. The rule was kept as written, because changing it after seeing the results would undermine the point of fixing it in advance. Sizes between 8 and 12 were not tried.

## Final run

`kerasMNIST_CNN.py` was run once in Colab on a T4 GPU, with the third search's settings and seed 1. The new net was accepted because it reached at least 99%, under the rule fixed before the third search's results existed.

| Set | Images | Accuracy | Loss | Errors |
|---|---:|---:|---:|---:|
| Training (at best epoch, Dropout active) | 48,000 | 99.55% | 0.0138 | Not comparable |
| Validation (at best epoch) | 12,000 | 99.42% | 0.0211 | About 70 |
| Test | 10,000 | 99.34% | 0.0188 | 66 |

The best epoch was 16 of 20, and the run took about 82 s. It did not exactly repeat the search's seed 1 run (99.38%, best epoch 15).

**Confusion matrix:** the script gave 9,934 correct of 10,000, matching the test accuracy. The weakest digits are 9 (98.32%) and 6 (98.75%). The most common errors are 9 read as 4 (8 times), 5 read as 3 (5 times) and 6 read as 0 (4 times).

**Checked:** the downloaded `Colab Download/kerasMNIST_CNN.keras` was reloaded on the PC with no training. It has 20,638 weights and Dropout of 0.1, and it reproduced every cell of the matrix.

**Looking at the mistakes:** `Model_Summary.ipynb` shows the 66 misclassified images. The right digit was the second choice in 58 of them, and the median confidence was 70.2%, against 100.0% on correct images. This came after the net was final, and nothing was changed because of it.

## First lightweight run (v2)

`kerasMNIST_CNN_v2.py` was run once in Colab on a GPU, with no changes after the second search. This was the first time a lightweight net saw the test set. The saved `.keras` file was lost when the Colab session ended, so this net has no confusion matrix.

| Set | Images | Accuracy | Loss | Errors |
|---|---:|---:|---:|---:|
| Training (at best epoch, rotated and shifted images) | 48,000 | 99.73% | 0.0087 | Not comparable |
| Validation (at best epoch) | 12,000 | 99.36% | 0.0223 | About 77 |
| Test | 10,000 | 99.34% | 0.0212 | 66 |

The best epoch was 31 of 40, and the run took about 5.3 minutes. It did not exactly repeat the search's seed 1 run (99.42%, best epoch 33), so matching the seed did not make runs identical.

## New digits (QMNIST)

The MNIST test set gives one score and has no unused images. `kerasMNIST_CNN_newdata.py` therefore runs the saved final net on QMNIST, which rebuilds the 50,000 test digits left out of the MNIST release. It ran on the PC with no training. The saved net did not change, and the script reads `Colab Download/kerasMNIST_CNN.keras` if there is no copy in the project folder. The QMNIST files download once to a Keras folder outside the project.

**Fixed before running:** all five group scores get reported, nothing about the net changes, and 99.34% stays the project's test result.

**Checked from the QMNIST metadata:** none of the 50,000 digits and none of their writers are in the MNIST training data, and no digit repeats. The first 10,000 labels match the MNIST test labels exactly. Writers and sources are evenly mixed through the file, so the groups are taken in file order.

| Images | Number of digits | Accuracy |
|---|---:|---:|
| MNIST test set | 10,000 | 99.34% |
| QMNIST's rebuilt copy of that set | 10,000 | 99.35% |
| New digits, groups 1 to 5 | 10,000 each | 99.17%, 99.27%, 99.23%, 99.21%, 99.29% |
| New digits, mean | 50,000 | 99.23% |

The standard deviation is 0.05 points. There were 383 errors, and the 95% interval is 99.16% to 99.31%.

| Writers | MNIST test set | New digits |
|---|---:|---:|
| High school students | 99.00% (50 errors of 5,001) | 98.96% (256 errors of 24,594) |
| Census employees | 99.68% (16 errors of 4,999) | 99.50% (127 errors of 25,406) |

**Corrected in the notes:** the test set's errors looked one-sided, but the new digits show that was chance. 4 was read as 9 22 times and 9 as 4 18 times, and 3 and 5 were confused 14 times each way. 9 is still the weakest digit (98.74%), but 6 is not (99.38%).

## v1 runs

`kerasMNIST_CNN_v1.py` was run a third time on 2026-11-01, for context, after the final net was chosen. It played no part in any choice. Its saved net and plots are in `Colab Download/v1/`.

| Run | Test accuracy | Test loss | Test errors | Best epoch | Stopped at epoch |
|---:|---:|---:|---:|---:|---:|
| 1 | 99.56% | 0.0155 | 44 | 15 | 22 |
| 2 | 99.57% | 0.0158 | 43 | 19 | 26 |
| 3 | 99.57% | 0.0173 | 43 | 15 | 22 |

Run 3 scored 99.67% on training data and 99.42% on validation data, and took about 2 minutes. The three runs stayed within 1 test error of each other, and the gap to the final net is still 22 or 23 errors.

## Project hints

These hints arrived after the final run. Until then, lightweight had been assumed to mean fewest weights.

| Hint | Status |
|---|---|
| Reach more than 99% on the test set | Met, 99.34% on both lightweight nets |
| Adapt the supplied template code | Met |
| Use Colab with a T4 GPU | Met, T4 confirmed for the third search and the final run |
| Favour the least training and testing resources | Met. Training time cut from 319 s to about 82 s, and testing cost unchanged |
| Investigate techniques like Dropout | Met. The third search tested two rates, and 0.1 won |
| Use validation accuracy, not validation loss, for early stopping | Met in `kerasMNIST_CNN.py` |
| Consider a confusion matrix | Met, for the final net |

**Timing evidence from the second search, all 40 epochs:**

| Run | Time per epoch |
|---|---:|
| `SLICES` 6 with augmentation | 7.8 s |
| `SLICES` 8 with augmentation | 7.9 s |
| `SLICES` 12 with augmentation | 7.8 to 8.0 s |
| `SLICES` 8 with no augmentation | 4.0 s |

Net size barely changes training time, and augmentation doubles it. A smaller net therefore cannot fix training time, but a shorter run or less augmentation can.

### Third search: candidate 1 passed

All candidates keep `SLICES` at 12, a cosine schedule from 0.002, and validation accuracy as the metric for choosing the best weights. Dropout goes after each maxpool, as in v1. The search ran on a T4 GPU.

**Step 1 results, all on seed 1:**

| Order | Batch Normalisation | Dropout | Augmentation | Epochs | Weights | Best epoch | Validation accuracy | Time | Time per epoch |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Yes | 0.1 | None | 20 | 20,638 | 15 | 99.38% | 88 s | 4.4 s |
| 2 | Yes | 0.25 | None | 20 | 20,638 | 19 | 99.36% | 77 s | 3.9 s |
| 3 | Yes | 0.25 | None | 30 | 20,638 | 28 | 99.40% | 114 s | 3.8 s |
| 4 | No | None | (0.02, 0.05) | 20 | 20,158 | 19 | 99.38% | 123 s | 6.2 s |
| 5 | Yes | None | (0.02, 0.05) | 20 | 20,638 | 16 | 99.31% | 152 s | 7.6 s |
| 6 | Yes | None | (0.02, 0.05) | 30 | 20,638 | 29 | 99.44% | 224 s | 7.5 s |

**Step 2 results, candidate 1:**

| Seed | Best epoch | Validation accuracy | Time |
|---:|---:|---:|---:|
| 1 | 15 | 99.38% | 88 s |
| 2 | 19 | 99.37% | 79 s |
| 3 | 20 | 99.33% | 78 s |

**Picked:** candidate 1, worst seed 99.33%, mean 99.36%, about 82 s on average.

All six candidates reached 99.30% on seed 1, so the order decided the winner, not the score. The six scores span 0.13 points, about 16 validation images, and each is one seed, so small gaps mean little. The extra time on seed 1 is probably start-up, because it was the first run of the session.

Candidate 4 suggests that Batch Normalisation may not be needed. It was not confirmed on other seeds, and a net with Dropout and no Batch Normalisation was not tried. No further search was run on this: the rule stops at the winner, and the gain would be at most 480 weights and about a fifth of the training time.

**Rule, fixed before any results:**

1. Run every candidate once on seed 1, so that the report has a figure for each.
2. In the order above, the first candidate with all 3 seeds at 99.30% or above wins.
3. If none passes, the current net stays.

**Changed on 2026-11-01, before any third search results were seen:** rule 3 used to end "and nothing is rerun". The saved 99.34% net was then lost, because Colab deleted `kerasMNIST_CNN.keras` when the session ended. If no candidate passes, `kerasMNIST_CNN.py` gets one unchanged run, only to produce the confusion matrix. 99.34% stays the main result, and the report gives both test results and says that the matrix came from the second run.

**Test set rule, also fixed in advance:** a winner gets `kerasMNIST_CNN.py` updated and run once. If it reaches at least 99%, it becomes the final net, and the report lists both test results. If it misses 99%, the 99.34% net stays. The reason for this second test is the lecturer's guidance, not the first score.

Picking the best epoch on accuracy instead of loss gives a slightly higher validation figure, so the 99.30% target is a little easier than in the earlier searches.

### Fourth search: no seed passed, the fifth layer stays

The fifth convolutional layer holds 10,608 of the 20,638 weights, and it was added on reasoning alone. `kerasMNIST_CNN_sweep4.py` removes that layer and keeps everything else from the final net. It ran in Colab on a GPU, after the final run, and used no test data.

| Net | Convolutional layers | Weights | Multiplications per image (calculated) |
|---|---:|---:|---:|
| Final | 5 | 20,638 | 3,133,344 |
| Without the fifth layer | 4 | 9,790 (53% fewer) | 2,625,072 (16% fewer) |

**Rule, fixed before any results:**

1. Run 3 seeds at 20 epochs, using validation data only.
2. If all 3 seeds reach 99.30%, the net without the fifth layer wins.
3. If any seed misses 99.30%, the final net stays and nothing else is run.

**Test set rule, the same as the third search:** a winner gets `kerasMNIST_CNN.py` updated and run once. If it reaches at least 99%, it becomes the final net, and the report lists all test results. If it misses 99%, the 99.34% net stays.

**Tested before the Colab run:** one epoch on 1,500 training images, on the PC. The net had 19 layers and 9,790 weights as calculated, and both outcomes of the rule printed correctly. The script saves no files, so it cannot overwrite the final net.

**Results, 20 epochs:**

| Seed | Best epoch | Validation accuracy | Time | Final net on this seed |
|---:|---:|---:|---:|---:|
| 1 | 16 | 99.17% | 85 s | 99.38% |
| 2 | 18 | 98.96% | 78 s | 99.37% |
| 3 | 20 | 99.02% | 77 s | 99.33% |

**Outcome:** worst seed 98.96%, mean 99.05%, about 80 s on average, so the final net stays and nothing else was run. Mean validation accuracy fell by 0.31 points, from about 77 errors to about 114 of 12,000. The best seed without the layer was below the worst seed with it. Training time barely changed, which matches the earlier finding that net size barely affects it.

## Team repo

The GitHub repo `Bleenane/AI_Project1` holds the results, notes and figures for the team writing the report. Its files are copies, and the original project folder stays as the full working record.

- **`code/Final_Model.ipynb`** combines `kerasMNIST_CNN.py`, `kerasMNIST_CNN_confusion.py` and `kerasMNIST_CNN_newdata.py`. It loads the saved net unless `TRAIN` is set to `True`, and then it saves under a new name, so the final net can't be overwritten.
- **`figures/`** holds the report figures, renamed by report section. The accuracy figure was redrawn without its inset, and its 99% target line moved onto the main plot. The confusion matrix was redrawn with a border around every cell and stronger fills.
- **Left out:** the Chauhan et al. PDF, `keras-examples/`, v1, v2, searches 1, 2 and 4, the raw `.diff` files and the raw Colab plots.

## Still to do

1. Fill in the report from `REPORT_NOTES.md`.
2. Write section 5, the Reflection, yourselves.
3. Check the references, and whether the module needs an AI declaration.

## Report sections affected

All of these are covered in `REPORT_NOTES.md`.

| Section | What changes |
|---|---|
| Theoretical Context | Add Global Average Pooling, Batch Normalisation, data augmentation and the cosine learning rate schedule. Keep Dropout, because the third search tests it. |
| 2 Approach and Design Decisions | Replace with the lightweight reasoning and all four search results. |
| 2.2 Model Architecture | New layer table for `SLICES` = 12, with Dropout and no augmentation. |
| 2.3 Training Setup | Add Dropout, the cosine schedule, 20 epochs and the T4 GPU. |
| 3.1 and 3.2 | New figures from the new run. |
| 3.3 Quantitative Results | New figures, the confusion matrix, and all three nets as a comparison. |
| 4 Discussion and 6 Conclusion | Rewrite around training and testing resources against accuracy. |
