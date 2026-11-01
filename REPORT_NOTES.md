# Project 1: Report Notes

These notes follow the report template in order and describe the final net. It is safe to fill in every section from them now.

**Final result: 99.34% on the test set, with 20,638 weights and about 82 s of training on a T4 GPU.**

## Check these first

1. **Hidden template pages:** pages 3, 8 and 9 of the template are blacked out. Section 2.1 below is a guess at page 3. The confusion matrix in section 3.3 may belong on pages 8 and 9.
2. **Meaning of "lightweight":** the lecturer favours nets needing "the least training and testing resources", and warns against unnecessary extra layers. Address both training and testing in section 4.
3. **AI use:** check whether the module requires a declaration.
4. **References:** all except Chauhan et al. and Yadav and Bottou are from memory. Check them before submitting.
5. **Reflection:** section 5 has prompts only, as it has to come from you.

## 1. Introduction

- **Task:** classify handwritten digits into 10 classes.
- **Data:** MNIST, with 60,000 training and 10,000 test images, each 28 × 28 greyscale.
- **Target:** at least 99% test accuracy, using the least training and testing resources.
- **Method:** extend the module's example CNN, shrink it, then cut its training time, choosing every setting on validation data only.
- **Result:** 99.34% on the test set, with 20,638 weights and about 82 s of training.

## Theoretical Context

**Basics**

- **Why a CNN:** a dense net flattens the image and loses its 2D layout. A CNN keeps it.
- **Convolution:** a small filter slides across the image with the same weights everywhere, so a feature is detected wherever it appears.
- **Stacked 3 × 3 layers:** two of them see the same 5 × 5 patch as one larger filter, with fewer weights and an extra ReLU.
- **Max pooling:** keeps the strongest response in each 2 × 2 patch, halving the image size.
- **ReLU:** adds non-linearity and suffers less from vanishing gradients than sigmoid.
- **Softmax and log-loss:** softmax turns outputs into class probabilities. Categorical cross-entropy penalises confident wrong answers heavily.
- **RMSprop:** scales each weight's update using a running average of its recent squared gradients.

**Lightweight techniques**

- **Global Average Pooling:** averages each final slice down to one number, and has no weights. Here the output layer needs 490 weights, where Flatten would have needed 23,530.
- **Batch Normalisation:** rescales each layer's outputs during training, which helps a small net train reliably. It adds 4 values per slice, 2 trained and 2 running averages.
- **Dropout:** on each training step, randomly sets a fraction of a layer's outputs to zero, so the net cannot rely on any one feature. It adds no weights and is off during testing.
- **Cosine learning rate schedule:** the rate falls smoothly from a starting value to 0 along a fixed curve. Large steps early give quick progress, and small steps late let the net settle.
- **Data augmentation (tested, not in the final net):** small random rotations and shifts show the net a slightly different version of each digit on every epoch. It adds no weights and is off during testing, but it doubled the training time.

**Counting weights**

- Conv2D: (filter height × filter width × input slices + 1) × slices. Example: (3 × 3 × 12 + 1) × 12 = 1,308.
- Dense: (inputs + 1) × outputs. Example: (48 + 1) × 10 = 490.

## 2. Approach and Design Decisions

**Stage 1: reach 99% (v1).** We extended the example net to 4 convolutional layers with Dropout and kept Flatten and Dense(256). It scored 99.56%, 99.57% and 99.57% on the test set in three runs, but used 870,634 weights, and 92% of them sat in the Dense layer.

**Stage 2: rebuild it to be lightweight.**

| Change | Reason |
|---|---|
| Replaced Flatten and Dense(256) with Global Average Pooling | Removes 803,072 weights |
| Narrowed the layers to S, S, 2S, 2S, 4S slices | One number, `SLICES`, sets the size |
| Added a fifth convolutional layer | Replaces some capability lost with the Dense layer. This was reasoned at first, and Stage 7 tested it |
| Added Batch Normalisation | Helps a small net train reliably |
| Added random rotation and shift | Gives extra training variety for no weights |
| Removed Dropout | Kept the net simple. This was reasoned, not tested, and Stage 5 reversed it |

**Stage 3: choose the size.** Each search's rule was fixed before its results existed. Every search threw the test set away when loading, so it played no part in the choice.

First search rule: the smallest size with at least 99.30% validation accuracy on 3 seeds. No size passed, and each was dropped at seed 1.

| `SLICES` | Weights | Validation accuracy |
|---:|---:|---:|
| 6 | 5,464 | 98.69% |
| 8 | 9,442 | 99.13% |
| 12 | 20,638 | 99.24% |
| 16 | 36,154 | 99.27% |

Going from 12 to 16 slices barely helped, so we suspected the training settings. The second search used a cosine schedule from 0.002 over 40 epochs, and compared three augmentation strengths with `SLICES` at 8 on seed 1.

| Rotation | Shift | Validation accuracy |
|---:|---:|---:|
| 0.03 | 0.1 | 99.17% |
| 0.02 | 0.05 | 99.18% (kept) |
| None | None | 98.89% |

Second search rule: drop a size at its first seed below 99.20%, and pick the smallest size with all 3 seeds at 99.30% or above.

| `SLICES` | Weights | Validation accuracy by seed | Outcome |
|---:|---:|---|---|
| 6 | 5,464 | 99.09% | Dropped |
| 8 | 9,442 | 99.18% | Dropped |
| 12 | 20,638 | 99.42%, 99.42%, 99.37% | Picked |

**Stage 4: first test.** We trained the chosen net once and evaluated it on the test set once. It scored 99.34% with 66 errors, but took about 5.3 minutes to train.

**Stage 5: cut the training resources.** The project favours nets needing the least training and testing resources, and our earlier searches only reduced the weights. The second search showed that net size barely changed training time: 6, 8 and 12 slices all took about 8 s per epoch. Augmentation doubled it, from about 4 s. We therefore kept `SLICES` at 12 and tested shorter runs, Dropout in place of augmentation, and removing Batch Normalisation. The cosine schedule was shortened to fit each run, and the best weights were chosen on validation accuracy, as the project hints recommend.

Third search rule: run every candidate once on seed 1, then take them in the order below, cheapest training first. The first with all 3 seeds at 99.30% or above wins. If none passes, the current net stays. The rule was fixed before any results existed, and the test set was discarded on loading.

Results on seed 1, on a T4 GPU:

| Order | Batch Normalisation | Dropout | Rotation, shift | Epochs | Weights | Best epoch | Validation accuracy | Time | Time per epoch |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Yes | 0.1 | None | 20 | 20,638 | 15 | 99.38% | 88 s | 4.4 s |
| 2 | Yes | 0.25 | None | 20 | 20,638 | 19 | 99.36% | 77 s | 3.9 s |
| 3 | Yes | 0.25 | None | 30 | 20,638 | 28 | 99.40% | 114 s | 3.8 s |
| 4 | No | None | 0.02, 0.05 | 20 | 20,158 | 19 | 99.38% | 123 s | 6.2 s |
| 5 | Yes | None | 0.02, 0.05 | 20 | 20,638 | 16 | 99.31% | 152 s | 7.6 s |
| 6 | Yes | None | 0.02, 0.05 | 30 | 20,638 | 29 | 99.44% | 224 s | 7.5 s |

All six passed on seed 1, so candidate 1 was tested first on the other seeds, and it passed. The search stopped there, so candidates 2 to 6 have one seed each.

| Candidate 1 seed | Best epoch | Validation accuracy | Time |
|---:|---:|---:|---:|
| 1 | 15 | 99.38% | 88 s |
| 2 | 19 | 99.37% | 79 s |
| 3 | 20 | 99.33% | 78 s |

Picked: candidate 1, with Batch Normalisation, Dropout of 0.1, no augmentation and 20 epochs. Its worst seed was 99.33% and its mean was 99.36%. The extra time on seed 1 is probably start-up, because it was the first run of the session.

Each pair below changes one thing, which gives direct comparisons for section 4:

| Candidates | What changes |
|---|---|
| 1 and 2 | Dropout 0.1 against 0.25 |
| 2 and 3 | 20 against 30 epochs, with Dropout |
| 2 and 5 | Dropout against augmentation |
| 4 and 5 | Without and with Batch Normalisation |
| 5 and 6 | 20 against 30 epochs, with augmentation |

Choosing the best epoch on accuracy gives a slightly higher validation figure than choosing on loss, so the 99.30% target was a little easier than in the earlier searches.

**Stage 6: final test.** We put the winning settings into the script, trained once on seed 1, and evaluated on the test set once. It scored 99.34% with 66 errors. The rule was fixed before the third search's results existed: the new net would replace the first if it reached at least 99%. The test scores played no part in choosing between the two nets, and the reason for this second test was the lecturer's guidance on resources.

**Stage 7: test the fifth layer.** The fifth convolutional layer holds about half of the weights, and the project warns against unnecessary extra layers, so we tested the net without it. Everything else stayed the same, including the seeds.

Fourth search rule: if all 3 seeds reach 99.30% validation accuracy, the 4-layer net wins. If any seed misses, the final net stays and nothing else is run. The rule was fixed before any results existed, and the test set was discarded on loading. This came after the final test, but it used no test data.

| Net | Weights | Multiplications per image (calculated) | Validation accuracy by seed | Mean | Mean time |
|---|---:|---:|---|---:|---:|
| Final, 5 convolutional layers | 20,638 | 3.13M | 99.38%, 99.37%, 99.33% | 99.36% | about 82 s |
| Without the fifth layer | 9,790 | 2.63M | 99.17%, 98.96%, 99.02% | 99.05% | about 80 s |

No seed reached 99.30%, so the final net stayed and it was not tested again.

**Stage 8: check the saved net on new digits.** The MNIST test set gives one score, and it has no unused images, so we used QMNIST (Yadav and Bottou). QMNIST rebuilds the 50,000 test digits that were left out of the MNIST release. Its metadata shows that none of these digits, and none of their writers, appear in the MNIST training data, which is also where our validation set came from. We ran the saved net on them in 5 groups of 10,000 and took the mean, with no training. We decided before running it that all five scores would be reported, and that nothing about the net would change.

## 2.1 Data Preparation (assumed heading)

- Scaled pixels from 0-255 to 0-1 as `float32`.
- Reshaped to (N, 28, 28, 1), because `Conv2D` needs a channel dimension.
- One-hot encoded the labels.
- Held back the last 20% of training images: 48,000 for training and 12,000 for validation.
- Used no data augmentation in the final net, as Dropout replaced it.

## 2.2 Model Architecture

The net has 22 layers, not counting the input. Six of them are the main learning layers: 5 convolutional and 1 Dense.

| Layer type | How many | What it does |
|---|---:|---|
| Conv2D | 5 | Slides 3 × 3 filters across the image to detect features |
| Batch Normalisation | 5 | Rescales the outputs of each convolutional layer |
| Activation | 6 | 5 ReLU after the convolutional layers, and softmax at the end |
| MaxPooling | 2 | Halves the height and width of the image |
| Dropout | 2 | Turns off 10% of the outputs on each training step, and is off during testing |
| Global Average Pooling | 1 | Averages each of the 48 final slices down to one number |
| Dense | 1 | Turns those 48 numbers into the 10 digit outputs |

The table below groups these layers to fit the report. Each convolutional row includes Batch Normalisation and ReLU, and its weight count includes the Batch Normalisation values. [Model_Summary.ipynb](Model_Summary.ipynb) lists all 22 layers one by one.

| Layer | Output shape | Weights |
|---|---:|---:|
| Input | 28 × 28 × 1 | 0 |
| Conv2D, 12 slices, 3 × 3 | 28 × 28 × 12 | 168 |
| Conv2D, 12 slices, 3 × 3 | 28 × 28 × 12 | 1,356 |
| MaxPooling 2 × 2 | 14 × 14 × 12 | 0 |
| Dropout 0.1 (training only) | 14 × 14 × 12 | 0 |
| Conv2D, 24 slices, 3 × 3 | 14 × 14 × 24 | 2,712 |
| Conv2D, 24 slices, 3 × 3 | 14 × 14 × 24 | 5,304 |
| MaxPooling 2 × 2 | 7 × 7 × 24 | 0 |
| Dropout 0.1 (training only) | 7 × 7 × 24 | 0 |
| Conv2D, 48 slices, 3 × 3 | 7 × 7 × 48 | 10,608 |
| Global Average Pooling | 48 | 0 |
| Dense 10, softmax | 10 | 490 |
| **Total** | | **20,638** |

Of these, 20,398 are trained and 240 are Batch Normalisation running averages. They take up 80.62 KB.

## 2.3 Training Setup

| Setting | Value |
|---|---|
| Framework | TensorFlow 2.21.0 with Keras, on Google Colab |
| Hardware | Colab T4 GPU |
| Optimiser | RMSprop |
| Learning rate | 0.002, falling to 0 along a cosine curve over 15,000 steps |
| Loss | Categorical cross-entropy |
| Batch size | 64 |
| Epochs | 20, every run takes all of them |
| Weights kept | From the epoch with the highest validation accuracy |
| Dropout | 0.1 after each maxpool |
| Train/validation split | 48,000 / 12,000 |
| Random seed | 1 |
| Actual run | About 82 s, about 4 s per epoch |

The 82 s is the sum of the log's rounded epoch times, including 11 s for epoch 1. It matches the 82 s average from the search.

## 3.1 Training and Validation Accuracy

Use [figures/3.1_accuracy.png](figures/3.1_accuracy.png).

**Figure caption:** Training and validation accuracy per epoch, with the best epoch and the 99% target marked.

- Training accuracy rises from 92.81% to 99.65%.
- Validation accuracy starts higher, at 96.72%. Training accuracy is averaged across an epoch with Dropout on, while validation accuracy is measured at the end of the epoch with Dropout off.
- Validation accuracy dips to 95.08% at epoch 2. The likely cause is the high early learning rate. We did not test this.
- It first reaches 99% at epoch 10, and stays above it for the rest of the run.
- It peaks at 99.42% at epochs 16 and 17, and the weights from epoch 16 are kept.

## 3.2 Training and Validation Loss

Use [figures/3.2_loss.png](figures/3.2_loss.png). It has a log axis, because the first epochs would otherwise flatten the rest of the curve.

**Figure caption:** Training and validation loss per epoch, on a log scale.

- Validation loss falls from 0.1055 to its lowest value, 0.0211, at epochs 15 and 16.
- It spikes at epochs 2, 5 and 6 while the learning rate is still high, and settles as the rate falls.
- Training loss keeps falling to 0.0116, while validation loss stays flat near 0.021 after epoch 15.
- Validation accuracy and validation loss pick the same point here, as epoch 16 ties for the lowest validation loss.

## 3.3 Quantitative Results

**Final net, at the best epoch (16):**

| Set | Images | Accuracy (%) | Loss | Errors |
|---|---:|---:|---:|---:|
| Training (Dropout active) | 48,000 | 99.55 | 0.0138 | Not comparable |
| Validation | 12,000 | 99.42 | 0.0211 | About 70 |
| **Test** | 10,000 | **99.34** | 0.0188 | **66** |

**Confusion matrix:** use [figures/3.3_confusion.png](figures/3.3_confusion.png). The saved net was reloaded and reproduced this matrix exactly, and each column adds up to the number of test images for that digit.

**Figure caption:** Confusion matrix on the 10,000 test images. Rows are predicted digits and columns are true digits. Grey cells are correct and blue cells are errors, with darker blue for more errors. Empty cells are zero.

**Accuracy on each digit:**

| True digit | Images | Errors | Accuracy (%) |
|---:|---:|---:|---:|
| 0 | 980 | 4 | 99.59 |
| 1 | 1,135 | 2 | 99.82 |
| 2 | 1,032 | 9 | 99.13 |
| 3 | 1,010 | 2 | 99.80 |
| 4 | 982 | 3 | 99.69 |
| 5 | 892 | 7 | 99.22 |
| 6 | 958 | 12 | 98.75 |
| 7 | 1,028 | 7 | 99.32 |
| 8 | 974 | 3 | 99.69 |
| 9 | 1,009 | 17 | 98.32 |

**The mistakes themselves:** use [figures/3.3_errors.png](figures/3.3_errors.png), from [Model_Summary.ipynb](Model_Summary.ipynb).

**Figure caption:** All 66 misclassified test images, sorted by true digit. Each label gives the true digit, the predicted digit and the net's confidence, so 9→4 87% is a 9 read as a 4 at 87%.

| Measure | Value |
|---|---:|
| Median confidence on correct images | 100.0% |
| Median confidence on the mistakes | 70.2% |
| Mistakes below 60% confidence | 19 of 66 |
| Mistakes at 90% confidence or above | 22 of 66 |
| Correct images below 60% confidence | 10 of 9,934 |
| Mistakes where the right digit was the second choice | 58 of 66 |

**The three nets:**

| Measure | v1 | First lightweight net | Final net |
|---|---:|---:|---:|
| Weights | 870,634 | 20,638 | 20,638 |
| Weight storage | 3.32 MB | 80.62 KB | 80.62 KB |
| Multiplications per image | 19.1 million | 3.1 million | 3.1 million |
| Epochs | 22, 26, 22 | 40 | 20 |
| Time per epoch | About 5 s | About 8 s | About 4 s |
| Training time | About 2 minutes | About 5.3 minutes | About 82 s |
| Test accuracy | 99.56%, 99.57%, 99.57% | 99.34% | 99.34% |
| Test errors | 44, 43, 43 | 66 | 66 |

The multiplication counts are calculated from the layer sizes, not measured. v1's third run was a rerun for context, made after the final net was chosen, and it played no part in any choice.

**Compared with other nets:**

| Net | Source | Weights | Test accuracy |
|---|---|---:|---:|
| **Final net** | **This project, one run** | **20,638** | **99.34%** |
| Small CNN | Chauhan et al. | 93,322 | 98.81% |
| Lecture example | Module slide | 504,686 | About 98% (expected, not measured) |
| Dense only | Chauhan et al. | 669,706 | 98.31% |
| v1 | This project, three runs | 870,634 | 99.56%, 99.57%, 99.57% |

**The saved net on new digits:**

| Images | Number of digits | Accuracy |
|---|---:|---:|
| MNIST test set (the project's result) | 10,000 | 99.34% |
| QMNIST's rebuilt copy of that set | 10,000 | 99.35% |
| New digits, group 1 | 10,000 | 99.17% |
| New digits, group 2 | 10,000 | 99.27% |
| New digits, group 3 | 10,000 | 99.23% |
| New digits, group 4 | 10,000 | 99.21% |
| New digits, group 5 | 10,000 | 99.29% |
| **New digits, mean of the 5 groups** | **50,000** | **99.23%** |

The standard deviation between the groups is 0.05 points. Across all new digits there were 383 errors, and the 95% interval is 99.16% to 99.31%.

| Writers | MNIST test set | New digits |
|---|---:|---:|
| High school students | 99.00% | 98.96% |
| Census employees | 99.68% | 99.50% |

## 4. Discussion

**Result**

- We met the target. The 95% interval on the test score is about 99.18% to 99.50%, so even the lower end clears 99%.
- Test accuracy is 0.08 points below validation accuracy, so we see no sign that the searches overfitted the validation set.
- Both lightweight nets made 66 test errors. Training about 4 times faster cost no measurable test accuracy, though one run each cannot prove the nets are equal.
- v1 remains the most accurate net, with 22 or 23 fewer errors. We chose the final net because both pass 99% and the lecturer favours lightweight solutions. The hints don't say how much accuracy that is worth, so this is a judgement.
- On 50,000 new digits the saved net averaged 99.23%, and every group of 10,000 stayed above 99%. That is 0.11 points below the test score, but inside its 95% interval, so 99.34% was a fair result that came out slightly high.
- Yadav and Bottou report the same pattern for other nets, with errors higher on these digits than on the MNIST test set. For example, LeNet5 rises from 0.82% to 1.08%, and ours rises from 0.66% to 0.77%.
- Almost all of the gap comes from one source. Students' digits scored about the same on both sets (99.00% and 98.96%), but Census employees' digits fell from 99.68% to 99.50%. Yadav and Bottou describe the Census half of the MNIST test set as very clean digits from only 48 writers, which may be why. We did not test this further.

**Resources**

- The final net has 42 times fewer weights than v1, and 6 times fewer multiplications per image. It costs 22 or 23 more errors out of 10,000.
- Removing Flatten and Dense(256) gave 94% of the saving in weights, but only 5% of the saving in multiplications. Narrowing the convolutional layers gave the other 95% (calculated).
- Against the lecture example, the final net has 24.5 times fewer weights but does about 5.5 times more multiplications per image (calculated). A convolutional filter reuses its weights at every position, so fewer weights doesn't always mean less arithmetic. That example does not reach 99%, so v1 is the fairer comparison.
- It beat the small CNN from Chauhan et al. with 4.5 times fewer weights. The setups differ, so this is a rough comparison.
- The first lightweight net trained about 2.5 times slower than v1. The final net trains about 1.5 times faster than v1, with fewer epochs and less time per epoch.
- Net size barely affected training time: 6, 8 and 12 slices all took about 8 s per epoch. Augmentation doubled it, so a smaller net alone would not have fixed training time.
- Over a full run the model does about 7 to 8 times less arithmetic than v1 (calculated), but runs only 1.5 times faster. Clock time on a GPU does not follow the multiplication count directly, and we did not test why.

**What the searches showed**

- Augmentation clearly helped at `SLICES` = 8 with no Dropout in the net: 99.18% against 98.89% without it, on one seed.
- The strengths (0.03, 0.1) and (0.02, 0.05) were effectively tied, about one validation image apart.
- The second search lifted `SLICES` = 12 from 99.24% to 99.42% on seed 1. It changed the schedule and augmentation together, so we can't credit either one alone.
- **Third search, overall:** all six candidates cleared 99.30% on seed 1. Their scores span 0.13 points, about 16 validation images, so small gaps mean little.
- **Dropout against augmentation:** Dropout of 0.25 reached 99.36% at 3.9 s per epoch. Augmentation reached 99.31% at 7.6 s per epoch. Accuracy was about the same at half the time, on one seed.
- **Batch Normalisation:** without it the net reached 99.38% at 6.2 s per epoch, against 99.31% at 7.6 s with it. It may not be needed, but that is one seed, with augmentation.
- **Epochs:** going from 20 to 30 added 0.04 points with Dropout and 0.13 points with augmentation, for 50% more time.
- **Fourth search, the fifth layer:** removing it cut the weights by 53% and the multiplications by 16%, but mean validation accuracy fell by 0.31 points, from about 77 errors to about 114 of 12,000. The best seed without the layer (99.17%) was below the worst seed with it (99.33%), so this gap is larger than the variation between seeds. Training time barely changed, at about 80 s against about 82 s, so the layer adds almost no training time and does not count as an unnecessary extra layer.

**Where the net goes wrong**

- 9 is the weakest digit at 98.32%, with 17 errors, followed by 6 at 98.75%, with 12. 1 and 3 are the strongest, with 2 errors each.
- The most common errors are 9 read as 4 (8 times), 5 read as 3 (5 times) and 6 read as 0 (4 times). Five other pairs tie at 3.
- On the test set these errors looked one-sided, as 9 was read as 4 eight times but 4 was read as 9 only once. The new digits show that this was chance. There, 4 was read as 9 22 times and 9 as 4 18 times, and 3 and 5 were confused 14 times each way.
- 9 was also the weakest digit on the new digits, at 98.74%, so that weakness looks real. 6 was not, at 99.38%, so its low test score was probably chance. The next weakest new digits were 2 (98.96%) and 5 (99.12%).
- **What the images show (our reading, so check it against the figure):** several 9s read as 4 have a top loop that is open or barely closed. Several 5s read as 3 have a weak or missing top stroke. Some digits are broken into separate strokes, such as the 6s read as 1, and some are very heavy.
- The net is usually unsure when it fails. Its median confidence is 70.2% on mistakes, against 100.0% on correct images, and only 10 correct images fall below 60%.
- It can still be wrong with confidence, as 22 of the 66 mistakes are at 90% or above.
- The right digit was the net's second choice in 58 of the 66 mistakes, so most errors are near misses.
- We looked at these images after the net was final, and changed nothing because of them.

**Limits**

- Each lightweight test score comes from one run, so it has no spread. Repeating a run to pick a better score would not be fair. The new digits show how the saved net's score varies between sets of images, not between training runs.
- QMNIST's digits are rebuilt, not original MNIST files. The rebuilt copy of the test set scored 99.35% against 99.34%, with every label matching, so the effect looks very small.
- `SLICES` = 8 missed the floor by 0.02 points on one seed, and sizes 9 to 11 were not tried. A smaller net may also pass.
- The 0.002 rate and the batch size of 64 were reasoned choices, not tested separately.
- We use validation accuracy to pick the weights, as the hints recommend, but the run never stops early. Epochs 17 to 20 gave no improvement, so that time was spent for nothing.
- A net with Dropout and no Batch Normalisation was not tried. It could have saved at most 480 weights and about a fifth of the training time, so we stopped at the search's winner.
- The same seed did not exactly repeat the search's run (99.38% and best epoch 15 then, 99.42% and best epoch 16 in the final run), so runs are not fully repeatable. v1's three runs used the same code and seeds and stayed within 1 test error of each other, though the lightweight nets may vary differently.
- T4 was confirmed from the third search onwards. The earlier runs show `GPU:0`, so their timing comparisons assume the same GPU. The matching 8 s per epoch with augmentation supports this.

## 5. Reflection

Write this yourselves. Prompts to use:

- What did you expect when the first search failed, and what did it teach you?
- How did the lecturer's definition of "lightweight" change your plan, and which of your earlier decisions did it reverse?
- Why does using the test set to pick a design make the score unfair?
- Looking at [figures/3.3_errors.png](figures/3.3_errors.png), how many of the 66 digits would you have got wrong yourselves?
- What would you try next, such as Dropout without Batch Normalisation, `SLICES` = 10 (14,500 weights), or fewer than 20 epochs?

## 6. Conclusion (draft)

> The project objective was met. The final CNN exceeded the required accuracy on the unseen test set while using a small fraction of the weights of our first design, and it also trains faster than that design. The largest saving in weights came from replacing the flattened Dense layers with Global Average Pooling. The largest saving in training time came from replacing data augmentation with Dropout and halving the epochs, which cost no measurable test accuracy. Because every setting was chosen on validation data, the test result is a fair measure of performance, and the saved net's mean of 99.23% on 50,000 new digits supports this. Removing the fifth convolutional layer was tested and fell short of the validation target, but a lighter net may still be possible, as Dropout without Batch Normalisation and sizes between 8 and 12 slices were not tested.

## References

1. Y. LeCun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-based learning applied to document recognition," *Proc. IEEE*, vol. 86, no. 11, pp. 2278-2324, 1998.
2. K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," in *Proc. ICLR*, 2015.
3. S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in *Proc. ICML*, 2015, pp. 448-456.
4. M. Lin, Q. Chen and S. Yan, "Network in network," in *Proc. ICLR*, 2014.
5. I. Loshchilov and F. Hutter, "SGDR: Stochastic gradient descent with warm restarts," in *Proc. ICLR*, 2017.
6. C. Shorten and T. M. Khoshgoftaar, "A survey on image data augmentation for deep learning," *J. Big Data*, vol. 6, no. 60, 2019.
7. T. Tieleman and G. Hinton, "Lecture 6.5: RMSProp," *Neural Networks for Machine Learning*, Coursera, 2012.
8. S. Chauhan, R. Kumar and B. Kumar, "Handwritten digits prediction system using convolution neural network for MINIST dataset," in *Proc. 2024 1st Int. Conf. Advances in Computing, Communication and Networking (ICAC2N)*, 2024, pp. 156-161, doi: 10.1109/ICAC2N63387.2024.10894911.
9. F. Chollet et al., "Keras," 2015. [Online]. Available: https://keras.io
10. Module example code, `kerasMNIST_ex_CNN.py`.
11. N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," *J. Mach. Learn. Res.*, vol. 15, no. 56, pp. 1929-1958, 2014.
12. C. Yadav and L. Bottou, "Cold case: The lost MNIST digits," in *Advances in Neural Information Processing Systems 32*, 2019. Data: github.com/facebookresearch/qmnist.

Keep "MINIST" in reference 8, because that is the published title.
