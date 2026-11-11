# Speech Emotion Recognition from Acoustic Features

### A Speaker-Independent Study of Noise Robustness

**Corinne D’Elia**  
Audio Pattern Recognition — Università degli Studi di Milano

## Overview

This project studies the recognition of four emotions from speech: **calm, happy, sad, and angry**. It compares Support Vector Machine (SVM) and Random Forest (RF) classifiers using acoustic features from the RAVDESS dataset.

The main goal is to evaluate recognition in speakers not encountered during training and to examine how background noise affects performance.

## Repository files

- `APR_Speech_Emotion_Recognition.ipynb` — Main notebook with explanations and experimental code.
- `APR_Speech_Emotion_Recognition.py` — Python version of the code.
- `APR_Report_D'Elia.pdf` — Project report with methods, results, and discussion.
- `README.md` — Project overview and execution instructions.

## Datasets

**RAVDESS** provides the speech recordings. The project uses the audio-only speech archive `Audio_Speech_Actors_01-24.zip`. Selecting four emotions and removing one duplicate gives **767 recordings from 24 actors**. [Download RAVDESS](https://zenodo.org/records/1188976).

**DEMAND** provides the environmental noise recordings. Download `STRAFFIC_16k.zip` for traffic noise and `PCAFETER_16k.zip` for cafeteria noise. The project uses the `ch01.wav` channel from each archive. [Download DEMAND](https://zenodo.org/records/1227121).

White Gaussian noise is generated directly in Python.

## Method

Audio is processed at 16 kHz and represented by **101 acoustic features**, including MFCCs, their temporal derivatives, energy, spectral descriptors, pitch, and periodicity. PCA and k-means are used to explore the feature space.

The evaluation is **speaker-independent**. Eighteen actors are used for development and six for the final test. Model selection uses three-fold cross-validation grouped by actor, with preprocessing fitted within each training fold.

SVM and Random Forest are trained on clean speech. They are then tested on clean recordings and on recordings mixed with white, traffic, and cafeteria noise at **20, 10, and 0 dB SNR**. Each noisy condition is repeated with three noise realizations. Permutation importance is used to examine which features the models rely on.

## Main results

On clean speech, SVM achieves **0.731 macro-F1** and Random Forest achieves **0.708**. The actor-based bootstrap interval for their difference includes zero, so the clean-test comparison remains uncertain.

Random Forest achieves higher mean macro-F1 in **all nine noisy conditions**. For example, with cafeteria noise at 20 dB, it reaches **0.638**, compared with **0.318** for SVM. MFCCs are the most influential feature group for both models on the clean test set.

The main limitations are the use of acted speech, only six test actors, and training on clean audio only. The report discusses these limitations and the gap between training and validation performance.

## Running the project in Google Colab

1. Download the three dataset archives listed above.
2. Upload them to a folder named `APR_data` in **My Drive**, keeping their original filenames.
3. Open `APR_Speech_Emotion_Recognition.ipynb` in Google Colab.
4. Check the settings in the data-loading cell:

   ```python
   DATA_MODE = 'drive'
   DRIVE_INPUT = '/content/drive/MyDrive/APR_data'
   INPUT_OVERRIDE = None
   ```

5. Run all cells in order, starting with the dependency installation cell, and authorize Google Drive access when prompted. If Colab requests a restart after installation, restart the runtime and rerun the cells from the top.
6. At the end, the notebook creates and downloads `APR_results.zip`, containing figures, result tables, trained models, and experimental settings.

If your data folder is in a different location, update `DRIVE_INPUT`. The notebook extracts the archives automatically. The main libraries are NumPy, SciPy, librosa, pandas, scikit-learn, matplotlib, and seaborn. Package versions are specified in the notebook, and the random seed is set to **42**.
