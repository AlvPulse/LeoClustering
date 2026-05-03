# Feature Dictionary

## `spectral_centroid_hz`
**Physical Meaning:** Frequency center of mass of the spectrum.

**Computation Summary:** librosa.feature.spectral_centroid

**Bioacoustic Rationale:** Higher for noisy/bright sounds, lower for tonal/dull sounds.

**Fraction NaNs:** 0.00%

**Global Typical Range:** [4058.44, 5541.12]

**Global Mean/Std:** 5222.34 ± 359.64

**Per-Class Stats (Top 10 + Rare):**
- `rare_animal`: 4061.77 ± 4.71
- `wind`: 5511.82 ± 13.17
- `bird`: 5029.65 ± 12.65

---

## `spectral_flatness`
**Physical Meaning:** Ratio of geometric mean to arithmetic mean of the power spectrum.

**Computation Summary:** librosa.feature.spectral_flatness

**Bioacoustic Rationale:** Differentiates tonal (close to 0) from noise-like (close to 1) signals.

**Fraction NaNs:** 0.00%

**Global Typical Range:** [0.01, 0.57]

**Global Mean/Std:** 0.29 ± 0.28

**Per-Class Stats (Top 10 + Rare):**
- `rare_animal`: 0.01 ± 0.00
- `wind`: 0.56 ± 0.00
- `bird`: 0.01 ± 0.00

---

## `dominant_frequency_hz`
**Physical Meaning:** Peak of the average spectrum.

**Computation Summary:** argmax of mean magnitude spectrum

**Bioacoustic Rationale:** Identifies the primary frequency of communication for an animal.

**Fraction NaNs:** 0.00%

**Global Typical Range:** [323.00, 10723.54]

**Global Mean/Std:** 4742.69 ± 2789.55

**Per-Class Stats (Top 10 + Rare):**
- `rare_animal`: 1001.29 ± 0.00
- `wind`: 5780.59 ± 3584.70
- `bird`: 4005.18 ± 0.00

---

## `low_band_power_db`
**Fraction NaNs:** 0.00%

**Global Typical Range:** [23.22, 38.35]

**Global Mean/Std:** 30.88 ± 7.24

**Per-Class Stats (Top 10 + Rare):**
- `rare_animal`: 24.08 ± 0.33
- `wind`: 38.02 ± 0.23
- `bird`: 23.69 ± 0.23

---

## `mid_band_power_db`
**Fraction NaNs:** 0.00%

**Global Typical Range:** [28.23, 55.66]

**Global Mean/Std:** 36.97 ± 8.31

**Per-Class Stats (Top 10 + Rare):**
- `rare_animal`: 55.66 ± 0.00
- `wind`: 42.80 ± 0.08
- `bird`: 28.43 ± 0.11

---

## `high_band_power_db`
**Fraction NaNs:** 0.00%

**Global Typical Range:** [34.47, 55.70]

**Global Mean/Std:** 51.20 ± 5.14

**Per-Class Stats (Top 10 + Rare):**
- `rare_animal`: 34.53 ± 0.08
- `wind`: 48.83 ± 0.04
- `bird`: 55.69 ± 0.01

---

## `low_high_power_ratio_db`
**Physical Meaning:** Ratio of low to high frequency band power.

**Computation Summary:** low_band_power_db - high_band_power_db

**Bioacoustic Rationale:** Measures the relative balance between low and high frequencies.

**Fraction NaNs:** 0.00%

**Global Typical Range:** [-32.46, -10.16]

**Global Mean/Std:** -20.32 ± 10.70

**Per-Class Stats (Top 10 + Rare):**
- `rare_animal`: -10.45 ± 0.41
- `wind`: -10.80 ± 0.25
- `bird`: -32.00 ± 0.23

---

## `spectral_entropy`
**Physical Meaning:** Shannon entropy of the normalized magnitude spectrum.

**Computation Summary:** scipy.stats.entropy of normalized spectrum.

**Bioacoustic Rationale:** Measures the peakiness or flatness of the spectrum.

**Fraction NaNs:** 0.00%

**Global Typical Range:** [5.77, 6.93]

**Global Mean/Std:** 6.36 ± 0.58

**Per-Class Stats (Top 10 + Rare):**
- `rare_animal`: 5.78 ± 0.00
- `wind`: 6.93 ± 0.00
- `bird`: 5.78 ± 0.00

---

## `bandwidth_hz`
**Physical Meaning:** Frequency range containing 90% of spectral energy.

**Computation Summary:** Difference between the 95th and 5th percentiles of cumulative spectral energy.

**Bioacoustic Rationale:** Differentiates narrowband whistles from broadband clicks/noise.

**Fraction NaNs:** 0.00%

**Global Typical Range:** [10.77, 9991.41]

**Global Mean/Std:** 4969.06 ± 5020.40

**Per-Class Stats (Top 10 + Rare):**
- `rare_animal`: 21.53 ± 0.00
- `wind`: 9926.27 ± 25.06
- `bird`: 10.77 ± 0.00

---

## `clip_duration_seconds`
**Physical Meaning:** The total duration of the clip.

**Computation Summary:** Retrieved from manifest.

**Bioacoustic Rationale:** Contextual length of the vocalization event.

**Fraction NaNs:** 0.00%

**Global Typical Range:** [1.00, 1.00]

**Global Mean/Std:** 1.00 ± 0.00

**Per-Class Stats (Top 10 + Rare):**
- `rare_animal`: 1.00 ± 0.00
- `wind`: 1.00 ± 0.00
- `bird`: 1.00 ± 0.00

---

## `rms_energy_db`
**Physical Meaning:** Root mean square of the audio signal converted to dB.

**Computation Summary:** 10 * log10(mean(RMS)^2) using librosa.feature.rms.

**Bioacoustic Rationale:** General measure of signal loudness and intensity.

**Fraction NaNs:** 0.00%

**Global Typical Range:** [-6.61, -3.23]

**Global Mean/Std:** -4.90 ± 1.67

**Per-Class Stats (Top 10 + Rare):**
- `rare_animal`: -3.24 ± 0.00
- `wind`: -6.55 ± 0.03
- `bird`: -3.24 ± 0.01

---

## `attack_time_seconds`
**Physical Meaning:** Time to reach peak envelope from start.

**Computation Summary:** argmax of librosa onset strength converted to time.

**Bioacoustic Rationale:** Helps distinguish percussive/impulsive sounds from sustained ones.

**Fraction NaNs:** 0.00%

**Global Typical Range:** [0.07, 0.07]

**Global Mean/Std:** 0.07 ± 0.00

**Per-Class Stats (Top 10 + Rare):**
- `rare_animal`: 0.07 ± 0.00
- `wind`: 0.07 ± 0.00
- `bird`: 0.07 ± 0.00

---

## `decay_time_seconds`
**Physical Meaning:** Time from peak envelope to the end of the clip.

**Computation Summary:** duration - attack_time_seconds.

**Bioacoustic Rationale:** Measures the fade-out or reverberation characteristics.

**Fraction NaNs:** 0.00%

**Global Typical Range:** [0.93, 0.93]

**Global Mean/Std:** 0.93 ± 0.00

**Per-Class Stats (Top 10 + Rare):**
- `rare_animal`: 0.93 ± 0.00
- `wind`: 0.93 ± 0.00
- `bird`: 0.93 ± 0.00

---

## `zero_crossing_rate`
**Physical Meaning:** Rate at which the signal changes sign.

**Computation Summary:** mean of librosa.feature.zero_crossing_rate.

**Bioacoustic Rationale:** Correlates strongly with the noise component of the signal.

**Fraction NaNs:** 0.00%

**Global Typical Range:** [0.09, 0.49]

**Global Mean/Std:** 0.40 ± 0.10

**Per-Class Stats (Top 10 + Rare):**
- `rare_animal`: 0.09 ± 0.00
- `wind`: 0.48 ± 0.00
- `bird`: 0.35 ± 0.00

---

## `harmonic_to_noise_ratio_db`
**Physical Meaning:** Ratio of harmonic energy to noise energy.

**Computation Summary:** librosa.effects.hpss separation to extract harmonic and percussive parts.

**Bioacoustic Rationale:** Measures the degree of acoustic periodicity (voicedness).

**Fraction NaNs:** 0.00%

**Global Typical Range:** [-0.11, 22.39]

**Global Mean/Std:** 11.13 ± 11.26

**Per-Class Stats (Top 10 + Rare):**
- `rare_animal`: 21.82 ± 0.02
- `wind`: 0.01 ± 0.06
- `bird`: 22.29 ± 0.05

---

## `pitch_mean_hz`
**Physical Meaning:** Fundamental frequency of the signal.

**Computation Summary:** Mean of voiced frames from librosa.pyin pitch tracker.

**Bioacoustic Rationale:** Key for characterizing the melody/structure of vocalizations.

**Fraction NaNs:** 5.00%

**Global Typical Range:** [69.61, 2010.06]

**Global Mean/Std:** 1040.09 ± 954.60

**Per-Class Stats (Top 10 + Rare):**
- `rare_animal`: 999.44 ± 0.28
- `wind`: 74.64 ± 3.11
- `bird`: 2010.06 ± 0.00

---

## `pitch_std_hz`
**Physical Meaning:** Variation in the fundamental frequency.

**Computation Summary:** Standard deviation of voiced frames from librosa.pyin.

**Bioacoustic Rationale:** Represents frequency modulation and variability in a call.

**Fraction NaNs:** 5.00%

**Global Typical Range:** [0.00, 12.50]

**Global Mean/Std:** 3.36 ± 3.96

**Per-Class Stats (Top 10 + Rare):**
- `rare_animal`: 0.73 ± 1.03
- `wind`: 7.01 ± 2.68
- `bird`: 0.00 ± 0.00

---

## `pitch_voiced_fraction`
**Physical Meaning:** Fraction of frames classified as voiced.

**Computation Summary:** Count of voiced frames divided by total frames from librosa.pyin.

**Bioacoustic Rationale:** Determines how much of the signal contains a strong fundamental pitch.

**Fraction NaNs:** 0.00%

**Global Typical Range:** [0.00, 1.00]

**Global Mean/Std:** 0.67 ± 0.37

**Per-Class Stats (Top 10 + Rare):**
- `rare_animal`: 1.00 ± 0.00
- `wind`: 0.34 ± 0.21
- `bird`: 1.00 ± 0.00

---

## `snr_db`
**Physical Meaning:** Signal-to-noise ratio.

**Computation Summary:** Ratio of peak-frame energy to median-frame energy, in dB.

**Bioacoustic Rationale:** Helps filter out low-quality/inaudible recordings.

**Fraction NaNs:** 0.00%

**Global Typical Range:** [0.02, 0.36]

**Global Mean/Std:** 0.13 ± 0.10

**Per-Class Stats (Top 10 + Rare):**
- `rare_animal`: 0.04 ± 0.01
- `wind`: 0.22 ± 0.07
- `bird`: 0.04 ± 0.01

---

## `background_noise_db`
**Physical Meaning:** Background noise energy level.

**Computation Summary:** Median-frame energy, in dB.

**Bioacoustic Rationale:** Helps characterize the environmental noise conditions.

**Fraction NaNs:** 0.00%

**Global Typical Range:** [26.67, 30.05]

**Global Mean/Std:** 28.39 ± 1.67

**Per-Class Stats (Top 10 + Rare):**
- `rare_animal`: 30.04 ± 0.00
- `wind`: 26.74 ± 0.03
- `bird`: 30.04 ± 0.01

---

## `syllable_count`
**Physical Meaning:** Distinct energy bursts in the audio.

**Computation Summary:** Peak picking on onset envelope.

**Bioacoustic Rationale:** Captures the temporal structure of animal calls (e.g., trills, chirps).

**Fraction NaNs:** 0.00%

**Global Typical Range:** [1.00, 2.00]

**Global Mean/Std:** 1.02 ± 0.16

**Per-Class Stats (Top 10 + Rare):**
- `rare_animal`: 1.00 ± 0.00
- `wind`: 1.00 ± 0.00
- `bird`: 1.06 ± 0.24

---

## `syllable_rate_hz`
**Physical Meaning:** Syllables per second.

**Computation Summary:** syllable_count / clip_duration_seconds.

**Bioacoustic Rationale:** Helps distinguish fast-paced vocalizations.

**Fraction NaNs:** 0.00%

**Global Typical Range:** [1.00, 2.00]

**Global Mean/Std:** 1.02 ± 0.16

**Per-Class Stats (Top 10 + Rare):**
- `rare_animal`: 1.00 ± 0.00
- `wind`: 1.00 ± 0.00
- `bird`: 1.06 ± 0.24

---

## `mean_syllable_duration_seconds`
**Physical Meaning:** Average length of a syllable.

**Computation Summary:** Approximated using the temporal spacing of peaks.

**Bioacoustic Rationale:** Distinguishes short clicks from long drawn-out calls.

**Fraction NaNs:** 0.00%

**Global Typical Range:** [0.10, 0.10]

**Global Mean/Std:** 0.10 ± 0.00

**Per-Class Stats (Top 10 + Rare):**
- `rare_animal`: 0.10 ± 0.00
- `wind`: 0.10 ± 0.00
- `bird`: 0.10 ± 0.00

---

## `frequency_modulation_slope_hz_per_s`
**Physical Meaning:** Linear fit slope of dominant frequency over time.

**Computation Summary:** Polynomial fit (degree 1) on the argmax of the magnitude spectrum across time frames.

**Bioacoustic Rationale:** Distinguishes up-sweeps, down-sweeps, and flat tones.

**Fraction NaNs:** 0.00%

**Global Typical Range:** [-3043.45, 4529.25]

**Global Mean/Std:** 436.28 ± 1357.71

**Per-Class Stats (Top 10 + Rare):**
- `rare_animal`: -0.00 ± 0.00
- `wind`: 872.56 ± 1839.31
- `bird`: -0.00 ± 0.00

---
