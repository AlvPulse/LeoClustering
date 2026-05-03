# Extraction Diagnostics

## Dataset: esc50
- Total Runtime: 9.55 s
- Throughput: 2.10 clips/s
- Failures: 0
- Feature Group Timings:
  - spectral: 1.8186 s
  - temporal: 0.0183 s
  - harmonic: 1.2376 s
  - quality: 0.0005 s
  - bioacoustic: 0.0085 s
### ⚠️ SLOW FEATURE GROUPS ( > 10x median cost)
- **spectral**: 1.8186 s
- **harmonic**: 1.2376 s

## Dataset: dcase2024_t5
- Total Runtime: 3.20 s
- Throughput: 6.26 clips/s
- Failures: 0
- Feature Group Timings:
  - spectral: 0.0087 s
  - temporal: 0.0270 s
  - harmonic: 0.5241 s
  - quality: 0.0004 s
  - bioacoustic: 0.0081 s
### ⚠️ SLOW FEATURE GROUPS ( > 10x median cost)
- **harmonic**: 0.5241 s
