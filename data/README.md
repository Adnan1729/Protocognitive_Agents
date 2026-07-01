# Data

## `raw/meta_data.csv`

The five raw proteinoid spike train datasets recorded with voltage-sensitive
dyes (VSDs) as described in Section I of the paper.

### Format

The file is a wide CSV with five `(Time, Spike Value)` column pairs:

| Data1   | (col B)         | Data2   | (col E)         | Data3   | ... | Data5   | (col N)         |
| ------- | --------------- | ------- | --------------- | ------- | --- | ------- | --------------- |
| Time(s) | Spike Value 0/1 | Time(s) | Spike Value 0/1 | Time(s) | ... | Time(s) | Spike Value 0/1 |
| 0.000   | 0               | 0.000   | 0               | ...     | ... | ...     | ...             |

- Row 1: dataset headers (`Data1`, blank, `Data2`, blank, ...) every two columns.
- Row 2: sub-headers (`Time (s)`, `Spike Value (0/1)`) repeated for each dataset.
- Row 3 onwards: numeric data. Each dataset pair has its own length, so the
  shorter columns are padded with empty cells in the CSV.

### Provenance

Proteinoid microspheres were prepared from L-Aspartic acid, L-Histidine,
L-Phenylalanine, L-Glutamic acid, and L-Lysine following the protocol in
Section I of the paper. Electrical activity was recorded with an ADC-24
data logger (Pico Technology, UK) at 600 samples/second and digitised into
0/1 spike indicators via voltage-sensitive aminonaphthylethenylpyridinium
dyes.

For further details on sample preparation, see:

- Sharma et al., *On morphological and functional complexity of proteinoid
  microspheres*, arXiv:2306.11458 (2023).
- Sharma et al., *A review on the protocols for the synthesis of proteinoids*,
  arXiv:2212.02261 (2022).