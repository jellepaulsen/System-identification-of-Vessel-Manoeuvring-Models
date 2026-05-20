# Project Status

## What's missing (private SSPA data)

| Missing | Blocks | Notebooks affected |
|---------|--------|--------------------|
| `vct.csv` — captive test force measurements | Force regression pipeline | 01.02 + any force_regression notebook |
| `data_TT_MDL.csv` — towing tank resistance | Resistance correction | Part of force regression |
| KVLCC2 HSVA raw test CSVs | kvlcc2_hsva pipeline | All 25.* notebooks (13) |
| Test 22774 (circle test) | Some accuracy notebooks | 06.01, 07.01, 07.10, 08.01, 08.02 |

## What works now

| Category | Status |
|----------|--------|
| wPCC pipeline (155/155 tasks) | Full run with 6 test runs, 3 VMMs |
| Motion regression (all 3 VMMs) | vmm_linear, vmm_martins_simple, vmm_abkowitz |
| EK filtering + smoothing | All 6 test runs |
| Simulation + prediction | All combinations |
| Catalog loading in notebooks | Works with `%load_ext` fix |

## Notebooks

| Status | Count | Details |
|--------|-------|---------|
| Works | ~11 | 01.01, 03.01, 08.02_monte_carlo, 10.*, 22.01, 24.01, 30.01, 31.01 |
| Minor fix (change VMM name/test ID) | ~7 | Reference `vmm_martin` → change to `vmm_martins_simple` |
| Missing data | ~11 | Need VCT, test 22774, or KVLCC2 |
| KVLCC2-specific | ~15 | All 25.*, 26.* — need HSVA data |
| docs/paper/ | 20 | Mixed — wPCC ones likely work, KVLCC2 ones don't |

## Bottom line

The motion regression workflow is fully functional for wPCC. What you can't do is force regression (needs captive test data) and KVLCC2 analysis (needs HSVA private data). Those are fundamentally different datasets that weren't in the public Mendeley download.
