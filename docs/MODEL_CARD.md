# Model card: HimSat surface segmentation (Prithvi-EO-2.0 fine-tune)

## Overview
- **Task:** per-pixel classification of Sentinel-2 L2A scenes into *other*, *water*, *snow/ice*,
  *debris-covered ice*. Used to map glacial lakes and ice extents.
- **Backbone:** `ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL`, a ViT-B/16 pretrained by NASA/IBM
  with masked autoencoding on HLS (Harmonised Landsat Sentinel-2) imagery, 6 bands (B02, B03,
  B04, B8A, B11, B12). The temporal and location encoders are unused (single date, any place).
- **Head:** four encoder depths (blocks 4/6/8/12) projected and fused at 1/16 resolution, then
  four ×2 conv upsampling stages. The last 4 encoder blocks are fine-tuned; the rest are frozen.
- **Inference:** 224 px windows with a 160 px stride, Hann-weighted blending. The physics veto
  removes "water" on slopes > 25°. Clouds and no-data come from the Sentinel-2 scene
  classification (SCL).
- **Code:** `himsat/ml/prithvi.py`, `himsat/ml/dataset.py`, `himsat/ml/train.py`.

## Training data (weak supervision)
No openly licensed, pixel-labelled Himalayan glacial-lake dataset matches Sentinel-2 at 10 m.
Labels are therefore assembled from sources that are reliable *where confident*:

| Class | Label source |
|---|---|
| water | spectral rules, probability ≥ 0.85, flat terrain (DEM slope), shadow-aware |
| snow/ice | spectral rules (NDSI, NIR), probability ≥ 0.85 |
| debris-covered ice | inside OpenStreetMap `natural=glacier` outlines (eroded by 3 px), neither snow nor water |
| other | spectral rules, probability ≥ 0.85, ≥ 3 px from glacier outlines |
| *ignored* | everything else: uncertain pixels, class seams, clouds, no data |

Scenes: the clearest post-monsoon Sentinel-2 acquisitions (Oct–Dec 2025) over four Nepal AOIs
(Rasuwa/Lhende, Rolwaling/Tsho Rolpa, Khumbu/Imja, Manaslu/Thulagi). The split is spatial: ~9 km
blocks go to train or validation, never both.

## Evaluation

**Training run v1** (2026-10-01): 1,009 train / 281 validation patches (224 px) from 8 scenes
over 4 AOIs. 10 epochs, batch 8, AdamW, AMP on an RTX 4050 laptop GPU (peak 1.3 GB, ~3 min per
epoch). Best epoch 8:

| class | IoU vs held-out weak labels |
|---|---|
| other | 0.895 |
| water | 0.610 |
| snow/ice | 0.931 |
| debris-covered ice | 0.368 |
| **mean** | **0.701** |

Validation labels are weak labels of the same kind. These scores measure agreement with
confident rule and inventory labels on unseen areas, not ground truth. The debris-ice score is
limited by outline quality: OSM glacier outlines are often older and larger than today's ice.

**Independent check: lake areas vs published values** (`himsat ml benchmark`; clear Oct–Dec
2025 scenes; published values are rounded literature values from different years, so ±5 % is
noise):

| lake | published km² | rules | Prithvi | rules + SAM | Prithvi + SAM |
|---|---|---|---|---|---|
| Gosainkunda | 0.138 | 0.106 (−23 %) | 0.126 (−9 %) | 0.124 (−10 %) | 0.123 (−11 %) |
| Tsho Rolpa | 1.55 | 1.589 (+2 %) | 1.761 (+14 %) | 1.617 (+4 %) | 1.616 (+4 %) |
| Imja Tsho | 1.30 | 1.216 (−6 %) | 1.471 (+13 %) | 1.266 (−3 %) | 1.416 (+9 %) |
| Thulagi | 0.95 | 0.778 (−18 %) | 0.887 (−7 %) | 0.784 (−18 %) | 0.786 (−17 %) |
| **mean abs. error** | | 12.6 % | 10.6 % | **8.6 %** | 10.3 % |

Interpretation: on dark or turbid lakes, which are hard for rules, Prithvi is clearly better
(Gosainkunda, Thulagi). It overestimates large lakes by ~13 %, probably by including wet
shoreline and lake ice. SAM refinement of rule-based outlines gives the lowest error overall.
**Recommendation:** use rules + SAM for lake-area time series. Use Prithvi where debris-covered
ice mapping is needed, and evaluate per region. Four lakes is a small sample: extend
`himsat/ml/benchmark.py` with surveyed outlines before relying on these numbers.

## Intended use and limitations
- Use it to map glacial lakes ≥ 0.005 km² and glacier ice in the Himalaya for monitoring. It is
  **not** a substitute for field survey or for glacier inventories.
- The debris-ice class inherits the quality of the OSM outlines, which are often traced from older
  imagery. Glacier retreat makes old outlines too large.
- Frozen or snow-covered lakes (winter) are not reliably detected by either the rules or the
  model. HimSat uses radar for lake area during those months.
- If the checkpoint is missing or fails to load, HimSat falls back to the spectral rules and logs
  an error. Monitoring never stops.

## Licences
Prithvi-EO-2.0: Apache-2.0. Training imagery: Copernicus Sentinel data. Glacier outlines:
© OpenStreetMap contributors (ODbL).
