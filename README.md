# MedForge

This repository provides the official release page for the **MedForge dataset** and the accompanying forgery detection model. MedForge contains real and forged RGB medical images from multiple medical scenarios, including colon histopathology, fundus imaging, trichoscopy, and dermoscopy.

---

## Overview

Recent generative models have made realistic medical image synthesis and local editing increasingly accessible. This creates new challenges for medical image authenticity verification.

MedForge is designed to evaluate whether a model can distinguish real and forged medical images while remaining robust to:

* Different medical image modalities;
* Different real image sources;
* Different forgery generation or editing methods;
* Unseen real-source distribution shifts;
* Unseen forgery types.

---

## Dataset Release

We currently release **a partial version** of the MedForge dataset for research preview.

The full dataset will be released after the paper is accepted.

We also plan to release an additional **original-resolution version** of MedForge, which preserves the original image sizes before resizing or preprocessing.

| Version                      | Status                  | Description                          |
| ---------------------------- | ----------------------- | ------------------------------------ |
| MedForge-Preview             | Available               | Partial release for research preview |
| MedForge-Full                | Coming after acceptance | Complete dataset used in the paper   |
| MedForge-Original-Resolution | Coming after acceptance | Original-size version of the dataset |

Download links:

```text
MedForge-Preview: [Available]
MedForge-Full: [Released after acceptance]
MedForge-Original-Resolution: [Released after acceptance]
```

---

## Example Images

Example visualization:

<p align="center">
  <img src="examples/medforge_examples.png" width="800">
</p>

---

## Model Release

We will release the accompanying forgery detection model and pretrained checkpoints in this repository.

The model code is intended for research use and will be updated progressively.

| Component              | Status      |
| ---------------------- | ----------- |
| Model code             | Available |
| Pretrained checkpoints | Coming soon |
| Evaluation scripts     | Coming soon |

---

## Usage

After downloading the dataset, place it under:

```text
data/MedForge/
```

The detailed usage instructions for the dataset and model will be provided with the official release.

---


## License

The released data and code are intended for academic research only.

Please do not use MedForge for clinical diagnosis, commercial deployment, or generating misleading medical content.

The full license will be provided with the official dataset release.

---

## Contact

For questions about MedForge, please contact:

```text
Chenquan Gong
13862991030@163.com
```

---

## Disclaimer
MedForge is a research benchmark for medical image forgery detection. It is not intended for clinical decision-making or medical diagnosis.
