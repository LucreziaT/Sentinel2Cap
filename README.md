# Sentinel2Cap

Repository for managing, analyzing, and using the **Sentinel2Cap** dataset, which contains captions for remote sensing images, both manually annotated and automatically generated with Qwen3-VL-8B-Instruct.

---

## 📁 Project Structure
├── scripts/ # Training, inference, and utility scripts
├── script_dataset/ # Dataset analysis and statistics scripts
├── Sentinel2Cap.zip # 12k manually annotated captions
├── Sentinel2Cap.parquet # Structured dataset metadata
├── Qwen3-VL-8B-Instruct_... # Outputs from two studies (different prompts)
├── install_flash_attn.sh # Flash Attention installation script
├── pyproject.toml # Project dependencies
├── .python-version
└── .gitignore

---

## 📊 Dataset

### Sentinel2Cap.zip
Contains **12,000 manually annotated captions** associated with Sentinel-2 RGB, Sentinel-2 multi-spectral and Sentinel-1 SAR images with a pseudo-RGB representations.

---

### Sentinel2Cap.parquet

File containing structured metadata for each dataset sample.

#### Columns

- `key`
- `image_index`
- `number_of_classes`
- `number_of_classes_30`
- `file_name`
- `path_to_S2` → path to Sentinel-2 image
- `path_to_SM` → path to reference maps
- `set` → train / val / test
- `used`
- `month`
- `occurrences`
- `s1_name` → associated Sentinel-1 image name

#### Example
key: N9999_R037_T29SNB_16_20
image_index: 431416
number_of_classes: 12
number_of_classes_30: 11
file_name: S2B_MSIL2A_20180326T112109_...
path_to_S2: BigEarthNet-S2/S2B_MSIL2A_20180326T112109_...
path_to_SM: Reference_Maps/S2B_MSIL2A_20180326T112109_...
set: train
used: True
month: march
occurrences: 3
s1_name: S1A_IW_GRDH_1SDV_20180327T064326_29SNB_16_20

---

## 🤖 Model Outputs

### Qwen3-VL-8B-Instruct

This file contains outputs from **two studies** performed using the same model:

- **Model**: Qwen3-VL-8B-Instruct  
- **Main difference**: prompting strategies used for caption generation  

Use cases:
- comparison of prompting strategies  
- qualitative and quantitative analysis of generated captions  

---

## ⚙️ Setup

Install dependencies:

```bash
pip install -e .
```

---

## 🚀 Usage
Dataset Preparation

Make sure that paths in the .parquet file are correctly set:

path_to_S2 → Sentinel-2 images
path_to_SM → reference maps
Training / Inference

Main scripts are located in:

scripts/

Examples:

python scripts/train.py
python scripts/inference.py
Dataset Analysis

Scripts available in:

script_dataset/

Useful for:

class distribution analysis
temporal distribution
caption analysis

---

## 📌 Notes

The dataset combines information from:

Sentinel-2 (RGB imagery)
Sentinel-2 (multi-spectral imagery)
Sentinel-1 (SAR imagery)
reference land cover maps

Manually annotated captions can be used as:

ground truth
benchmark for generative models

---

## 📄 License

Specify here the license for the dataset and code.

---

## ✉️ Contact

Lucrezia Tosato: ltosato (at) sarmap.ch
Gianluca Lombardi: gianluca.lombardi.fr (at) gmail.com
Ronny Hansch: rww.haensch (at) gmail.com

---

## ✅ Citation: 
The paper is under review; for the moment, please use the following citation:
xxx
