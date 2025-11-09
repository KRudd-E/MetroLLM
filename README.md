# MetroLLM: Case Study Classification & Generation Pipeline

## Paper Abstract

3D Measurement and Metrology supports crucial fields across manufacturing, engineering, and scientific research, yet the associated technical documentation is often verbose and inconsistently structured. The Explore 3D Metrology project aims to digitalize the industry by providing a variety of resources, including an applications database containing 2,183 case studies, application reports and user stores. This work analysed how technologies in Natural Language Processing can understand and categorise entries in this database. Three approaches were taken to automatically identify measurement tasks: (i) a transformer-based text classifier - 

ModernBERT, (ii) a fine-tuned large language model - DeepSeek-R1 with LoRA adapters, and (iii) an API-hosted generative model to act as a baseline. Across recall, precision and F1-score, both generation approaches outperformed the classifier, whose performance was owed to label imbalance, noisy classification, and long inputs. The API-based model achieved the most reliable extractions, with a weighted F1-score of 0.518, while the locally fine-tuned generator produced a comparable 0.456 but suffered from complex outputs and occasional refusal to answer. These findings indicate that modern language models can generalise to 3D measure-
ment and metrology data, but effective learning required a larger and more curated dataset. 

Overall, prompt-driven text generation was identified as the most practical solution: it avoided retraining overheads, extended easily to new fields (e.g., sector or tolerance extraction), and integrated with existing infrastructure through a lightweight API. This work demonstrates the suitability of applying modern NLP to 3D measurement documentation and outlines a process for automated classification in the Explore 3DM applications database.


## Overview

MetroLLM is a modular pipeline for processing, formatting, and modeling case studies from the applications database, with support for definitions and company data. The project enables the transformation of raw documents (PDFs, Word files, websites) into structured datasets, which are then used to train and evaluate machine learning models for text classification, text generation, and text2text generation. The overarching aim is to classify case studies from the applications database.

---

## Workflow Summary

### 1. DataDev1: Raw Data Processing

**Purpose:**  
Convert raw data sources into usable formats.

- **Applications:** PDF files → `.txt` files (text extraction, OCR, image extraction)
- **Definitions:** Word documents → `.csv` files (structured definitions)
- **Companies:** Website scraping → `.txt` files

**How to Run:**
```sh
conda activate <your_env>
python dataDev1
```
- Configure `dataDev1/config.yaml` as needed.
- Select data source via command-line argument (`--data_source`).

---

### 2. DataDev2: Data Formatting

**Purpose:**  
Format usable data for model development.

- **Text2Text Generation:**  
  - Applications `.txt` → t2t pairs via OpenAI API  
  - Definitions `.csv` → t2t pairs via OpenAI API  
  - Companies `.txt` → t2t pairs (not implemented)
- **Text Generation:**  
  - Applications `.txt` → tg pairs via OpenAI API  
  - Definitions `.csv` → tg pairs via OpenAI API  
  - Companies `.txt` → tg pairs (not implemented)
- **Text Classification:**  
  - Applications `.txt` & `.xlsx` → `.csv` for classification  
  - Companies `.txt` & `.xlsx` → `.csv` for classification

**How to Run:**
```sh
conda activate <your_env>
python dataDev2
```
- Configure `dataDev2/config.yaml`.
- Choose model type and data source via command-line arguments (`--model_type`, `--data_source`).

---

### 3. ModelDev: Model Training & Evaluation

**Purpose:**  
Train and evaluate ML models using formatted data.

- **Text2Text Generation Model:**  
  - Uses all sources (applications, definitions, companies)
- **Text Generation Model:**  
  - Uses all sources
- **Text Classification Model:**  
  - Uses source-specific data (applications or companies)

**How to Run:**
Each model has its own directory and entry point:
- **Text2Text Generation:**  
  ```sh
  python modelDev-text2textGen
  ```
- **Text Generation:**  
  ```sh
  python modelDev-textGen
  ```
- **Text Classification:**  
  ```sh
  python modelDev-textClass-A
  ```
- **Pure API Classification:**  
  ```sh
  python modelDev-pureAPI-mature
  ```
- Configure each model's `config.yaml` as needed.

---

## Directory Structure

- `dataDev1/` — Raw data extraction and conversion scripts
- `dataDev2/` — Data formatting and pair generation scripts
- `modelDev-text2textGen/` — Text2Text generation model code and results
- `modelDev-textGen/` — Text generation model code and results
- `modelDev-textClass-A/` — Text classification model code and results
- `modelDev-pureAPI-mature/` — API-based classification pipeline
- `Training Analysis/` — Analysis notebooks and results

---

## Key Functionality

- **PDF/Text Extraction:**  
  Extracts text and images from PDFs, applies OCR if needed ([`Applications_Reformat`](dataDev1/src/applications.py)).
- **Definition Parsing:**  
  Parses Word documents into structured CSVs ([`Definitions_Reformat`](dataDev1/src/definitions.py)).
- **Data Formatting:**  
  Generates training/evaluation pairs for various ML tasks ([`Text2TextGen`](dataDev2/src/text2textGen.py), [`TextGen`](dataDev2/src/textGen.py), [`TextClass`](dataDev2/src/textClass.py)).
- **Model Training/Evaluation:**  
  Fine-tunes and evaluates models for classification and generation ([`FinetunePipeline`](modelDev-textGen/src/pipeline.py), [`FineTunePipeline`](modelDev-textClass-A/src/pipeline.py), [`Pipeline`](modelDev-pureAPI-mature/src/pipeline.py)).
- **Interactive Confirmation:**  
  Prompts user for confirmation before major processing steps ([`dataDev1_query`](dataDev1/src/utils.py), [`dataDev2_query`](dataDev2/src/utils/utils.py)).

---

## Configuration

- Each stage uses its own YAML config file (`config.yaml`).
- Paths, model parameters, and options are set in these config files.

---

## Notes

- All scripts prompt for confirmation before running major operations.
- Some features (e.g., company data processing for generation) are not yet implemented.
- For troubleshooting in HPC environments, see [`disable_compilation.py`](modelDev-textClass-A/src/utils/disable_compilation.py).

---

## Example Run

```sh
# Step 1: Process raw data
python dataDev1 --data_source Applications

# Step 2: Format data for Text Classification
python dataDev2 --model_type textclass --data_source applications

# Step 3: Train Text Classification Model
python modelDev-textClass-A --run train

# Step 4: Evaluate Model
python modelDev-textClass-A --run evaluate
```

---

## Overarching Aim

The primary goal of MetroLLM is to enable robust classification of case studies from the applications database, supporting further research and development in automated document understanding.
