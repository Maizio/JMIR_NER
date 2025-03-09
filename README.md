# Prompt Framework for Extracting Scale-Related Knowledge Entities from Chinese Medical Literature

## Overview
This study introduces MedScaleNER, a task-oriented prompt framework designed to enhance LLM performance in recognizing medical scale–related entities from Chinese medical literature. The framework integrates SimCSE semantic similarity computation, KNN nearest-neighbour search and large language model (LLM) inference in a multi-step approach. This approach first predicts the entity type and then performs entity extraction to improve the accuracy of entity recognition.

## Key Features
- **Compute KNN Similarity Using SimCSE**: Calculate text similarity between test and training data using SimCSE.
- **Predict Entity Types in Text**: Identify potential entity types in the text before entity extraction.
- **Generate MRC-Based Prompts**: Construct Machine Reading Comprehension (MRC) prompts based on predicted types and KNN neighbors.
- **Perform Entity Extraction with LLM**: Use large language models such as Zhipu AI for entity recognition.
- **Validate and Optimize Results**: Post-process LLM-generated results to verify entity classification accuracy.

## Installation and Usage

### 1. Requirements
This project requires Python 3.8 or later and the following Python libraries:
```bash
pip install numpy faiss-cpu simcse tqdm
```
If using GPU for computation, install `faiss-gpu`:
```bash
pip install faiss-gpu
```

### 2. Execution Steps

#### **1. Compute KNN**
Run `extract_mrc_knn.py` to compute KNN neighbors between test and training data:
```bash
python extract_mrc_knn.py --model_path <model_path> --data_path <data_directory> --knn_num 20
```

#### **2. Predict Entity Types**
Run `get_type_results.py` to classify entity types:
```bash
python get_type_results.py --model_path <model_path> --data_path <data_directory>
```

#### **3. Generate Prompts and Extract Entities**
Use `get_results_mrc_knn.py` for LLM inference:
```bash
python get_results_mrc_knn.py --bigmodel zhipu --example_num 3 --data_path <data_directory>
```

#### **4. Validate Results**
Run `verify_results.py` to validate entity extraction results:
```bash
python verify_results.py --mrc-dir <data_directory> --gpt-dir <results_directory>
```

## Directory Structure
```
├── main_code/extract_mrc_knn.py       # Compute KNN neighbors
├── main_code/get_type_results.py      # Predict entity types
├── main_code/get_results_mrc_knn.py   # Generate prompts and extract entities
├── main_code/verify_results.py        # Validate results
├── data set/                          # Store datasets
├── models/                            # Models
└── README.md                          # Project documentation
```

---
For any issues or suggestions, feel free to submit an issue or contact the author! 🚀

