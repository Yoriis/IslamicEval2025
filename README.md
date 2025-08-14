# IslamicEval2025: Islamic Question Answering Evaluation Framework

## Overview

IslamicEval2025 is a comprehensive framework for evaluating and fine-tuning retrieval models for Islamic Question Answering (QA) tasks. This project implements state-of-the-art sentence transformer models and evaluation methodologies specifically designed for Arabic Islamic texts, including Quran verses and Hadith passages.

## Project Structure

```
IslamicEval2025/
├── data/                          # Data directory
│   ├── External Data/             # External datasets (HAQA, QUQA, Tafseer, TyDiQA)
│   └── Task Data/                 # Main task datasets
│       ├── data/                  # QH-QA-25 datasets (dev, test, train)
│       ├── qrels/                 # Query relevance judgments
│       ├── Sahih-Bukhari/         # Hadith dataset
│       └── Thematic_QPC/          # Quran passages
├── src/                           # Source code
│   ├── retrieval/                 # Core retrieval scripts
│   │   ├── data_preprocessing.py  # Data preprocessing pipeline
│   │   ├── finetune_baseline_models.py  # Baseline models fine-tuning
│   │   ├── finetune_main_model.py # Main model fine-tuning
│   │   └── evaluate_models.py     # Model evaluation
│   ├── Cross-encoder/             # Cross-encoder implementations
│   ├── Evaluation/                # Evaluation utilities
│   ├── Fine-tuning/               # Fine-tuning implementations
│   ├── Gemini/                    # Gemini model integration
│   └── Utils/                     # Utility functions
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Features

- **Multi-Modal Data Support**: Handles Quran verses, Hadith passages, and thematic content
- **Baseline Model Fine-tuning**: Supports 35+ pre-trained Arabic and multilingual models
- **Advanced Loss Functions**: Implements contrastive, cosine similarity, and multiple negatives ranking losses
- **Comprehensive Evaluation**: FAISS-based retrieval evaluation with multiple metrics
- **Arabic Language Optimization**: Specifically designed for Arabic Islamic texts

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 16GB+ RAM (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/IslamicEval2025.git
   cd IslamicEval2025
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download required data files**
   - Place your data files in the appropriate directories under `data/`
   - Ensure the following files are present:
     - `data/Task Data/data/combined_questions_with_passages_train.tsv`
     - `data/Task Data/data/combined_questions_with_passages_dev.tsv`
     - `data/Task Data/Thematic_QPC/QH-QA-25_Subtask2_QPC_v1.1.tsv`
     - `data/Task Data/Sahih-Bukhari/QH-QA-25_Subtask2_Sahih-Bukhari_v1.0.jsonl`

## Usage

### 1. Data Preprocessing

Generate fine-tuning datasets with different top-k configurations:

```bash
python src/retrieval/data_preprocessing.py
```

**Note**: All scripts can be run from the project root directory.

This script will:
- Load Quran and Hadith passages
- Encode questions and passages using NAMAA-Space/AraModernBert-Base-STS
- Generate datasets with top-k values: [60, 70, 80, 90]
- Create positive-only and balanced datasets

**Output**: `finetune_data_final_ara_60-90/` directory containing:
- `finetune_dataset_posonly.tsv`
- `finetune_dataset_top60.tsv`
- `finetune_dataset_top70.tsv`
- `finetune_dataset_top80.tsv`
- `finetune_dataset_top90.tsv`

### 2. Fine-tuning Baseline Models

Fine-tune 35+ baseline models using your dataset:

```bash
python src/retrieval/finetune_baseline_models.py
```

**Note**: All scripts can be run from the project root directory.

**Features**:
- Supports 35 pre-trained models from the PDF evaluation
- Uses both contrastive and cosine similarity losses
- Automatically handles different file formats (CSV/TSV)
- Memory-efficient training with GPU cleanup

**Output**: `baseline_models_finetuned/` directory with model checkpoints

### 3. Fine-tuning Main Model

Fine-tune the main NAMAA retriever model:

```bash
python src/retrieval/finetune_main_model.py
```

**Note**: All scripts can be run from the project root directory.

**Features**:
- Fine-tunes `yoriis/NAMAA-retriever-tydi-tafseer-quqa-haqa-cos`
- Supports multiple loss functions: contrastive, cosine_similarity, mnrl
- Processes multiple dataset configurations
- Comprehensive training monitoring

**Output**: `results/` directory with fine-tuned models

### 4. Model Evaluation

Evaluate all fine-tuned models:

```bash
python src/retrieval/evaluate_models.py
```

**Note**: All scripts can be run from the project root directory.

**Features**:
- FAISS-based similarity search
- Multiple similarity types: cosine and L2
- Top-k evaluation: [20, 30, 40, 50, 60, 70, 80]
- Comprehensive metrics: Precision, Recall, F1
- Automatic checkpoint discovery and evaluation

**Output**: `model_evaluation_results.csv` with detailed evaluation metrics

## Baseline Models

The framework supports 35+ pre-trained models including:

- **NAMAA-Space/AraModernBert-Base-STS** (Best performing)
- **silma-ai/silma-embeddding-sts-v0.1**
- **omarelshehy/Arabic-Retrieval-v1.0**
- **Omartificial-Intelligence-Space/GATE-AraBert-v1**
- **Alibaba-NLP/gte-multilingual-base**
- **sentence-transformers/LaBSE**
- And 30+ more models...

## Data Format

### Input Data Structure

**Questions Dataset** (TSV format):
```tsv
question_text    relevant_passages
"What is...?"   ["passage_id_1", "passage_id_2"]
```

**Passages Dataset**:
- **Quran**: TSV with `id` and `passage` columns
- **Hadith**: JSONL with `hadith_id` and `hadith` fields

### Output Data Structure

**Fine-tuning Dataset**:
```tsv
question    passage_id    label
"What is...?"    "passage_1"    1
"What is...?"    "passage_2"    0
```

## Configuration

### Training Parameters

- **Epochs**: 1 (configurable)
- **Batch Size**: 8 (per device)
- **Learning Rate**: 2e-5
- **Warmup Steps**: 100
- **Mixed Precision**: BF16 (enabled)

### Evaluation Parameters

- **Similarity Types**: cosine, L2
- **Top-K Values**: 20, 30, 40, 50, 60, 70, 80
- **Metrics**: Precision, Recall, F1

## Performance

Based on the evaluation results:
- **Best Model**: NAMAA-Space/AraModernBert-Base-STS
- **Performance**: 0.4451 (evaluation metric)
- **Model Size**: 149M parameters

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{islamiceval2025,
  title={IslamicEval2025: A Framework for Islamic Question Answering Evaluation},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NAMAA-Space for the base Arabic models
- Hugging Face for the sentence-transformers library
- FAISS for efficient similarity search
- The Islamic research community for datasets and guidance

## Support

For questions and support:
- Open an issue on GitHub
- Contact the maintainers
- Check the documentation

## Roadmap

- [ ] Support for more Arabic models
- [ ] Cross-encoder fine-tuning
- [ ] Multi-language support
- [ ] Web interface for evaluation
- [ ] Integration with more evaluation frameworks
