# Hinglish Sentiment Analysis
Code-switched Hinglish sentiment classification using XLM-R, comparing Full Fine-Tuning vs PEFT/LoRA across two frameworks: HuggingFace Trainer and manual PyTorch.

---

## Dataset
This project uses the **SemEval 2020 Task 9 — SentiMix** dataset (Hinglish subtask).

Download the following files from the [CodaLab competition page](https://competitions.codalab.org/competitions/20654) under **Participate → Files**:
- `Hinglish_train_14k_split_conll.txt`
- `Hinglish_dev_3k_split_conll.txt`
- `Hinglish_test_unlabelled_conll_updated.txt`

Place all three files in the project root directory.

---

## Project Structure
```
project/
├── preprocessing.ipynb               # Data loading, normalization, tokenization
├── groupname_huggingface.ipynb       # HuggingFace Trainer — Full FT + LoRA
├── groupname_pytorch.ipynb           # Manual PyTorch loop — Full FT + LoRA
├── peft_implementation.py            # Custom LoRA implementation
└── requirements.txt
```

---

## Setup

### 1. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **GPU note:** The default `pip install torch` installs the CPU-only version. For GPU support, install PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) with the correct CUDA version for your system before running the above command.

---

## Running the Project

### Step 1 — Preprocessing
Run `preprocessing.ipynb` top to bottom. This will:
- Parse the CoNLL data files
- Normalize Hinglish text
- Tokenize using XLM-R tokenizer
- Save `train_tok`, `val_tok`, and `test_tok` to disk

### Step 2 — Training
Run either or both training notebooks. Both load from the saved splits produced in Step 1.

**HuggingFace Trainer:**
Run `groupname_huggingface.ipynb` top to bottom. Trains and saves:
- `./hugging_full_ft_model`
- `./hugging_peft_model`

**Manual PyTorch:**
Make sure `peft_implementation.py` is in the same directory, then run `groupname_pytorch.ipynb` top to bottom. Trains and saves:
- `./pytorch_full_ft_model`
- `./pytorch_peft_model`

### Step 3 — Evaluation
Evaluation runs automatically at the end of each training notebook:
- **Part 3** — Classification report (precision, recall, F1) on the validation set
- **Part 4** — Predictions on the unlabelled test set + sample sentence inference

---

## Running on Google Colab

1. Upload the three CoNLL data files and `peft_implementation.py` to your Colab session or mount Google Drive
2. Install missing dependencies at the top of each notebook:
```python
!pip install peft sentencepiece accelerate -q
```
3. In `groupname_huggingface.ipynb`, change `bf16=True` to `fp16=True` if using a free tier T4 GPU
4. Update file paths in `preprocessing.ipynb` if loading from Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
train_df = parse_conll("/content/drive/MyDrive/Hinglish_train_14k_split_conll.txt")
```

---

## Hardware
Tested on a local GPU. Training times per framework (5 epochs, XLM-R base):
| Approach | HuggingFace | PyTorch |
|---|---|---|
| Full Fine-Tuning | ~5 min | ~5 min |
| PEFT/LoRA | ~3.5 min | ~3.5 min |

CPU training is supported but will be significantly slower.
