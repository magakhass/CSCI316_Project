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
├── requirements.txt                  # Full dependencies for training
└── app/
    ├── app.py                        # FastAPI inference API
    ├── Dockerfile
    ├── requirements-inference.txt    # Minimal dependencies for inference only
    └── templates/
        └── index.html                # Web UI
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

After training, merge and save the LoRA weights for use in the inference API:
```python
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("./app/hugging_peft_model_merged")
tokenizer.save_pretrained("./app/hugging_peft_model_merged")
```

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

## Docker — Inference API

The project includes a FastAPI inference API with a web UI, served via Docker using the merged HuggingFace LoRA model.

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed and running
- The merged model folder `app/hugging_peft_model_merged/` present (produced by the merge step in Step 2 above)

> **Note:** Model files are excluded from git via `.gitignore` due to their size (~1GB). You must train and merge the model locally before building the Docker image.

### Step 1 — Build the image
From the `app/` directory:
```bash
cd app
docker build -t hinglish-sentiment-api .
```

### Pull from DockerHub (instead of step 1 - recommended)
```bash
docker pull mgkh/hinglish-sentiment-api:latest
docker run -p 8000:8000 yourusername/hinglish-sentiment-api:latest
```

### Step 2 — Run the container
```bash
docker run -p 8000:8000 hinglish-sentiment-api
```

Open `http://localhost:8000` in your browser to use the web UI.

### Step 3 — Test the API

**Web UI:**
Open `http://localhost:8000` — enter a Hinglish sentence and click Analyze.

**Using curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "yaar aaj ka din bahut accha tha"}'
```

**Expected response:**
```json
{
  "text": "yaar aaj ka din bahut accha tha",
  "sentiment": "positive",
  "confidence": 0.915
}
```

**Health check:**
```bash
curl http://localhost:8000/health
```

**Swagger UI (API docs):**
Open `http://localhost:8000/docs`

### API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web UI |
| GET | `/health` | Health check |
| POST | `/predict` | Predict sentiment for a Hinglish sentence |
| GET | `/docs` | Swagger UI |

---

## Hardware
Tested on a local GPU. Training times per framework (5 epochs, XLM-R base):
| Approach | HuggingFace | PyTorch |
|---|---|---|
| Full Fine-Tuning | ~5 min | ~5 min |
| PEFT/LoRA | ~3.5 min | ~3.5 min |

CPU training is supported but will be significantly slower.
