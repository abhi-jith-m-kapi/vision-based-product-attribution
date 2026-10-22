# Vision-Based Product Attribution and BOM Mapping

A Streamlit application for analyzing mechanical-part images. It detects parts with YOLO, extracts visual attributes with Amazon Bedrock, finds similar catalog images with Amazon Nova embeddings and ChromaDB, and builds a BOM summary.

For step-by-step user and operator instructions, see [RUNBOOK.md](RUNBOOK.md).

The optional MPRNet checkout and RealESRGAN weight are local extras; the Streamlit app does not require them.

## Project layout

```text
.
├── data/                  # local datasets, crops, and ChromaDB indexes
│   └── archives/          # source dataset archives
├── models/                # YOLO weights used by the app
├── notebooks/             # exploration and experiment notebooks
├── src/                   # detection, embeddings, retrieval, and attributes
├── test_images/           # sample images for the app
├── streamlit_app.py       # Streamlit entry point
├── dev.sh                 # convenience launcher
├── requirements.txt
└── README.md
```

## Set up

Use Python 3.11 or newer. From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `BEDROCK_API_KEY`. `AWS_REGION` defaults to `us-east-1`. Keep `.env` private; it is excluded from version control.

## Run the Streamlit app

From the project root, with the virtual environment active:

```bash
streamlit run streamlit_app.py
```

Or use the convenience launcher:

```bash
./dev.sh
```

Open the local URL printed by Streamlit, usually <http://localhost:8501>. Upload a mechanical-part image, select detections, and review the extracted attributes, similar catalog matches, and BOM summary.

The app expects these local assets:

- `models/machine_parts_best.pt` for YOLO detection. Set `YOLO_MODEL_PATH` to use a different checkpoint.
- `data/chroma_nova_db/` with the `mechanical_parts_nova` collection for catalog similarity search.
- A valid Bedrock API key for Claude attribute extraction and Nova image embeddings.

The dataset and ChromaDB files are local assets and are not included in Git. If the ChromaDB collection is missing, detection can still run, but similarity retrieval will be unavailable.

## Notebooks

All notebooks are in `notebooks/`. Start Jupyter from the project root so their relative paths to `data/` work:

```bash
jupyter lab
```

Some notebooks use Bedrock or OpenAI and need the corresponding keys in `.env`. The notebooks are experiments; the Streamlit app is the main user interface.
