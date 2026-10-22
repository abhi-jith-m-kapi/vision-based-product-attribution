"""Vision-Based Product Attribution and BOM Mapping — Streamlit app.

Layout (tabs instead of one long page):
    📤 Upload & Detection  - upload, YOLO detection, detected-part thumbnails
    🧩 Parts               - pick ONE part, then switch between its
                             Attributes and Catalog matches
    📋 BOM summary         - one table for all parts + CSV download

Pipeline (runs automatically after "Identify Parts"):
    1. YOLO detection
    2. In parallel, for every unique detected part:
         - Claude attribute extraction (Bedrock)
         - Nova image embedding (Bedrock)
    3. Catalog retrieval (local ChromaDB) from the cached embeddings

Launch:  streamlit run streamlit_app.py
"""

import csv
import html
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from PIL import Image

from src import attribute_extraction as attr
from src import config
from src import detection as det
from src import embeddings as emb
from src import retrieval as ret
from src.utils import is_verified, load_catalog_image

st.set_page_config(
    page_title="Vision-Based Product Attribution & BOM Mapping",
    page_icon="⚙️",
    layout="wide",
)

# ---------------- Image sizes (px) — tweak these to taste ----------------
UPLOAD_W = 420     # uploaded image preview
ANNOT_W = 560      # annotated detection image
CROP_W = 140       # detected-part thumbnails
PART_W = 150       # selected-part header image
CARD_W = 150       # catalog match thumbnails

# ---------------- Styling ----------------

LIGHT = {
    "ink": "#14202a", "strong": "#000000", "muted": "#55636f",
    "panel": "#f3f6f8", "line": "#d5dde3",
    "accent": "#0b6e6e", "accent-fg": "#ffffff",
    "ok-bg": "#ddf3e6", "ok-fg": "#14572f",
    "warn-bg": "#fcebc4", "warn-fg": "#7a4f00",
    "code-bg": "#e7edf1",
}
DARK = {
    "ink": "#e8eef2", "strong": "#ffffff", "muted": "#9fb0bc",
    "panel": "#1b232a", "line": "#35434e",
    "accent": "#35bdb5", "accent-fg": "#04201f",
    "ok-bg": "#17382a", "ok-fg": "#8be8b0",
    "warn-bg": "#3d2f0e", "warn-fg": "#f6c964",
    "code-bg": "#26323b",
}


def _vars(pal):
    return ":root{" + "".join(f"--{k}:{v};" for k, v in pal.items()) + "}"


def detect_theme():
    """Return 'light' / 'dark' if Streamlit reports it, else None."""
    try:
        t = st.context.theme.type
        return t if t in ("light", "dark") else None
    except Exception:
        return None


_mode = detect_theme()
if _mode == "dark":
    THEME_VARS = _vars(DARK)
elif _mode == "light":
    THEME_VARS = _vars(LIGHT)
else:  # older Streamlit: follow the OS/browser preference
    THEME_VARS = _vars(LIGHT) + "@media (prefers-color-scheme: dark){" + _vars(DARK) + "}"

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
/*THEME_VARS*/

html, body, [class*="css"], .stMarkdown, .stButton button {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
}
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px; }

h1 { font-weight: 600; letter-spacing: -0.02em; color: var(--ink); }
h2 { font-size: 1.35rem !important; font-weight: 600; color: var(--ink);
     margin-top: .6rem; padding-bottom: .35rem; border-bottom: 1px solid var(--line); }
h3 { font-size: 1.05rem !important; font-weight: 600; color: var(--ink); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--muted) !important; }

/* Top-level tabs act as the navbar */
div[data-testid="stTabs"] button[role="tab"] {
    font-size: 1rem; font-weight: 500; padding: .6rem 1.1rem;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--accent) !important; font-weight: 600;
}

/* Bordered containers = soft panels */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 10px; border-color: var(--line) !important;
}

/* Primary button */
.stButton button[kind="primary"] {
    background: var(--accent); color: var(--accent-fg); border: none;
    border-radius: 8px; padding: .55rem 1.4rem; font-weight: 500;
}
.stButton button[kind="primary"] * { color: var(--accent-fg) !important; }
.stButton button[kind="primary"]:hover { filter: brightness(1.1); }

/* Images */
div[data-testid="stImage"] img { border-radius: 8px; border: 1px solid var(--line); }

/* Width / height boxes */
.dim-box { background: var(--panel); border: 1px solid var(--line);
           border-radius: 10px; padding: .6rem .9rem; }
.dim-label { color: var(--strong) !important; font-size: .85rem; font-weight: 500; }
.dim-value { color: var(--strong) !important; font-size: 1.6rem; font-weight: 600; line-height: 1.3; }

/* Match cards */
.m-rank { font-weight: 600; color: var(--ink); font-size: .95rem; margin: .4rem 0 .1rem; }
.m-sku  { font-family: ui-monospace, Menlo, monospace; font-size: .82rem; color: var(--ink);
          background: var(--code-bg); border-radius: 5px; padding: 1px 6px; }
.m-line { color: var(--muted); font-size: .8rem; line-height: 1.6; }
.m-line b { color: var(--ink); font-weight: 500; }
.badge  { display: inline-block; font-size: .74rem; font-weight: 600;
          border-radius: 999px; padding: 2px 10px; margin: .35rem 0 .2rem; }
.badge.ok   { background: var(--ok-bg);   color: var(--ok-fg); }
.badge.warn { background: var(--warn-bg); color: var(--warn-fg); }
.thr-pass { color: var(--ok-fg);   font-size: .8rem; font-weight: 600; }
.thr-fail { color: var(--warn-fg); font-size: .8rem; font-weight: 600; }
</style>
""".replace("/*THEME_VARS*/", THEME_VARS)

st.markdown(STYLE, unsafe_allow_html=True)

# ---------------- Cached resources ----------------

@st.cache_resource(show_spinner="Loading YOLO model…")
def get_yolo():
    if not config.YOLO_MODEL_PATH.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {config.YOLO_MODEL_PATH}")
    return det.load_yolo(str(config.YOLO_MODEL_PATH))


@st.cache_resource(show_spinner="Loading Amazon Nova embedding client…")
def get_embedding():
    """Initialize the Amazon Nova embedding client."""

    client, model_name, dim = emb.load_embedding(
        config.EMBEDDING_MODEL_NAME
    )

    if dim != config.EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: "
            f"got {dim}, expected {config.EMBEDDING_DIM}"
        )

    return client, model_name, dim


@st.cache_resource(show_spinner="Connecting to Nova ChromaDB…")
def get_collection():
    import chromadb

    if not config.CHROMA_DIR.is_dir():
        raise FileNotFoundError(
            f"ChromaDB directory not found: {config.CHROMA_DIR}"
        )

    client = chromadb.PersistentClient(
        path=str(config.CHROMA_DIR)
    )

    collection = client.get_collection(
        name=config.COLLECTION_NAME
    )

    if collection.count() == 0:
        raise ValueError(
            f"Collection '{config.COLLECTION_NAME}' is empty."
        )

    print(f"Database path: {config.CHROMA_DIR}")
    print(f"Collection: {collection.name}")
    print(f"Records: {collection.count()}")

    return collection


# ---------------- Helpers ----------------

def group_similar(dets, tol, min_conf):
    """Group detections of the same class with similar, high confidence.

    Returns a list of groups; the first item of each group is the representative
    (highest confidence). Groups are ordered by the representative's id.
    """
    groups = []
    for d in sorted(dets, key=lambda x: -x["confidence"]):
        for g in groups:
            r = g[0]
            if (r["class_name"] == d["class_name"]
                    and r["confidence"] >= min_conf and d["confidence"] >= min_conf
                    and r["confidence"] - d["confidence"] <= tol):
                g.append(d)
                break
        else:
            groups.append([d])
    groups.sort(key=lambda g: g[0]["id"])
    return groups


def esc(value) -> str:
    return html.escape(str(value))


def part_key(d) -> str:
    """Cache key for a part's attributes / embedding (depends on crop padding)."""
    return f"{d['id']}:{padding_pct}"


# ---- Attribute display helpers ----

CONF_ICON = {"high": "🟢 high", "medium": "🟡 medium", "low": "🔴 low"}
CORE_ATTRS = [
    ("part_type", "Part type"),
    ("geometry", "Geometry"),
    ("material", "Material"),
    ("dimensions", "Dimensions"),
    ("manufacturer", "Manufacturer"),
    ("manufacturer_part_number", "Manufacturer part no."),
]


def _fmt(value):
    """Format a value from the attribute JSON for display."""
    if value is None:
        return "— (undetermined)"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def render_attributes(attrs: dict):
    """Render the extracted attribute table and the list-type fields."""
    rows = []
    for key, label in CORE_ATTRS:
        item = attrs.get(key) or {}
        conf = str(item.get("confidence", "low")).lower()
        rows.append({
            "Attribute": label,
            "Value": _fmt(item.get("value")),
            "Confidence": CONF_ICON.get(conf, conf),
            "Evidence": item.get("evidence", ""),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    lists = [
        ("Visible markings", "visible_markings"),
        ("Physical features", "physical_features"),
        ("Additional observations", "additional_observations"),
        ("Unresolved attributes", "unresolved_attributes"),
    ]
    cols = st.columns(len(lists))
    for col, (title, key) in zip(cols, lists):
        with col:
            st.markdown(f"**{title}**")
            items = attrs.get(key) or []
            if items:
                for it in items:
                    st.markdown(f"- {_fmt(it)}")
            else:
                st.caption("None")


# ---- Parallel processing ----

def process_parts_parallel(parts, crops, workers):
    """Run Claude attribute extraction and Nova embedding for every part
    that has not been processed yet, all concurrently.

    Worker threads only call the Bedrock-backed functions (no Streamlit calls);
    results are written to st.session_state from the main thread.
    """
    attrs_cache = st.session_state["attributes"]
    embs_cache = st.session_state["embeddings"]

    jobs = []  # (kind, part, cache_key)
    for d in parts:
        key = part_key(d)
        if key not in attrs_cache:
            jobs.append(("attr", d, key))
        if key not in embs_cache:
            jobs.append(("emb", d, key))

    if not jobs:
        return

    def work(kind, d):
        crop = crops[d["id"] - 1]
        if kind == "attr":
            return attr.extract_attributes(crop)
        return emb.embed_image(
            client=bedrock_client,
            model_name=emb_model,
            dimension=emb_dim,
            image=crop,
        )

    progress = st.progress(0.0, text=f"Processing {len(parts)} part(s) in parallel…")
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, kind, d): (kind, d, key) for kind, d, key in jobs}
        for fut in as_completed(futures):
            kind, d, key = futures[fut]
            try:
                entry = {"data": fut.result(), "error": None}
            except Exception as e:
                entry = {"data": None, "error": str(e)}
            (attrs_cache if kind == "attr" else embs_cache)[key] = entry
            done += 1
            progress.progress(
                done / len(futures),
                text=f"{done}/{len(futures)} tasks complete "
                     "(attribute extraction + embeddings)",
            )
    progress.empty()


def retrieve_all(unique_parts):
    """Query ChromaDB for every unique part from its cached embedding.

    Local and fast, so it runs on every rerun and the retrieval sliders
    (top-K, class filter, threshold) stay live without new Bedrock calls.
    Returns {part_id: {"matches": list, "error": str | None}}.
    """
    out = {}
    for d in unique_parts:
        entry = st.session_state["embeddings"].get(part_key(d))
        if entry is None or entry["error"]:
            err = entry["error"] if entry else "not computed"
            out[d["id"]] = {"matches": [], "error": f"Embedding failed: {err}"}
            continue

        q_emb = entry["data"]
        if q_emb.shape[0] != config.EMBEDDING_DIM:
            out[d["id"]] = {
                "matches": [],
                "error": (f"Query embedding dim {q_emb.shape[0]} != catalog dim "
                          f"{config.EMBEDDING_DIM}."),
            }
            continue

        try:
            matches = ret.query_similar(collection, q_emb, top_k, class_filter or None)
            out[d["id"]] = {"matches": matches, "error": None}
        except Exception as e:
            out[d["id"]] = {"matches": [], "error": f"Catalog retrieval failed: {e}"}
    return out


# ---------------- Sidebar ----------------

st.sidebar.title("⚙️ Settings")

with st.sidebar.expander("Detection", expanded=True):
    conf_thr = st.slider("YOLO confidence threshold", 0.05, 0.95, 0.25, 0.05)
    iou_thr = st.slider("YOLO IoU threshold", 0.10, 0.95, 0.45, 0.05)
    padding_pct = st.slider("Crop padding (%)", 0, 30, 5, 1)

with st.sidebar.expander("Duplicate parts", expanded=True):
    merge_dups = st.checkbox(
        "Merge similar detections", value=True,
        help="Detections of the same class with very similar, high confidence "
             "are treated as one part (counted as quantity) and processed once.",
    )
    merge_tol = st.slider("Max confidence difference", 0.0, 0.20, 0.05, 0.01,
                          disabled=not merge_dups)
    merge_min_conf = st.slider("Min confidence to merge", 0.50, 0.99, 0.80, 0.01,
                               disabled=not merge_dups)

with st.sidebar.expander("Parallel processing", expanded=False):
    workers = st.slider(
        "Parallel workers", 1, 8, 4, 1,
        help="Number of simultaneous Bedrock calls (Claude attribute extraction "
             "and Nova embeddings). Lower this if you hit throttling errors.",
    )

with st.sidebar.expander("Retrieval", expanded=True):
    top_k = st.slider("Top K similar results", 1, 20, 5, 1)
    min_sim = st.slider(
        "Minimum similarity threshold", 0.0, 1.0, 0.70, 0.01,
        help="Similarity measures visual resemblance (cosine similarity), "
             "not the probability that the SKU is correct.",
    )
    class_filter = st.multiselect(
        "Component-class filter (empty = all)",
        options=["Bearing", "Bolt", "Gear", "Nut"],
        default=[],
    )

with st.sidebar.expander("ℹ️ About similarity scores", expanded=False):
    st.markdown(
        "Similarity scores measure **visual similarity** (cosine similarity between "
        "image embeddings). They are **not** probabilities that an SKU is correct. "
        "Always compare multiple candidates before assigning an SKU, especially for "
        "synthetic / unverified catalog records."
    )

# ---------------- Header ----------------

st.title("⚙️ Vision-Based Product Attribution & BOM Mapping")
st.caption(
    "Upload an image of mechanical parts. Parts are detected with YOLO, then "
    "Claude attribute extraction and catalog embeddings run automatically in "
    "parallel, followed by similar-product retrieval with SKUs."
)

# ---------------- System status ----------------

try:
    yolo_model = get_yolo()
    yolo_status = (
        f"✅ YOLO loaded ({', '.join(yolo_model.names.values())})"
    )
except Exception as e:
    yolo_model = None
    yolo_status = f"❌ YOLO failed: {e}"


try:
    bedrock_client, emb_model, emb_dim = get_embedding()

    emb_status = (
        f"✅ Embedding model loaded "
        f"({emb_model}, dim={emb_dim})"
    )
except Exception as e:
    bedrock_client = None
    emb_model = None
    emb_dim = 0
    emb_status = f"❌ Embedding model failed: {e}"


try:
    collection = get_collection()
    n_records = collection.count()

    chroma_status = (
        f"✅ ChromaDB connected "
        f"(`{config.COLLECTION_NAME}`, {n_records} records)"
    )
except Exception as e:
    collection = None
    n_records = 0
    chroma_status = f"❌ ChromaDB failed: {e}"


with st.expander("System status", expanded=False):
    st.write(yolo_status)
    st.write(emb_status)
    st.write(chroma_status)


if yolo_model is None or collection is None or bedrock_client is None:
    st.error(
        "A required backend component failed to load. "
        "See system status above."
    )
    st.stop()


def _reset_run():
    for k in ("results", "attributes", "embeddings"):
        st.session_state.pop(k, None)


# ======================================================================
# TAB 1 — Upload & Detection
# ======================================================================

def render_upload_and_detection():
    """Render upload, detection and the detected-part thumbnails, then run the
    parallel pipeline. Returns a context dict, or None if not ready yet
    (an early `return` replaces st.stop() so the other tabs still render)."""

    st.header("Upload image")
    uploaded = st.file_uploader(
        "Drag and drop an image containing mechanical parts",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded is None:
        st.info("Upload a JPG, JPEG, or PNG image to begin.")
        return None

    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"Could not read the uploaded file as an image: {e}")
        return None

    # Clear stale results when a different file is uploaded
    if st.session_state.get("run_id") != uploaded.name:
        _reset_run()

    up_left, up_right = st.columns([1, 1.2], gap="large")
    with up_left:
        st.image(image, caption="Uploaded image", width=UPLOAD_W)
    with up_right:
        m1, m2 = st.columns(2)
        m1.markdown(
            f'<div class="dim-box"><div class="dim-label">Width</div>'
            f'<div class="dim-value">{image.size[0]} px</div></div>',
            unsafe_allow_html=True,
        )
        m2.markdown(
            f'<div class="dim-box"><div class="dim-label">Height</div>'
            f'<div class="dim-value">{image.size[1]} px</div></div>',
            unsafe_allow_html=True,
        )
        st.caption(f"File: {uploaded.name}")
        identify = st.button("🔍 Identify Parts", type="primary")

    if identify:
        st.session_state["run_id"] = uploaded.name
        _reset_run()

    # ---- Detection ----
    st.header("Detection")

    if not identify and "results" not in st.session_state:
        st.info(
            "Press **Identify Parts** to start. YOLO detection runs first, then "
            "attribute extraction and embeddings run automatically in parallel."
        )
        return None

    if "results" not in st.session_state:
        with st.spinner("Running YOLO detection…"):
            try:
                detections = det.run_detection(yolo_model, image, conf_thr, iou_thr)
            except Exception as e:
                st.error(f"Detection failed: {e}")
                return None
            annotated = det.draw_boxes(image, detections) if detections else image
            st.session_state["results"] = {
                "detections": detections,
                "annotated": annotated,
                "params": {"conf": conf_thr, "iou": iou_thr, "pad": padding_pct},
            }
    else:
        prev = st.session_state["results"]["params"]
        if prev.get("conf") != conf_thr or prev.get("iou") != iou_thr:
            st.info("Thresholds changed — press **Identify Parts** again to re-run detection.")

    detections = st.session_state["results"]["detections"]
    annotated = st.session_state["results"]["annotated"]

    if not detections:
        st.warning(
            "No mechanical parts detected. Try lowering the confidence threshold, "
            "using a clearer/closer photo, or checking the lighting."
        )
        return None

    det_left, det_right = st.columns([1.1, 1], gap="large")
    with det_left:
        st.image(annotated, caption="Annotated image (bounding boxes + labels)", width=ANNOT_W)
    with det_right:
        st.success(f"Detected **{len(detections)}** part(s).")
        det_table = [
            {"Part": d["label"], "Class": d["class_name"], "Confidence": f"{d['confidence']:.2%}"}
            for d in detections
        ]
        st.dataframe(det_table, use_container_width=True, hide_index=True)

    # ---- Detected parts (thumbnails) ----
    st.header("Detected parts")
    pad_ratio = padding_pct / 100.0
    crops = [det.crop_detection(image, d["box"], padding_ratio=pad_ratio) for d in detections]

    if merge_dups:
        groups = group_similar(detections, merge_tol, merge_min_conf)
    else:
        groups = [[d] for d in detections]
    unique_parts = [g[0] for g in groups]
    group_of = {g[0]["id"]: g for g in groups}

    n_merged = len(detections) - len(unique_parts)
    if n_merged > 0:
        st.info(
            f"{len(detections)} detections → {len(unique_parts)} unique part(s). "
            f"{n_merged} near-identical detection(s) of the same class were merged "
            "and counted as quantity."
        )

    PER_ROW = 6
    for start in range(0, len(unique_parts), PER_ROW):
        row = st.columns(PER_ROW)
        for offset, d in enumerate(unique_parts[start:start + PER_ROW]):
            qty = len(group_of[d["id"]])
            qty_txt = f" ×{qty}" if qty > 1 else ""
            with row[offset]:
                st.image(
                    crops[d["id"] - 1],
                    caption=f"{d['label']}{qty_txt} · {d['class_name']} ({d['confidence']:.0%})",
                    width=CROP_W,
                )

    # ---- Automatic parallel processing (all unique parts, only what's missing) ----
    st.session_state.setdefault("attributes", {})
    st.session_state.setdefault("embeddings", {})
    process_parts_parallel(unique_parts, crops, workers)

    st.success(
        "Processing complete — open the **🧩 Parts** tab to inspect each part, "
        "or **📋 BOM summary** for the overview."
    )

    return {"crops": crops, "unique_parts": unique_parts, "group_of": group_of}


# ======================================================================
# TAB 2 — Parts (one part at a time, attributes / matches in inner tabs)
# ======================================================================

def render_attributes_panel(d, crop):
    key = part_key(d)

    if st.button("↻ Re-extract attributes", key=f"reextract_{d['id']}"):
        with st.spinner(f"Re-extracting attributes for {d['label']}…"):
            try:
                st.session_state["attributes"][key] = {
                    "data": attr.extract_attributes(crop),
                    "error": None,
                }
            except Exception as e:
                st.session_state["attributes"][key] = {"data": None, "error": str(e)}

    st.caption(
        "Claude reports only what is visibly supported. Confidence is qualitative, "
        "not a calibrated probability. Manufacturer, part number, material grade and "
        "exact dimensions stay empty unless clearly legible or evidenced."
    )

    entry = st.session_state["attributes"].get(key)
    if entry is None:
        st.caption("No attributes available.")
    elif entry["error"]:
        st.error(f"Attribute extraction failed for {d['label']}: {entry['error']}")
    else:
        render_attributes(entry["data"])
        with st.expander("Raw JSON"):
            st.json(entry["data"])


def render_matches_panel(d, res):
    if res["error"]:
        st.error(f"{d['label']}: {res['error']}")
        return

    matches = res["matches"]
    if not matches:
        st.warning(f"No catalog candidates returned for {d['label']}.")
        return

    confident = [m for m in matches if m["similarity"] >= min_sim]
    if not confident:
        st.warning(
            f"No confident match found for {d['label']} "
            f"(best similarity {matches[0]['similarity']:.1%} < {min_sim:.0%} threshold). "
            "No SKU is assigned automatically — compare candidates below or adjust the threshold."
        )
    else:
        st.caption(
            f"{len(confident)} of {len(matches)} candidate(s) meet the {min_sim:.0%} threshold. "
            "Scores reflect visual similarity, not SKU correctness."
        )

    CARDS_PER_ROW = 5
    for start in range(0, len(matches), CARDS_PER_ROW):
        row = st.columns(CARDS_PER_ROW)
        for offset, m in enumerate(matches[start:start + CARDS_PER_ROW]):
            with row[offset]:
                with st.container(border=True):
                    img = load_catalog_image(m["crop_path"])
                    if img is not None:
                        st.image(img, width=CARD_W)
                    else:
                        st.caption("Catalog image unavailable")

                    meets = m["similarity"] >= min_sim
                    thr_cls = "thr-pass" if meets else "thr-fail"
                    thr_txt = "meets threshold" if meets else "below threshold"
                    verified = is_verified(m)
                    badge_cls = "ok" if verified else "warn"
                    badge_txt = "Verified" if verified else "Synthetic / unverified"

                    st.markdown(
                        f"""
                        <div class="m-rank">#{esc(m['rank'])} · {m['similarity_pct']:.1f}%</div>
                        <div class="{thr_cls}">{thr_txt}</div>
                        <div class="m-line" style="margin-top:.35rem">
                            <b>SKU</b> <span class="m-sku">{esc(m['sku'])}</span><br>
                            <b>Class</b> {esc(m['class_name'])}<br>
                            <b>Maker</b> {esc(m['manufacturer'])}<br>
                            <b>MPN</b> <span class="m-sku">{esc(m['manufacturer_part_number'])}</span>
                        </div>
                        <span class="badge {badge_cls}">{badge_txt}</span>
                        <div class="m-line">Origin: {esc(m['data_origin'])}<br>Crop: {esc(m['crop_id'])}</div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if not verified:
                        st.caption("Do not treat as a real product identity.")


def render_parts_tab(ctx, retrieval):
    crops = ctx["crops"]
    unique_parts = ctx["unique_parts"]
    group_of = ctx["group_of"]

    def _label(part_id):
        d = next(p for p in unique_parts if p["id"] == part_id)
        qty = len(group_of[part_id])
        qty_txt = f" ×{qty}" if qty > 1 else ""
        return f"{d['label']} · {d['class_name']}{qty_txt}"

    # Part selector = secondary navbar
    chosen_id = st.radio(
        "Part",
        options=[p["id"] for p in unique_parts],
        format_func=_label,
        horizontal=True,
        label_visibility="collapsed",
    )
    d = next(p for p in unique_parts if p["id"] == chosen_id)
    group = group_of[d["id"]]
    res = retrieval[d["id"]]

    # Part header: thumbnail + quick facts
    with st.container(border=True):
        c_img, c_txt = st.columns([1, 6], vertical_alignment="center")
        with c_img:
            st.image(crops[d["id"] - 1], width=PART_W)
        with c_txt:
            st.subheader(f"{d['label']} — predicted {d['class_name']}")
            cap = f"Detection confidence: {d['confidence']:.0%}"
            if len(group) > 1:
                others = ", ".join(x["label"] for x in group[1:])
                cap += f" · Quantity: {len(group)} (also detected as {others})"
            st.caption(cap)

            a_entry = st.session_state["attributes"].get(part_key(d))
            if a_entry and a_entry["data"]:
                pt = a_entry["data"]["part_type"].get("value")
                if pt:
                    st.markdown(f"**Claude sees:** {_fmt(pt)}")
            if res["matches"]:
                best = res["matches"][0]
                st.markdown(
                    f"**Top candidate:** `{best['sku']}` · {best['similarity']:.1%} visual similarity"
                )

    tab_attr, tab_match = st.tabs(["🔬 Attributes", "🔎 Catalog matches"])
    with tab_attr:
        render_attributes_panel(d, crops[d["id"] - 1])
    with tab_match:
        render_matches_panel(d, res)


# ======================================================================
# TAB 3 — BOM summary
# ======================================================================

def build_bom_rows(ctx, retrieval):
    rows = []
    for d in ctx["unique_parts"]:
        res = retrieval[d["id"]]
        row = {
            "Part": d["label"],
            "Detected class": d["class_name"],
            "Qty": len(ctx["group_of"][d["id"]]),
        }
        if res["matches"]:
            best = res["matches"][0]
            row.update({
                "Top candidate SKU": best["sku"],
                "Top candidate class": best["class_name"],
                "Similarity": f"{best['similarity']:.1%}",
                "Status": best["verification_status"],
            })
        else:
            row.update({
                "Top candidate SKU": "—",
                "Top candidate class": "—",
                "Similarity": "—",
                "Status": res["error"] or "No candidates",
            })

        entry = st.session_state["attributes"].get(part_key(d))
        a = entry["data"] if entry and entry["data"] else None
        if a:
            row["Extracted part type"] = _fmt(a["part_type"].get("value"))
            row["Material (visual)"] = _fmt(a["material"].get("value"))
            row["Markings"] = (
                ", ".join(_fmt(m) for m in a.get("visible_markings", [])) or "—"
            )
        rows.append(row)
    return rows


def rows_to_csv(rows):
    fieldnames = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def render_bom_tab(ctx, retrieval):
    st.header("BOM summary")
    st.info(
        "Similarity scores measure **visual similarity only** and are not probabilities "
        "that an SKU is correct. No definitive SKU is assigned automatically — review the "
        "candidates in the **Parts** tab before selecting one. Synthetic / unverified "
        "records must not be presented as real-world product identities."
    )
    rows = build_bom_rows(ctx, retrieval)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download BOM as CSV",
            data=rows_to_csv(rows),
            file_name="bom_summary.csv",
            mime="text/csv",
        )
    else:
        st.info("No candidates to summarize yet.")


# ======================================================================
# Main layout — top-level tabs act as the navbar
# ======================================================================

tab_overview, tab_parts, tab_bom = st.tabs(
    ["📤 Upload & Detection", "🧩 Parts", "📋 BOM summary"]
)

with tab_overview:
    ctx = render_upload_and_detection()

if ctx is not None:
    retrieval = retrieve_all(ctx["unique_parts"])
    with tab_parts:
        render_parts_tab(ctx, retrieval)
    with tab_bom:
        render_bom_tab(ctx, retrieval)
else:
    with tab_parts:
        st.info("Upload an image and run **Identify Parts** first.")
    with tab_bom:
        st.info("Upload an image and run **Identify Parts** first.")