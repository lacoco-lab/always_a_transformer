import datetime
import io
import jsonlines
import re
import tarfile
from pathlib import Path

import arxiv  # pip install arxiv

# ── CONFIG ─────────────────────────────────────────────────────────────────────

# How many days back to pull:
DAYS_BACK     = 30
# Max papers to consider
MAX_RESULTS   = 1000
# Minimum words per chunk to keep
MIN_WORDS     = 500
# Stop once we've gathered this many chunks
TARGET_CHUNKS = 1500
# Where to drop downloaded tarballs
SRC_DIR       = Path("tmp_arxiv_src")
# Where to write out final JSONL
OUT_FILE      = Path("datasets/copy_controlled/copy_arxiv.jsonl")

# ── PREPARE ────────────────────────────────────────────────────────────────────

# Make sure directories exist
SRC_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Regex for splitting on each \section{…}
section_re = re.compile(r'(\\section\*?\{.*?\})', flags=re.DOTALL)

def chunk_text(text: str, max_words: int=1500):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i+max_words])

# ── FETCH METADATA ─────────────────────────────────────────────────────────────

today      = datetime.date.today()
start_date = today - datetime.timedelta(days=DAYS_BACK)
query      = (
    f"submittedDate:[{start_date:%Y%m%d}0000 TO {today:%Y%m%d}2359]"
)
search     = arxiv.Search(
    query     = query,
    max_results = MAX_RESULTS,
    sort_by     = arxiv.SortCriterion.SubmittedDate
)
client = arxiv.Client()  # respects a 3 s delay between pages :contentReference[oaicite:1]{index=1}

# ── DOWNLOAD & SPLIT ───────────────────────────────────────────────────────────

chunks = []
for paper in client.results(search):
    pid = paper.get_short_id()
    print(f"[{len(chunks):4d}] downloading source for {pid}")

    # 1) fetch the .tar.gz of the source into a local file
    tar_path = SRC_DIR / f"{pid}.tar.gz"
    try:
        paper.download_source(
            dirpath = str(SRC_DIR),
            filename = f"{pid}.tar.gz"
        )  # writes to <SRC_DIR>/<pid>.tar.gz :contentReference[oaicite:2]{index=2}
    except Exception as e:
        print(f"   ✗ failed to download {pid}: {e}")
        continue

    # 2) open it and grab the first .tex (or the one matching pid.tex)
    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            members = [m for m in tf.getmembers() if m.name.endswith(".tex")]
            if not members:
                raise RuntimeError("no .tex files")
            main = next((m for m in members if Path(m.name).stem == pid),
                        members[0])
            content = tf.extractfile(main).read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"   ✗ bad archive for {pid}: {e}")
        continue

    # 3) regex‑split into [pre, sec1, body1, sec2, body2, …]
    parts = section_re.split(content)
    if len(parts) <= 1:
        # no sections → skip
        continue

    # 4) assemble each section + body, then re‑chunk by 2000 words
    for i in range(1, len(parts), 2):
        header = parts[i]
        body   = parts[i+1] if i+1 < len(parts) else ""
        full   = header + "\n" + body
        for chunk in chunk_text(full, max_words=2000):
            if len(chunk.split()) >= MIN_WORDS:
                chunks.append(chunk)
        if len(chunks) >= TARGET_CHUNKS:
            break
    if len(chunks) >= TARGET_CHUNKS:
        break

# ── DUMP TO JSONL ─────────────────────────────────────────────────────────────

with OUT_FILE.open("w", encoding="utf-8") as f:
    writer = jsonlines.Writer(f)
    for c in chunks[:TARGET_CHUNKS]:
        writer.write({"input": c})

print(f"✅ saved {len(chunks[:TARGET_CHUNKS])} chunks to {OUT_FILE}")
