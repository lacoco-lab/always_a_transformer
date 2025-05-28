import datetime
import jsonlines
import re
import tarfile
from pathlib import Path
import arxiv  # pip install arxiv

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DAYS_BACK = 30
MAX_RESULTS = 500
MIN_WORDS = 100
TARGET_CHUNKS = 1500
SRC_DIR = Path("tmp_arxiv_src")
OUT_FILE = Path("datasets/realistic/arxiv/copy_arxiv.jsonl")
section_re = re.compile(r'(\\section\*?\{.*?\})', flags=re.DOTALL)


def ensure_dirs():
    """Ensure source and output directories exist."""
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def download_sources(days_back: int = DAYS_BACK, max_results: int = MAX_RESULTS) -> list[Path]:
    """
    Download arXiv source tarballs for papers submitted within the given timeframe.
    Returns list of local tar.gz file paths.
    """
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=days_back)
    query = (
        f"submittedDate:[{start_date:%Y%m%d}0000 TO {today:%Y%m%d}2359]"
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    client = arxiv.Client()
    tar_files = []

    for paper in client.results(search):
        pid = paper.get_short_id()
        tar_path = SRC_DIR / f"{pid}.tar.gz"
        if tar_path.exists():
            tar_files.append(tar_path)
            continue
        try:
            print(f"Downloading source for {pid}")
            paper.download_source(dirpath=str(SRC_DIR), filename=tar_path.name)
            tar_files.append(tar_path)
        except Exception as e:
            print(f"Failed to download {pid}: {e}")
    return tar_files


def parse_source_dir(src_dir: Path = SRC_DIR) -> list[Path]:
    """
    Return list of existing .tar.gz files in source directory.
    """
    return list(src_dir.glob("*.tar.gz"))


def chunk_text(text: str, max_words: int = 250) -> list[str]:
    """
    Split text into sentence-ending chunks of up to max_words.
    Ensures each chunk ends at the last full sentence.
    """
    words = text.split()
    chunks: list[str] = []
    # Build raw chunks by word count
    for i in range(0, len(words), max_words):
        raw = " ".join(words[i:i+max_words])
        # Truncate to end of last sentence punctuation
        # Find last occurrence of ., !, or ?
        m = re.search(r".*[\.\!\?]", raw)
        if m:
            truncated = m.group(0)
        else:
            truncated = raw  # no sentence end found
        chunks.append(truncated.strip())
    return chunks



def extract_chunks_from_tar(tar_path: Path, min_words: int = MIN_WORDS, \
                              target_chunks: int = TARGET_CHUNKS) -> list[str]:
    """
    Extract .tex files from tar, split into sections and chunks.
    Returns list of text chunks.
    """
    chunks = []
    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            members = [m for m in tf.getmembers() if m.name.endswith(".tex")]
            if not members:
                return []
            main = next((m for m in members if Path(m.name).stem == tar_path.stem), members[0])
            content = tf.extractfile(main).read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"Bad archive {tar_path.name}: {e}")
        return []

    parts = section_re.split(content)
    if len(parts) <= 1:
        return []

    for i in range(1, len(parts), 2):
        header = parts[i]
        body = parts[i+1] if i+1 < len(parts) else ""
        full = header + "\n" + body
        for chunk in chunk_text(full, max_words=250):
            if len(chunk.split()) >= min_words:
                chunks.append(chunk)
                if len(chunks) >= target_chunks:
                    return chunks
    return chunks


def dump_chunks_to_jsonl(chunks: list[str], out_file: Path = OUT_FILE, \
                         target_chunks: int = TARGET_CHUNKS) -> None:
    """
    Write up to target_chunks text chunks to JSONL.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open('w', encoding='utf-8') as f:
        writer = jsonlines.Writer(f)
        for chunk in chunks[:target_chunks]:
            writer.write({"input": chunk})
    print(f"✅ saved {len(chunks[:target_chunks])} chunks to {out_file}")


def main(use_local: bool = True):
    """
    Main entry point: either parse local files or download then parse.
    """
    ensure_dirs()
    if use_local:
        tar_files = parse_source_dir(SRC_DIR)
    else:
        tar_files = download_sources()

    all_chunks = []
    for tar in tar_files:
        chunks = extract_chunks_from_tar(tar)
        all_chunks.extend(chunks)
        if len(all_chunks) >= TARGET_CHUNKS:
            break

    dump_chunks_to_jsonl(all_chunks)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="ArXiv Data Downloader and Parser")
    p.add_argument('--no-download', action='store_true', help='Use only local tar.gz files')
    args = p.parse_args()
    main(use_local=args.no_download)
