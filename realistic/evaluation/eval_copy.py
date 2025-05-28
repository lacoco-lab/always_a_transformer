import os
import json
import tqdm
import jsonlines
from typing import List
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from transformers import AutoTokenizer
from realistic.evaluation.eval_utils import compare_sequences_context_aware


HF_TOKEN = os.environ.get("HF_TOKEN", None)

def get_tokenizer_for_model(model_name: str):
    """
    Get the appropriate tokenizer for a given model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        A tokenizer instance
    """
    # Map model names to their corresponding HuggingFace model identifiers
    model_name = model_name.lower()
    model_to_hf_mapping = {
        "qwen2.5_7b": "Qwen/Qwen2.5-7B",
        "qwen2.5_32b": "Qwen/Qwen2.5-32B",
        "qwen2.5_7b_instruct": "Qwen/Qwen2.5-32B-Instruct",
        "qwen2.5_32b_instruct": "Qwen/Qwen2.5-32B-Instruct",
        "llama3_8b": "meta-llama/Llama-3.1-8B",
        "llama3_8b_instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama3_70b": "meta-llama/Meta-Llama-3.1-70B",
        "llama3_70b_instruct": "meta-llama/Llama-3.3-70B-Instruct",
    }
    
    # Get the huggingface model identifier
    hf_model_name = model_to_hf_mapping.get(model_name)
    
    if not hf_model_name:
        raise ValueError(f"Unknown model name: {model_name}. Supported models are: {list(model_to_hf_mapping.keys())}")
    
    # Load the tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, token=HF_TOKEN, trust_remote_code=True)
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer for {model_name} ({hf_model_name}): {e}")
        return None

def tokenize_text(text: str, tokenizer=None) -> List[str]:
    """
    Tokenize text using the specified tokenizer or fallback to simple tokenization.
    
    Args:
        text: The text to tokenize
        tokenizer: Optional tokenizer instance
        
    Returns:
        List of tokens
    """
    try:
        # Use the model's tokenizer
        return tokenizer.tokenize(text.strip())
    except Exception as e:
        print(f"Error using model tokenizer: {e}. Falling back to simple tokenization.")
    return []
    
def is_text_similar_enough(text1: str, text2: str) -> bool:
    """
    Check if the length ratio of two texts is at least 0.5
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        True if texts are similar enough in length, False otherwise
    """
    if not text1 or not text2:
        return False
        
    len1, len2 = len(text1), len(text2)
    ratio = min(len1, len2) / max(len1, len2)
    # The lengths should be within 75 % of each other
    return ratio > 0.75

def process_jsonl_file(file_path: Path, output_file: Path = None, verbose: bool = True, tokenizer = None):
    """
    Process a JSONL file and compare gold answers with model answers using context-aware comparison.
    
    Args:
        file_path: Path to the JSONL file
        output_file: Optional path to write results to
        verbose: Whether to print results to console
        tokenizer: Tokenizer to use for processing
    """
    results = []
    overall_stats = {
        "total_examples": 0,
        "exact_matches": 0,
        "length_mismatches": 0,
        "compared_examples": 0,
        "overall_similarity_sum": 0.0,
        "deterministic_similarity_sum": 0.0,
        "non_deterministic_similarity_sum": 0.0
    }
    
    # Open and process the JSONL file
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in tqdm.tqdm(enumerate(f, 1)):
            try:
                # Parse the JSON line
                entry = json.loads(line.strip())
                
                # Extract gold answer and model answer
                gold_ans = entry.get('gold_ans', '')
                model_ans = entry.get('answer', '')
                
                if gold_ans == "" or model_ans == "":
                    gold_ans = entry.get('input_text', '')
                    model_ans = entry.get('full_answer', '')
                    model_ans = model_ans.replace('<end>', '')
                    model_ans = model_ans.strip()
                # Pre-comparison checks
                overall_stats["total_examples"] += 1
                
                # Case 1: Exact match after stripping [Answer is completely contained in the model answer]
                if gold_ans.strip() in model_ans.strip():
                    metrics = {
                        "scores": {
                            "overall": 1.0,
                            "deterministic": 1.0,
                            "non_deterministic": 1.0
                        },
                        "exact_match": True,
                        "length_mismatch": False
                    }
                    overall_stats["exact_matches"] += 1
                
                # Case 2: Length mismatch
                elif not is_text_similar_enough(gold_ans, model_ans):
                    metrics = {
                        "scores": {
                            "overall": 0.0,
                            "deterministic": 0.0,
                            "non_deterministic": 0.0
                        },
                        "exact_match": False,
                        "length_mismatch": True
                    }
                    overall_stats["length_mismatches"] += 1
                
                # Case 3: Normal comparison
                else:
                    # Extract or generate tokenized versions
                    gold_tokens = tokenize_text(gold_ans, tokenizer)
                    model_tokens = tokenize_text(model_ans, tokenizer)
                    
                    # Compare the sequences with context-aware comparison
                    metrics = compare_sequences_context_aware(gold_tokens, model_tokens)
                    metrics["exact_match"] = False
                    metrics["length_mismatch"] = False
                    
                    # Update stats
                    overall_stats["compared_examples"] += 1
                    overall_stats["overall_similarity_sum"] += metrics["scores"]["overall"]
                    overall_stats["deterministic_similarity_sum"] += metrics["scores"]["deterministic"]
                    overall_stats["non_deterministic_similarity_sum"] += metrics["scores"]["non_deterministic"]
                
                # Store result
                result = {
                    "line_num": line_num,
                    "metrics": metrics
                }
                results.append(result)
                
                # Print if verbose
                if verbose:
                    print(f"\n=== Entry {line_num} ===")
                    print(f"Gold Answer: {gold_ans[:100]}..." if len(gold_ans) > 100 else gold_ans)
                    print(f"Model Answer: {model_ans[:100]}..." if len(model_ans) > 100 else model_ans)
                    
                    if metrics["exact_match"]:
                        print("Result: EXACT MATCH")
                    elif metrics["length_mismatch"]:
                        print("Result: LENGTH MISMATCH")
                    else:
                        print(f"Overall similarity: {metrics['scores']['overall']:.4f}")
                        print(f"Deterministic similarity: {metrics['scores']['deterministic']:.4f}")
                        print(f"Non-deterministic similarity: {metrics['scores']['non_deterministic']:.4f}")
                    
                    print("-" * 80)
                
            except json.JSONDecodeError:
                print(f"Error parsing JSON on line {line_num}. Skipping.")
            except Exception as e:
                print(f"Error processing line {line_num}: {str(e)}. Skipping.")
    
    # Calculate averages
    compared_count = overall_stats["compared_examples"]
    if compared_count > 0:
        avg_stats = {
            "avg_overall_similarity": overall_stats["overall_similarity_sum"] / compared_count,
            "avg_deterministic_similarity": overall_stats["deterministic_similarity_sum"] / compared_count,
            "avg_non_deterministic_similarity": overall_stats["non_deterministic_similarity_sum"] / compared_count,
        }
    else:
        avg_stats = {
            "avg_overall_similarity": 0.0,
            "avg_deterministic_similarity": 0.0,
            "avg_non_deterministic_similarity": 0.0,
        }
    
    # Print summary
    if verbose:
        print("\n=== SUMMARY ===")
        print(f"Total examples processed: {overall_stats['total_examples']}")
        print(f"Exact matches: {overall_stats['exact_matches']}")
        print(f"Length mismatches: {overall_stats['length_mismatches']}")
        print(f"Compared examples: {overall_stats['compared_examples']}")
        
        if compared_count > 0:
            print(f"Average overall similarity: {avg_stats['avg_overall_similarity']:.4f}")
            print(f"Average deterministic similarity: {avg_stats['avg_deterministic_similarity']:.4f}")
            print(f"Average non-deterministic similarity: {avg_stats['avg_non_deterministic_similarity']:.4f}")
    
    # Write to output file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "results": results,
                "summary": {
                    "overall_stats": overall_stats,
                    "avg_stats": avg_stats
                }
            }, f, indent=2)
            print(f"Results written to {output_file}")
    
    return results, avg_stats

def process_sample_entry():
    """
    Process the sample entry provided in the paste-2.txt file
    """
    try:
        gold_ans = "Quisquam dolorem tempora modi numquam porro. Est dolore adipisci numquam. Ut amet numquam non. Velit modi modi labore quisquam quiquia. Tempora magnam voluptatem quaerat ipsum amet dolore. Etincidunt dolore consectetur quiquia. Voluptatem labore quaerat adipisci. Est velit neque eius quaerat labore aliquam ipsum. Porro velit tempora consectetur. Eius ut dolor labore etincidunt dolor modi.  Dolore sed adipisci ipsum etincidunt est sit neque. Numquam consectetur modi adipisci. Eius neque aliquam porro. Amet adipisci magnam sit quisquam voluptatem. Ipsum sed est eius dolorem ut. Neque aliquam modi quiquia numquam sed. Consectetur labore ut sed numquam quaerat modi eius. Non voluptatem labore adipisci quiquia sed adipisci labore.  Neque non amet aliquam magnam. Ut ipsum etincidunt modi sed quiquia dolore quisquam. Dolor voluptatem est porro consectetur velit. Dolore tempora est eius eius amet etincidunt ipsum. Consectetur quisquam ut tempora. Modi aliquam neque etincidunt consectetur velit sed etincidunt. Quisquam quaerat sed modi quisquam amet. Quiquia adipisci tempora amet aliquam quisquam. Amet etincidunt ut quiquia quiquia dolor labore.  Magnam velit non non. Adipisci modi porro quiquia adipisci quaerat velit. Dolorem consectetur quisquam sed dolore. Magnam ipsum dolore porro non velit adipisci voluptatem. Est dolorem labore tempora. Non tempora tempora dolore numquam. Consectetur porro labore est labore. Est dolore voluptatem amet magnam dolore dolore.  Dolorem sed adipisci numquam consectetur. Eius dolore adipisci magnam ipsum. Adipisci ut dolorem velit neque. Ut labore numquam ut quisquam adipisci consectetur aliquam. Dolor tempora porro magnam velit aliquam. Adipisci ipsum velit non quaerat.  Labore est etincidunt dolor. Eius labore adipisci neque. Numquam quiquia etincidunt numquam non. Amet voluptatem adipisci aliquam non sit. Etincidunt sit magn"
        
        model_ans = "Quisquam dolorem tempora modi numquam porro. Est dolore adipisci numquam. Ut amet numquam non. Velit modi modi labore quisquam quiquia. Tempora magnam voluptatem quaerat ipsum amet dolore. Etincidunt dolore consectetur quiquia. Voluptatem labore quaerat adipisci. Est velit neque eius quaerat labore aliquam ipsum. Porro velit tempora consectetur. Eius ut dolor labore etincidunt dolor modi.  Dolore sed adipisci ipsum etincidunt est sit neque. Numquam consectetur modi adipisci. Eius neque aliquam porro. Amet adipisci magnam sit quisquam voluptatem. Ipsum sed est eius dolorem ut. Neque aliquam modi quiquia numquam sed. Consectetur labore ut sed numquam quaerat modi eius. Non voluptatem labore adipisci quiquia sed adipisci labore.  Neque non amet aliquam magnam. Ut ipsum etincidunt modi sed quiquia dolore quisquam. Dolor voluptatem est porro consectetur velit. Dolore tempora est eius eius amet etincidunt ipsum. Consectetur quisquam ut tempora. Modi aliquam neque etincidunt consectetur velit sed etincidunt. Quisquam quaerat sed modi quisquam amet. Quiquia adipisci tempora amet aliquam quisquam. Amet etincidunt ut quiquia quiquia dolor labore.  Magnam velit non non. Adipisci modi porro quiquia adipisci quaerat velit. Dolorem consectetur quisquam sed dolore. Magnam ipsum dolore porro non velit adipisci voluptatem. Est dolorem labore tempora. Non tempora tempora dolore numquam. Consectetur porro labore est labore. Est dolore voluptatem amet magnam dolore dolore.  Dolorem sed adipisci numquam consectetur. Eius dolore adipisci magnam ipsum. Adipisci ut dolorem velit neque. Ut labore numquam ut quisquam adipisci consectetur aliquam. Dolor tempora porro magnam velit aliquam. Adipisci ipsum velit non quaerat.  Labore est etincidunt dolor. Eius labore adipisci neque. Numquam quiquia etincidunt numquam non. Amet voluptatem adipisci aliquam non sit. Etincidunt sit magn <end>"
                
        model_ans = model_ans.replace('<end>', '')
        model_ans = model_ans.strip()
        # Pre-comparison checks
        if gold_ans.strip() == model_ans.strip():
            print("\n=== Sample Entry Analysis ===")
            print("Result: EXACT MATCH")
            return
            
        if not is_text_similar_enough(gold_ans, model_ans):
            print("\n=== Sample Entry Analysis ===")
            print("Result: LENGTH MISMATCH")
            return
        
        gold_tokens = tokenize_text(gold_ans, tokenizer=get_tokenizer_for_model("llama3.1_8B"))
        model_tokens = tokenize_text(model_ans, tokenizer=get_tokenizer_for_model("llama3.1_8B"))
        print(gold_tokens)
        print(model_tokens)
        print("\n=== Sample Entry Analysis ===")
        print(f"Gold Answer Length: {len(gold_ans)} characters, {len(gold_tokens)} tokens")
        print(f"Model Answer Length: {len(model_ans)} characters, {len(model_tokens)} tokens")
        
        # Compare using context-aware comparison
        metrics = compare_sequences_context_aware(gold_tokens, model_tokens)
        print(f"Overall similarity: {metrics['scores']['overall']:.4f}")
        print(f"Deterministic similarity: {metrics['scores']['deterministic']:.4f}")
        print(f"Non-deterministic similarity: {metrics['scores']['non_deterministic']:.4f}")
        
    except Exception as e:
        print(f"Error processing sample entry: {str(e)}")


def process_multiple_models(base_dir: str, output_dir: str = None, verbose: bool = True):
    """
    Process JSONL files from multiple model folders and compare results.
    
    Args:
        base_dir: Directory containing model folders and their JSONL files
        output_dir: Directory to write output files to
        verbose: Whether to print results to console
    """
    base_dir, output_dir, all_results = Path(base_dir), Path(output_dir), {}
    
    for model_result_dir in base_dir.iterdir():
        model_name = model_result_dir.name
        print(model_name)
        try:
            tokenizer = get_tokenizer_for_model(model_name)
        except Exception as we:
            continue
                
        model_results = {}
        curr_output_dir = output_dir / model_name 
        curr_output_dir.mkdir(parents=True, exist_ok=True)

        for input_file in model_result_dir.rglob('*.jsonl'):

            if '1500' in input_file.stem:
                continue
                
            # Get the parts of the path we need to preserve
            # First determine what parts of the paths we have in common
            input_parts = list(input_file.parts)
            model_dir_parts = list(model_result_dir.parts)

            # Extract the common subpath by finding all common directories
            common_index = 0
            for i, (part1, part2) in enumerate(zip(input_parts, model_dir_parts)):
                if part1 == part2:
                    common_index = i + 1
                else:
                    break
            # Get the part of the path after the common directories
            subpath = input_parts[common_index:] 
            # Combine with output directory
            output_file = curr_output_dir.joinpath(*subpath)
            # Create the parent directories if they don't exist
            output_file.parent.mkdir(parents=True, exist_ok=True)
            print(f"\nProcessing {input_file}...{output_file}")
            
            _, avg_stats = process_jsonl_file(input_file, output_file, verbose, tokenizer=tokenizer)
            model_results[output_file.name] = avg_stats
        
        all_results[model_name] = model_results

    # Write overall comparison if output directory is specified
    if output_dir:
        comparison_file = output_dir / "model_comparison.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nModel comparison written to {comparison_file}")
    
    # Print overall comparison
    if verbose:
        print("\n=== MODEL COMPARISON ===")
        # Create a table of results
        print(f"{'Model':<25} | {'Dataset':<15} | {'Overall Sim':<12} | {'Det. Sim':<12} | {'Non-Det. Sim':<12}")
        print('-' * 90)
        
        for model, datasets in all_results.items():
            for dataset, stats in datasets.items():
                print(f"{model:<25} | {dataset:<15} | {stats['avg_overall_similarity']:<12.4f} | {stats['avg_deterministic_similarity']:<12.4f} | {stats['avg_non_deterministic_similarity']:<12.4f}")
    
    return all_results


######################################################################
# 1. Exact‑match statistics                                           #
######################################################################

def check_for_exact(base_dir: Path = Path("results/realistic/arxiv")) -> tuple[
    Dict[str, Dict[str, Tuple[int, int]]],  # per_model_exact
    Dict[str, Dict[int, Set[str]]],         # per_model_correct_idxs
    Set[str],                               # *all* models we found
]:
    """Compute per‑model exact‑match statistics.

    Parameters
    ----------
    base_dir : pathlib.Path, optional
        Root directory whose *immediate* children are per‑model result
        folders.  Defaults to ``results/realistic/arxiv``.

    Returns
    -------
    per_model_exact
        ``{model -> {dataset -> (correct, total)}}``
    per_model_correct_idxs
        ``{dataset -> {example_index -> {models_that_copied_it}}}``
    all_models
        A set with the *folder names* of every model discovered.
    """

    per_model_exact: Dict[str, Dict[str, Tuple[int, int]]] = {}
    per_model_correct_idxs: Dict[str, Dict[int, Set[str]]] = defaultdict(lambda: defaultdict(set))
    all_models: Set[str] = set()

    for model_result_dir in base_dir.iterdir():
        if not model_result_dir.is_dir():
            continue
        model_name = model_result_dir.name
        all_models.add(model_name)

        for input_file in model_result_dir.rglob("*.jsonl"):
            if "arxiv_echo" not in input_file.stem:
                continue

            with jsonlines.open(input_file) as reader:
                all_lines = list(reader)

            total = len(all_lines)
            correct = 0
            for idx, row in enumerate(all_lines):
                if row["input_text"] in row["full_answer"]:
                    correct += 1
                    per_model_correct_idxs[input_file.stem][idx].add(model_name)

            per_model_exact.setdefault(model_name, {})[input_file.stem] = (correct, total)
            pct = correct / total if total else 0.0
            print(
                f"Model: {model_name:25s}  Dataset: {input_file.stem:20s}  "
                f"Exact Match: {correct}/{total} ({pct:.2%})"
            )

    return per_model_exact, per_model_correct_idxs, all_models

######################################################################
# 2. Which paragraphs are copied by *every* model?                    #
######################################################################

def paragraphs_copied_by_all_models(
    base_dir: Path = Path("results/realistic/arxiv"),
) -> Dict[str, List[int]]:
    """Return paragraphs copied verbatim by **all** discovered models.

    The helper delegates to :pyfunc:`check_for_exact` so there is no need to
    call it manually beforehand.

    Returns
    -------
    dict
        ``{dataset -> [example_indices_copied_by_all_models]}``
    """

    _, per_model_correct_idxs, all_models = check_for_exact(base_dir)

    copied_by_all: Dict[str, List[int]] = {}
    print("\n=== Paragraphs copied verbatim by ALL models ===")
    for dataset, idx_map in per_model_correct_idxs.items():
        common_idxs = [idx for idx, copied_by in idx_map.items() if all_models <= copied_by]
        copied_by_all[dataset] = sorted(common_idxs)
        print(f"{dataset:20s}: {len(common_idxs):4d} common paragraphs")

    return copied_by_all

######################################################################
# 3. Shrink the datasets                                              #
######################################################################

def write_common_exact_files(
    base_dir: Path = Path("results/realistic/arxiv"),
    out_suffix: str = "_common_exact",
) -> None:
    """Create reduced ``*.jsonl`` files containing only common exact‑matches.

    For every original ``arxiv_echo*.jsonl`` file this helper writes a new
    file in *the same directory* named ``<stem><out_suffix>.jsonl`` that keeps
    only the rows which were **perfectly copied by all models**.
    """

    copied_by_all = paragraphs_copied_by_all_models(base_dir)

    for model_dir in base_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for input_file in model_dir.rglob("*.jsonl"):
            if "arxiv_echo" not in input_file.stem:
                continue
            dataset = input_file.stem
            keep_indices = set(copied_by_all.get(dataset, []))
            if not keep_indices:
                # Nothing in *this* dataset is common across models → skip.
                continue

            with jsonlines.open(input_file) as reader:
                rows = list(reader)

            filtered_rows = [row for idx, row in enumerate(rows) if idx in keep_indices]
            if not filtered_rows:
                continue

            out_file = input_file.with_name(f"{input_file.stem}{out_suffix}{input_file.suffix}")
            with jsonlines.open(out_file, mode="w") as writer:
                writer.write_all(filtered_rows[:500])

            print(
                f"[✓] {out_file.relative_to(base_dir)} – "
                f"kept {len(filtered_rows[:500]):4d}/{len(rows):4d} examples"
            )


def sync_copy_arxiv(
    base_dir: Path = Path("results/realistic/arxiv"),
    copy_file: Path = Path("copy_arxiv.jsonl"),
) -> None:
    """Create or update ``copy_arxiv.jsonl`` so it contains **only** the
    paragraphs that appear in *any* of the ``*_common_exact.jsonl`` files.

    The resulting file strictly follows the schema::

        {"input": <paragraph>}

    Parameters
    ----------
    base_dir
        Root of the per‑model folders that already contain the
        ``*_common_exact.jsonl`` files (written by
        :pyfunc:`write_common_exact_files`).
    copy_file
        Location of the *copy* dataset.  If it exists, it will be **filtered
        in‑place** to keep only the allowed paragraphs; otherwise it will be
        created from scratch.
    common_suffix
        Suffix that identifies the reduced files; keep in sync with
        :pyfunc:`write_common_exact_files`.
    """

    # 1. Gather the reference paragraphs (ordered, unique).
    common_paragraphs: List[str] = []
    seen: Set[str] = set()
    for common_file in base_dir.rglob(f"*.jsonl"):
        with jsonlines.open(common_file) as reader:
            for row in reader:
                text = row.get("input_text")
                if text and text not in seen:
                    common_paragraphs.append(text)
                    seen.add(text)

    if not common_paragraphs:
        print("[!] No common‑exact files found – did you run write_common_exact_files()?")
        return

    # 2. If the copy‑file exists, filter its rows; else start fresh.
    kept_rows: List[Dict[str, str]] = []
    if copy_file.exists():
        with jsonlines.open(copy_file) as reader:
            for row in reader:
                text = row.get("input")
                if text in seen:
                    kept_rows.append({"input": text})
    else:
        kept_rows = [{"input": t} for t in common_paragraphs]

    # 3. Write back.
    with jsonlines.open(copy_file, mode="w") as writer:
        writer.write_all(kept_rows)
    print(f"[✓] {copy_file} – contains {len(kept_rows)} unique paragraphs")


######################################################################
# CLI convenience                                                     #
######################################################################

# if __name__ == "__main__":
#     # Running the module directly executes the full pipeline.
#     # write_common_exact_files()
#     sync_copy_arxiv()


if __name__ == "__main__":
    # base_dir = 'results/loremipsum'
    # output_dir = 'analysis/loremipsum'
    base_dir = 'results/realistic/loremipsum'
    output_dir = 'results/realistic/loremipsum/analysis'
    process_multiple_models(base_dir, output_dir, verbose=False)
    # process_sample_entry()
