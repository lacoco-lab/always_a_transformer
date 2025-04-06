import os
import json
import tqdm
from pathlib import Path
from typing import List

from transformers import AutoTokenizer
from bigram_compare import BigramComparator, compare_sequences


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
    model_to_hf_mapping = {
        "llama3.1_8B": "meta-llama/Llama-3.1-8B",
        "llama3.1_8B-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama3.1_70B": "meta-llama/Meta-Llama-3.1-70B",
        "llama3.3_70B-instruct": "meta-llama/Llama-3.3-70B-Instruct",
        "OLMo_7B-instruct": "allenai/OLMo-7B-Instruct"
    }
    
    # Get the huggingface model identifier
    hf_model_name = model_to_hf_mapping.get(model_name)
    
    if not hf_model_name:
        raise ValueError(f"Unknown model name: {model_name}. Supported models are: {list(model_to_hf_mapping.keys())}")
    
    # Load the tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, token=HF_TOKEN)
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
    

def process_jsonl_file(file_path: Path, output_file: Path = None, verbose: bool = True, tokenizer = None):
    """
    Process a JSONL file and compare gold answers with model answers using BigramComparator.
    
    Args:
        file_path: Path to the JSONL file
        output_file: Optional path to write results to
        verbose: Whether to print results to console
    """
    results = []
    overall_stats = {
        "total_examples": 0,
        "overall_similarity_sum": 0.0,
        "token_match_ratio_sum": 0.0,
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
                
                # Extract or generate tokenized versions
                gold_tokens = tokenize_text(gold_ans, tokenizer)
                model_tokens = tokenize_text(model_ans, tokenizer)
                
                # Compare the sequences
                metrics = BigramComparator.compare_token_sequences(
                    gold_tokens, model_tokens, return_formatted=False
                )
                
                # Add to overall stats
                overall_stats["total_examples"] += 1
                overall_stats["overall_similarity_sum"] += metrics["overall"]["levenshtein_similarity"]
                overall_stats["token_match_ratio_sum"] += metrics["token_level"]["token_match_ratio"]
                overall_stats["deterministic_similarity_sum"] += metrics["deterministic"]["levenshtein_similarity"]
                overall_stats["non_deterministic_similarity_sum"] += metrics["non_deterministic"]["levenshtein_similarity"]
                
                # Store result
                result = {
                    "line_num": line_num,
                    "metrics": metrics,
                    "formatted_metrics": BigramComparator.format_metrics(metrics)
                }
                results.append(result)
                
                # Print if verbose
                if verbose:
                    print(f"\n=== Entry {line_num} ===")
                    print(f"Gold Answer: {gold_ans[:100]}..." if len(gold_ans) > 100 else gold_ans)
                    print(f"Model Answer: {model_ans[:100]}..." if len(model_ans) > 100 else model_ans)
                    print(result["formatted_metrics"])
                    print("-" * 80)
                
            except json.JSONDecodeError:
                print(f"Error parsing JSON on line {line_num}. Skipping.")
            except Exception as e:
                print(f"Error processing line {line_num}: {str(e)}. Skipping.")
    
    # Calculate averages
    if overall_stats["total_examples"] > 0:
        avg_stats = {
            "avg_overall_similarity": overall_stats["overall_similarity_sum"] / overall_stats["total_examples"],
            "avg_token_match_ratio": overall_stats["token_match_ratio_sum"] / overall_stats["total_examples"],
            "avg_deterministic_similarity": overall_stats["deterministic_similarity_sum"] / overall_stats["total_examples"],
            "avg_non_deterministic_similarity": overall_stats["non_deterministic_similarity_sum"] / overall_stats["total_examples"],
        }
    else:
        avg_stats = {
            "avg_overall_similarity": 0.0,
            "avg_token_match_ratio": 0.0,
            "avg_deterministic_similarity": 0.0,
            "avg_non_deterministic_similarity": 0.0,
        }
    
    # Print summary
    if verbose:
        print("\n=== SUMMARY ===")
        print(f"Total examples processed: {overall_stats['total_examples']}")
        print(f"Average overall similarity: {avg_stats['avg_overall_similarity']:.4f}")
        print(f"Average token match ratio: {avg_stats['avg_token_match_ratio']:.4f}")
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
        with open('paste-2.txt', 'r', encoding='utf-8') as f:
            sample_entry = json.loads(f.read().strip())
        
        gold_ans = sample_entry.get('gold_ans', '')
        model_ans = sample_entry.get('answer', '')
        
        
        gold_tokens = tokenize_text(gold_ans)
        model_tokens = tokenize_text(model_ans)
        
        print("\n=== Sample Entry Analysis ===")
        print(f"Gold Answer Length: {len(gold_ans)} characters, {len(gold_tokens)} tokens")
        print(f"Model Answer Length: {len(model_ans)} characters, {len(model_tokens)} tokens")
        
        # Compare using BigramComparator
        formatted_metrics = compare_sequences(gold_tokens, model_tokens)
        print(formatted_metrics)
        
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
        tokenizer = get_tokenizer_for_model(model_name)
                
        model_results = {}
        curr_output_dir = output_dir / model_name 
        curr_output_dir.mkdir(parents=True, exist_ok=True)

        for input_file in model_result_dir.iterdir():
            if input_file.suffix != '.jsonl':
                continue

            # Same name as input file ; just in the output directory
            output_file = curr_output_dir / input_file.name            
            print(f"\nProcessing {input_file}...")
            _, avg_stats = process_jsonl_file(input_file, output_file, verbose, tokenizer=tokenizer)
            model_results[output_file.name] = avg_stats
        
        # TO REVIEW EVERYTHING BELOW HERE ... 
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
        print(f"{'Model':<25} | {'Dataset':<15} | {'Overall Sim':<12} | {'Token Match':<12} | {'Det. Sim':<12} | {'Non-Det. Sim':<12}")
        print('-' * 100)
        
        for model, datasets in all_results.items():
            for dataset, stats in datasets.items():
                print(f"{model:<25} | {dataset:<15} | {stats['avg_overall_similarity']:<12.4f} | {stats['avg_token_match_ratio']:<12.4f} | {stats['avg_deterministic_similarity']:<12.4f} | {stats['avg_non_deterministic_similarity']:<12.4f}")
    
    return all_results

if __name__ == "__main__":
    base_dir = 'results/loremipsum'
    output_dir = 'analysis/loremipsum'
    process_multiple_models(base_dir, output_dir, verbose=False)
    # Analysis of how and where the mistakes are -- would be required. 