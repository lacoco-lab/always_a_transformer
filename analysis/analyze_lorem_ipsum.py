from argparse import ArgumentParser
import os
from utils import *

SEEDS = {
    "EXACT": "exact_seed-5.jsonl",
    "VERBATIM": "verbatim_seed-5.jsonl",
    "REVERSE": "reverse_seed-5.jsonl",
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK = "loremipsum"
RESULTS = "results"

parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="model", required=True,
                    help="Choose model to parse results from (e.g. llama_8b, llama_70b, olmo, pythia).")
parser.add_argument("-s", "--seed", dest="seed", required=True,
                    help="Choose seed to load (EXACT, VERBATIM, BIGGER).")
parser.add_argument("-pt", "--prompt_type", dest="prompt_type",
                    help="Choose prompt type: chat or completion", default="chat")
parser.add_argument("-v", "--version", dest="version",
                    help="Instruct or completion model version")
parser.add_argument("-nt", "--num_tokens", dest="num_tokens", default="500",
                    help="Number of tokens in the seed (default: 500).")

args = parser.parse_args()

seed_suffix = SEEDS.get(args.seed)
if not seed_suffix:
    raise ValueError(f"Seed '{args.seed}' not recognized")

seed_path = f"{args.num_tokens}_{seed_suffix}"
print(args.model)
model = get_model_name(args.model, args.version)

path = os.path.join(ROOT, RESULTS, TASK, model, 'zero-shot_'+args.prompt_type+'_v0', seed_path)
data = get_data(path)

OUTPUT_FILE = f"output_{model}_{args.prompt_type}_{args.seed}_{args.num_tokens}.txt"
new_data = []

if model == "llama3.1_8B-instruct" and args.seed == "VERBATIM":
    for line in data:
        if len(line["answer"].split("\n")) > 1:
            line["answer"] = line["answer"].split("\n")[1]
        new_data.append(line)

if new_data:
    data = new_data

inputs, outputs = [], []
incorrect_outputs = []
num_incorrect = 1

for line in data:
    cleaned_inputs = clean_text(line["input"])
    cleaned_outputs = clean_text(line["answer"])

    if not line["is_correct"]:  # or line["is_correct"] is False
        num_incorrect += 1
        incorrect_outputs.append({"input": cleaned_inputs, "output": cleaned_outputs})

    inputs.append(cleaned_inputs)
    outputs.append(cleaned_outputs)

print(f"Total incorrect: {num_incorrect} out of {len(data)}.")

accuracy_first, incorrect_firsts = get_accuracy_first(inputs, outputs)
accuracy_last, incorrect_lasts = get_accuracy_last(inputs, outputs)
accuracy_ind, absent_copies, total_unique = get_accuracy_unique_right(inputs, outputs)
accuracy_unique_bigrams, accuracy_repeated_bigrams = get_bigram_accuracy(inputs, outputs)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    print(f"First accuracy: {accuracy_first * 100}%")
    print(f"Last accuracy: {accuracy_last * 100}%")
    print(f"Unique token + right accuracy: {accuracy_ind * 100}% for total of {total_unique} tokens.")
    print(f"Unique bigrams accuracy: {accuracy_unique_bigrams * 100}%.")
    print(f"Repeated bigrams accuracy: {accuracy_repeated_bigrams * 100}%.")

    if incorrect_outputs:
        f.write("Marked as incorrect initially\n")
        for inp in incorrect_outputs:
            f.write(str(inp["input"].split()) + "\n" +
                    str(inp["output"].split()) + "\n" +
                    "=====================\n")

    f.write(f"First accuracy: {accuracy_first * 100}%\n")
    if incorrect_firsts:
        f.write(f"Incorrect first copies: {len(incorrect_firsts)}\n")
        for inp in incorrect_firsts:
            f.write(f"{inp['token']}\n{inp['input']}\n{inp['output']}\n=====================\n")

    f.write(f"Last accuracy: {accuracy_last * 100}%\n")
    if incorrect_lasts:
        f.write(f"Incorrect last copies: {len(incorrect_lasts)}\n")
        for inp in incorrect_lasts:
            f.write(f"{inp['token']}\n{inp['input']}\n{inp['output']}\n=====================\n")

    f.write(f"Unique token + right accuracy: {accuracy_ind * 100}% for total of {total_unique} tokens.\n")
    if absent_copies:
        f.write(f"Absent copies: {len(absent_copies)}\n")
        for inp in absent_copies:
            f.write(f"{inp['token']}\n{inp['input']}\n{inp['output']}\n=====================\n")

print(f"Output written to {OUTPUT_FILE}")
