import json
import tqdm
import random
import lorem
import random
from collections import defaultdict

from hf_olmo import OLMoTokenizerFast

# Load the tokenizer for the "allenai/OLMo-7B-Instruct" model
tokenizer = OLMoTokenizerFast.from_pretrained("allenai/OLMo-7B-Instruct")
results_file = [
        # "results/loremipsum/llama3.1_8B-instruct/zero-shot_chat_v0/500_exact_seed-5.jsonl",
        # "/Users/yash/Desktop/Lacoco/len-gen/results/loremipsum/llama3.1_8B-instruct/zero-shot_chat_v0/500_verbatim_seed-5.jsonl",
        # "/Users/yash/Desktop/Lacoco/len-gen/results/loremipsum/llama3.3_70B-instruct/zero-shot_chat_v0/500_exact_seed-5.jsonl",
        # "/Users/yash/Desktop/Lacoco/len-gen/results/loremipsum/llama3.3_70B-instruct/zero-shot_chat_v0/500_verbatim_seed-5.jsonl",
        "results/loremipsum/OLMo_7B-instruct/zero-shot_chat_v0/500_exact_seed-5.jsonl",
        "results/loremipsum/OLMo_7B-instruct/zero-shot_chat_v0/500_verbatim_seed-5.jsonl"
]


# def calculate_tokens_per_task(results_file):
#     with open(results_file, 'r') as jsonl_file:
#         lines = jsonl_file.readlines()
#         # For each lines, parse the string of the line as a dictionary
#         # and sum the number of tokens in each line
#         lines = [json.loads(line) for line in lines]
        
#         # Verified with LLaMa 3.1 8B outputs, that outputs always start with the following :: output": ["<", "paragraph", ">\n"
#         # "</", "paragraph", ">\n\n", "THE", "_END" OR "</", "paragraph", ">\n", "THE", "_END" ..
#         correct = 0
#         for line in lines:
#             gold_ans = line["gold_ans"]
#             full_answer = line["full_answer"]
#             # full_answer.replace('\n', ' ')
#             # Leave only characters and '.'s
#             full_answer_ = ''.join(e for e in full_answer if e.isalnum())# or e == '.')
#             gold_ans_ = ''.join(e for e in gold_ans if e.isalnum()) # or e == '.')
#             if gold_ans_ in full_answer_:
#                 correct += 1
#             else:
#                 print("Gold:", gold_ans)
#                 print("Full:", full_answer)
#                 break
#                 # pass
#         print("file", results_file, "Correct:", correct, "Total:", len(lines))

#     counts_task = defaultdict(int)
#     for sample_data in lines:
#         # First 
#         counts_task['first'] += 1
#         counts_task['last'] += 1

#         copied_tokens = sample_data['tokenized_output']
#         # Get the tokens and the indices of those tokens, that only appear once in the copied_tokens list
#         token_continuations = defaultdict(list)
#         # Traverse till the last token, and get all possible next tokens.
#         for index, token in enumerate(copied_tokens[:-1]):
#             token_continuations[token].append(copied_tokens[index+1])
        
#         # Get the induction head token-value pairs
#         for token, next_tokens in token_continuations.items():
#             if len(next_tokens) == 1:
#                 counts_task['induction_head'] += 1
#                 print("IH : TOKEN -", token, "NEXT TOKEN -", next_tokens[0])
#             elif len(next_tokens) > 1:
#                 # If there are multiple next tokens, and they are different, then it's a flip-flop task
#                 # Maybe, this needs to be changed, but let's go with this definition
#                 # Can save the actual token by gettig [0], [-1] from the list `next_tokens`
#                 # But it's not needed here, as the accuracy is 100% on copying. 
#                 counts_task['flip_flop_last'] += 1
#                 counts_task['flip_flop_first'] += 1
#                 if len(set(next_tokens)) > 1:
#                     counts_task['flip_flop_diff_last'] += 1
#                     counts_task['flip_flop_diff_first'] += 1
#                 print("FF : TOKEN -", token, "NEXT TOKEN -", next_tokens)
#         break
#     # The other symmteric tasks -- induction_left, flip_flop_last_left, flip_flop_first_left don't make sense here.
#     print(counts_task)


def add_new_variation(tokenizer, to_add_variation, max_tokens=2000):
    num_tokens = len(tokenizer.encode(to_add_variation))

    if num_tokens > max_tokens:
        # Then take only the max_tokens number of tokens from added_text
        to_add_variation = tokenizer.decode(tokenizer.encode(to_add_variation)[:max_tokens])
        num_tokens = max_tokens

    return {
        "input": to_add_variation,
        "olmo_num_tokens": num_tokens,
    }    


def generate_lorem_ipsum_variations(file_name, total_samples=1500, max_tokens=2000):
    variations = []
    generated_lp, added_text = 0, []

    while generated_lp < total_samples:
        # Generate a basic lorem ipsum text
        variation = lorem.text() + lorem.text() + lorem.text()
        # Split the sentences in the lorem-ipsum variation.
        variation = variation.replace('\n', ' ')
        sentences = variation.split('. ')
        if len(sentences) > 90:
            # Restrict number of sentences, as we want repetition to occur
            sentences = sentences[:90]
        
        while len(sentences) < 110:
            # Shuffle the order of the sentences
            random.shuffle(sentences)

            # End the loop if we have generated enough samples
            if generated_lp >= total_samples:
                break

            # Pick a random sentence, shuffle the words
            sentence_index = random.choice(range(len(sentences)))
            sentence = sentences[sentence_index]

            # Split into sentences
            words = sentence.split()
            # Shuffle the words, not the first one (as capital)
            to_shuffle = words[1:]
            random.shuffle(to_shuffle)
            # Set the shuffled words back
            words[1:] = to_shuffle
            sentences[sentence_index] = ' '.join(words)

             # Add duplicate sentences randomly
            if random.random() < 0.3:  # 30% chance to duplicate a sentence
                sentence_to_repeat = random.choice(sentences)
                sentences.append(sentence_to_repeat)
            
            # Duplicate words in random sentences
            if random.random() < 0.5:  # 50% chance to duplicate a word in a sentence
                sentence_index = random.choice(range(len(sentences)))
                words = sentences[sentence_index].split()
                word_to_duplicate = random.choice(words)
                words.append(word_to_duplicate)
                sentences[sentence_index] = ' '.join(words)

        # Join the sentences back into a string
        to_add_variation = '. '.join(sentences)

        if to_add_variation in added_text:
            continue
            
        generated_lp += 1
        added_text.append(to_add_variation)
        data_point = add_new_variation(tokenizer, to_add_variation, max_tokens)
        variations.append(data_point)

    # Write all variations to a .jsonl file
    with open(file_name, 'w') as jsonl_file:
        for item in variations:
            jsonl_file.write(json.dumps(item) + "\n")  # Write each variation as a separate line
    
    print(f"Generated {total_samples} variations and saved to {file_name}.")
    return variations


def main():
    # Specify the file to save the variations
    output_file = "datasets/500/loremipsum/data_bigger.jsonl"
    total_samples = 1500  # Number of variations to generate
    max_tokens = 2000  # Maximum number of tokens in each variation
    variations = generate_lorem_ipsum_variations(output_file, total_samples, max_tokens)


if __name__ == "__main__":
    main()
    # for f in results_file:
    #     calculate_tokens_per_task(f)
    
