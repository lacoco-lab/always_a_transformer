import json
import random
import lorem
import random

from hf_olmo import OLMoTokenizerFast

# Load the tokenizer for the "allenai/OLMo-7B-Instruct" model
tokenizer = OLMoTokenizerFast.from_pretrained("allenai/OLMo-7B-Instruct")


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
        variation = lorem.text()  # Generate a basic lorem ipsum text
        # Split the sentences in the lorem-ipsum variation.
        variation = variation.replace('\n', ' ')
        sentences = variation.split('. ')
        
        for _ in range(20):
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
    output_file = "datasets/500/loremipsum/data.jsonl"
    total_samples = 1500  # Number of variations to generate
    max_tokens = 2000  # Maximum number of tokens in each variation
    variations = generate_lorem_ipsum_variations(output_file, total_samples, max_tokens)


if __name__ == "__main__":
    main()
