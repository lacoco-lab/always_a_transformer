import os
import json
import random
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional


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
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, token=HF_TOKEN, trust_remote_code=True)
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer for {model_name} ({hf_model_name}): {e}")
        return None

class ControlledBigramGenerator:
    """
    A class to generate paragraphs with controlled bigram patterns.
    Creates text with two types of bigrams:
    1. Deterministic: First token always followed by the same second token
    2. Non-deterministic: First token can be followed by various possible tokens
    
    The tokens are chosen to avoid semantic relationships that might provide
    predictive clues, making this ideal for testing LLM copying performance.
    """
    
    def __init__(
        self,
        tokenizer,
        output_file: str = "controlled_bigram_dataset.jsonl",
        deterministic_ratio: float = 0.5,  # Ratio of deterministic bigrams
        vocabulary_size: int = 280,       # Number of unique words
        paragraph_length: int = 450,       # Target words per paragraph
        total_samples: int = 1500,         # Number of paragraphs to generate
        max_tokens: int = 2000,            # Maximum tokens per paragraph
        tokenizer_name: str = "default",   # Tokenizer identifier
    ):
        """
        Initialize the controlled bigram generator.
        
        Args:
            tokenizer: Tokenizer to use for counting tokens
            output_file: Path to save generated paragraphs
            deterministic_ratio: Proportion of deterministic bigrams (0.0 to 1.0)
            vocabulary_size: Size of vocabulary to use
            paragraph_length: Target number of words per paragraph
            total_samples: Total number of paragraphs to generate
            max_tokens: Maximum tokens allowed per paragraph
            tokenizer_name: Name identifier for the tokenizer
        """
        self.tokenizer = tokenizer
        self.output_file = output_file
        self.deterministic_ratio = deterministic_ratio
        self.vocabulary_size = vocabulary_size
        self.paragraph_length = paragraph_length
        self.total_samples = total_samples
        self.max_tokens = max_tokens
        self.tokenizer_name = tokenizer_name
        
        # Output tracking
        self.paragraphs = []
        self.unique_texts = set()
        self.generated_count = 0
        
        # Generate vocabulary
        self.vocabulary = self._generate_vocabulary()
        # Randomly assign a bunch of words to deterministic / non - deterministic
        self.deterministic_words = list()
        self.non_deterministic_words = list()
        self._split_vocabulary()
        self.deterministic_map = self._create_deterministic_bigrams()
        
        # Statistics for evaluation
        self.bigram_stats = {}

    def _is_clean_tokenization(self, word: str) -> bool:
        """
        Check if a word tokenizes cleanly (reconstructs to the original).
        
        Args:
            word: Word to check
            
        Returns:
            True if tokenization is clean, False otherwise
        """
        tokens = self.tokenizer.encode(word)
        decoded = self.tokenizer.decode(tokens)
        # Allow for potential extra spaces in the decoded text
        return decoded.strip() == word

    def _generate_vocabulary(self) -> List[str]:
        """
        Generate a vocabulary of semantically unrelated words.
        Focus on words that tokenize cleanly with the provided tokenizer.
        
        Returns:
            List of words for the vocabulary
        """
        # Start with common, simple words that should tokenize well
        base_words = [
            # Common nouns
            "table", "chair", "book", "door", "window", "pen", "paper", "clock",
            "phone", "desk", "car", "tree", "house", "road", "sky", "water", "fire",
            "earth", "moon", "sun", "star", "cloud", "rain", "snow", "wind", "river",
            "mountain", "ocean", "forest", "garden", "city", "town", "school", "store",
            "market", "office", "building", "street", "park", "bridge", "field", "path",
            "floor", "wall", "roof", "room", "hall", "door", "box", "cup", "plate", "fork",
            "knife", "spoon", "bowl", "glass", "bottle", "can", "jar", "bag", "hat", "coat",
            "shirt", "shoe", "sock", "glove", "ring", "watch", "belt", "key", "lock", "coin",
            "card", "dog", "cat", "bird", "fish", "horse", "cow", "sheep", "goat", "pig",
            "duck", "chicken", "bee", "fly", "ant", "snake", "frog", "bear", "wolf", "fox",
            "wood", "stone", "metal", "glass", "cloth", "gold", "silver", "iron", "steel",
            "bus", "train", "plane", "ship", "boat", "bike", "truck", "van", "cake", "bread",
            "milk", "juice", "tea", "coal", "oil", "gas", "salt", "soap", "mail", "film",
            
            # Colors
            "red", "blue", "green", "yellow", "orange", "purple", "black", "white", "gray",
            "brown", "pink", "gold", "silver", "bronze", "teal", "navy", "lime", "cream", "beige",
            
            # Adjectives
            "big", "small", "tall", "short", "wide", "narrow", "thick", "thin", "heavy",
            "light", "hot", "cold", "warm", "cool", "soft", "hard", "smooth", "rough",
            "fast", "slow", "loud", "quiet", "bright", "dark", "clear", "foggy", "clean",
            "dirty", "new", "old", "young", "modern", "fresh", "wet", "dry", "sharp", 
            "dull", "sweet", "sour", "bitter", "salty", "spicy", "mild", "strong", "weak",
            "rich", "poor", "brave", "wise", "smart", "dumb", "kind", "mean", "nice", "rude",
            "glad", "sad", "mad", "calm", "wild", "tame", "good", "bad", "low", "high", 
            "deep", "flat", "raw", "ripe", "full", "empty", "open", "shut", "safe", "true",
            "pure", "whole", "fine", "real", "fake", "firm", "tight", "loose", "round", "square",
            
            # Body parts
            "head", "face", "eye", "ear", "nose", "mouth", "lip", "chin", "neck", "chest",
            "arm", "hand", "leg", "foot", "back", "hair", "bone", "skin", "nail", "tooth",
            
            # Nature
            "lake", "hill", "rock", "sand", "soil", "seed", "leaf", "root", "stem", "bud",
            "weed", "moss", "rose", "lily", "pine", "oak", "elm", "mud", "dust", "ash", 
            "wave", "foam", "mist", "fog", "dew", "ice", "frost", "salt", "cave", "cliff",
            
            # Time
            "day", "week", "year", "hour", "dawn", "dusk", "noon", "night", "time", "date",
            
            # Numbers
            "one", "two", "six", "ten", "half", "pair", "lots", "none", "some", "all",
            
            # Verbs (simple present tense)
            "walk", "run", "jump", "swim", "fly", "eat", "drink", "look", "see", "hear",
            "read", "write", "speak", "talk", "sit", "stand", "lie", "sleep", "wake", "move",
            "stop", "start", "turn", "work", "play", "rest", "sing", "dance", "think", "feel",
            "know", "help", "give", "take", "make", "build", "fix", "break", "push", "pull",
            "drop", "lift", "cut", "grow", "fall", "rise", "buy", "sell", "pay", "meet",
            "find", "lose", "hide", "seek", "drive", "ride", "cook", "wash", "dry", "clean",

            # Function words
            "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for", "with",
            "by", "from", "as", "is", "are", "was", "were", "be", "been", "being", "this", "that",
            "these", "those", "it", "its", "they", "them", "their", "theirs", "what", "which", "who",
            "whom", "whose", "when", "where", "why", "how", "not", "no", "yes", "all", "any", "some",
            "many", "few", "each", "every", "other",

            # Adverbs 
            "now", "then", "here", "there", "just", "also", "very", "well",
            "only", "even", "back", "down", "still", "too", "much", "more",
            "most", "less", "least", "often", "never", "ever", "soon", "once",
            "twice", "quite", "so", "thus", "when", "why", "how", "where", "out",
            "up", "off", "on", "in", "over", "yet", "ago"            
        ]
        
        # Remove duplicates
        base_words = list(set(base_words))
        
        # Filter words that tokenize well (ideally to single tokens)
        good_words = []
        for word in base_words:
            tokens = self.tokenizer.encode(word)
            # Check if the word tokenizes to a single token or cleanly
            if len(tokens) == 2:
                good_words.append(word)
        print("GOOD - ", len(good_words), "BASE - ", len(base_words))        
        print(f"Found {len(good_words)} words that tokenize well out of {len(base_words)} base words")
        
        # Generate additional words if needed
        result = good_words.copy()
        # If we have too many words, select a random subset
        if len(result) > self.vocabulary_size:
            random.shuffle(result)
            result = result[:self.vocabulary_size]
        else:
            # If we need more words, create them
            while len(result) < self.vocabulary_size:
                # Create words like "word1", "word2", etc.
                for i in range(len(result), self.vocabulary_size):
                    new_word = f"word{i+1}"
                    if self._is_clean_tokenization(new_word):
                        result.append(new_word)
                        if len(result) >= self.vocabulary_size:
                            break
                
                # If still not enough, add simple arbitrary words
                if len(result) < self.vocabulary_size:
                    for i in range(len(result), self.vocabulary_size):
                        result.append(f"w{i}")
        
        # Shuffle the vocabulary
        random.shuffle(result)
        return result[:self.vocabulary_size]
    
    def _split_vocabulary(self) -> None:
        """
        Split vocabulary into sets for deterministic and non-deterministic bigrams.
        """
        deterministic_count = int(self.vocabulary_size * self.deterministic_ratio)
        # Create two non-overlapping sets
        self.deterministic_words = list(self.vocabulary[:deterministic_count])
        self.non_deterministic_words = [w for w in self.vocabulary if w not in self.deterministic_words]
    
    def _create_deterministic_bigrams(self) -> Dict[str, str]:
        """
        Create deterministic bigram mappings.
        Each first word always leads to the same second word.
        
        Returns:
            Dictionary mapping first words to their fixed second words
        """
        # The same mapping always from deterministic to some non-deterministic word
        return {first_word: random.choice(self.non_deterministic_words) for first_word in self.deterministic_words}

    def generate_paragraph(self) -> str:
        """
        An alternative implementation for paragraph generation.
        This version creates a random paragraph first, then selectively enforces bigram patterns.
        
        Returns:
            A paragraph string with the specified bigram patterns
        """
        # Step 1: Generate a completely random sequence of words
        words = []
        for _ in range(self.paragraph_length):
            # Append a bunch of non-deterministic words
            words.append(random.choice(self.non_deterministic_words))
            
        # Step 2: Randomly select positions to enforce patterns        
        target_deterministic = self.deterministic_ratio * self.paragraph_length
        target_positions = get_random_positions(target_deterministic, self.paragraph_length)

        # Track usage : number of bigrams in a paragraph is 1 less than length of the paragraph
        bigram_usage = {"deterministic": 0, "non_deterministic": self.paragraph_length - 1}
        
        # First pass: Apply deterministic patterns
        for position in target_positions:
            words[position] = random.choice(self.deterministic_words)
            # Change the word at that position to be a deterministic word
            # And the word folowing it, to be the map from it.
            if position + 1 < self.paragraph_length:
                words[position + 1] = self.deterministic_map[words[position]]
                bigram_usage["deterministic"] += 1
                bigram_usage["non_deterministic"] -= 1
            
        # Join words into a paragraph
        paragraph = " ".join(words)
        # Record stats for this paragraph
        self.bigram_stats[paragraph] = bigram_usage
        return self.tokenize_and_truncate(paragraph)

    def tokenize_and_truncate(self, text: str) -> Dict[str, Any]:
        """
        Tokenize text and truncate if it exceeds max_tokens.
        
        Args:
            text: Text to tokenize and potentially truncate
            
        Returns:
            Dictionary with text and token count
        """
        num_tokens = len(self.tokenizer.encode(text))
        if num_tokens > self.max_tokens:
            # Truncate to max_tokens
            text = self.tokenizer.decode(self.tokenizer.encode(text)[:self.max_tokens])
            num_tokens = self.max_tokens
        
        return {
            "input": text,
            f"{self.tokenizer_name}_num_tokens": num_tokens,
            "bigram_stats": self.bigram_stats.get(text, {})
        }
    
    def generate_variation(self) -> Optional[Dict[str, Any]]:
        """
        Generate a single paragraph with controlled bigram patterns.
        
        Returns:
            Dictionary with the generated paragraph or None if generation failed
        """
        # Generate paragraph
        text_dict = self.generate_paragraph()
        
        # Check if this text has been generated before)
        while text_dict['input'] in self.unique_texts:
            text_dict = self.generate_paragraph()
        
        # Add to tracking sets
        self.unique_texts.add(text_dict['input'])
        
        return text_dict
    
    def generate_all_variations(self) -> List[Dict[str, Any]]:
        """
        Generate all paragraphs according to specified parameters.
        
        Returns:
            List of all generated paragraphs
        """
        self.paragraphs = []
        self.unique_texts = set()
        self.generated_count = 0
        
        print(f"Generating {self.total_samples} controlled bigram paragraphs...")
        while self.generated_count < self.total_samples:
            variation = self.generate_variation()
            # print(variation)
            if variation:
                self.paragraphs.append(variation)
                self.generated_count += 1
                
                if self.generated_count % 100 == 0:
                    print(f"Generated {self.generated_count}/{self.total_samples} paragraphs")
        
        # Save to file
        self.save_to_file()
        
        # Print bigram statistics
        self._print_bigram_stats()
        
        return self.paragraphs
    
    def save_to_file(self) -> None:
        """Save all paragraphs to the specified output file."""
        with open(self.output_file, 'w') as jsonl_file:
            for item in self.paragraphs:
                jsonl_file.write(json.dumps(item) + "\n")
        
        print(f"Successfully saved {len(self.paragraphs)} paragraphs to {self.output_file}")
    
    def _print_bigram_stats(self) -> None:
        """Print statistics about bigram usage in the generated paragraphs."""
        total_deterministic = sum(stats.get("deterministic", 0) for stats in self.bigram_stats.values())
        total_non_deterministic = sum(stats.get("non_deterministic", 0) for stats in self.bigram_stats.values())
        total_random = sum(stats.get("random", 0) for stats in self.bigram_stats.values())
        total_bigrams = total_deterministic + total_non_deterministic + total_random
        
        print("\nBigram Statistics:")
        print(f"Total bigrams: {total_bigrams}")
        print(f"Deterministic bigrams: {total_deterministic} ({total_deterministic/total_bigrams:.2%})")
        print(f"Non-deterministic bigrams: {total_non_deterministic} ({total_non_deterministic/total_bigrams:.2%})")
        print(f"Random bigrams: {total_random} ({total_random/total_bigrams:.2%})")
        print(f"Target deterministic ratio: {self.deterministic_ratio:.2%}")


def get_random_positions(target_count: int, curr_length: int = 100):
    """
    Get a list of random positions to replace non-deterministic with deterministic
    bigrams. 

    
    Args:
        target_count: Number of positions to select
        curr_length: Current length of the paragraph (default is 100)
        
    Returns:
        List of random positions
    """
    positions = set()
    while len(positions) < target_count:
        pos = random.randint(0, curr_length - 2)
        # Check if pos + 1 and pos - 1 don't exist in the set
        if pos + 1 not in positions and pos - 1 not in positions and pos not in positions:
            positions.add(pos)
    return sorted(list(positions)) 


if __name__ == "__main__":
    # Arg parse
    import argparse
    parser = argparse.ArgumentParser(description="Generate controlled bigram paragraphs.")
    parser.add_argument("--deterministic_ratio", type=float, default=0.2, help="Ratio of deterministic bigrams")
    parser.add_argument("--vocabulary_size", type=int, default=350, help="Vocabulary size")
    parser.add_argument("--paragraph_length", type=int, default=1500, help="Paragraph length")
    parser.add_argument("--total_samples", type=int, default=1500, help="Total samples to generate")
    parser.add_argument("--max_tokens", type=int, default=2000, help="Max tokens per paragraph")

    # Get parameters from argparse
    args = parser.parse_args()

    # for tokenizer_name in ["llama3.1_8B", "llama3.1_8B-instruct", "llama3.1_70B", "llama3.3_70B-instruct", "OLMo_7B-instruct"]:
    for tokenizer_name in ["llama3.1_8B"]:
        tokenizer = get_tokenizer_for_model(tokenizer_name)
        output_file = os.path.join("datasets", "copy_controlled", "_".join([tokenizer_name, str(args.deterministic_ratio), str(args.max_tokens)]) + '.jsonl')

        # Create a generator with custom settings
        generator = ControlledBigramGenerator(
            tokenizer=tokenizer,
            output_file=output_file,
            deterministic_ratio=args.deterministic_ratio,
            vocabulary_size=args.vocabulary_size,  # Vocabulary size
            paragraph_length=args.paragraph_length,  # Paragraph length
            total_samples=args.total_samples,  # Dataset size for prompting a given model
            max_tokens=args.max_tokens,   # Limit to these many tokens per sample
            tokenizer_name=tokenizer_name    # Identifier for the tokenizer
        )
        # Generate all paragraphs
        paragraphs = generator.generate_all_variations()