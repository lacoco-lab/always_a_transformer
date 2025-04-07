import json
import random

import lorem
from typing import List, Dict, Any, Optional


class LoremIpsumGenerator:
    """
    A class to generate and manage lorem ipsum text variations with configurable properties.
    """
    
    def __init__(
        self, 
        tokenizer,
        output_file: str = "dataset.jsonl",
        num_sentences: int = 200,
        lorem_paragraphs: int = 5,
        total_samples: int = 1500,
        max_tokens: int = 2000,
        duplicate_sentence_prob: float = 0.3,
        duplicate_word_prob: float = 0.5,
        shuffle_sentence_prob: float = 1.0,
        duplicate_count: int = 4,
        tokenizer_name: str = "default"
    ):
        """
        Initialize the Lorem Ipsum generator with configurable parameters.
        
        Args:
            tokenizer: The tokenizer to use for counting tokens
            output_file: Path to save the generated variations
            num_sentences: Target number of sentences per variation
            lorem_paragraphs: Number of lorem ipsum paragraphs to generate initially
            total_samples: Total number of variations to generate
            max_tokens: Maximum number of tokens allowed per variation
            duplicate_sentence_prob: Probability of duplicating a sentence
            duplicate_word_prob: Probability of duplicating words in a sentence
            shuffle_sentence_prob: Probability of shuffling words in a sentence
            duplicate_count: How many times to duplicate sentences/words
            tokenizer_name: Name identifier for the tokenizer
        """
        self.tokenizer = tokenizer
        self.output_file = output_file
        self.num_sentences = num_sentences
        self.lorem_paragraphs = lorem_paragraphs
        self.total_samples = total_samples
        self.max_tokens = max_tokens
        self.duplicate_sentence_prob = duplicate_sentence_prob
        self.duplicate_word_prob = duplicate_word_prob
        self.shuffle_sentence_prob = shuffle_sentence_prob
        self.duplicate_count = duplicate_count
        self.tokenizer_name = tokenizer_name
        
        # Track generated variations
        self.variations = []
        self.unique_texts = set()
        self.generated_count = 0
        
    def tokenize_and_truncate(self, text: str) -> Dict[str, Any]:
        """
        Tokenize text and truncate if it exceeds max_tokens.
        
        Args:
            text: The text to tokenize and potentially truncate
            
        Returns:
            Dictionary with input text and token count
        """
        num_tokens = len(self.tokenizer.encode(text))
        
        if num_tokens > self.max_tokens:
            # Truncate to max_tokens
            text = self.tokenizer.decode(self.tokenizer.encode(text)[:self.max_tokens])
            num_tokens = self.max_tokens
            
        return {
            "input": text,
            f"{self.tokenizer_name}_num_tokens": num_tokens
        }
    
    def generate_base_text(self) -> str:
        """Generate base lorem ipsum text from multiple paragraphs."""
        variation = ''
        for _ in range(self.lorem_paragraphs):
            variation += lorem.text()
        return variation.replace('\n', ' ')
    
    def process_sentences(self, sentences: List[str]) -> List[str]:
        """
        Process the sentences by shuffling, duplicating words, and adding duplicates.
        
        Args:
            sentences: List of sentences to process
            
        Returns:
            Processed list of sentences
        """
        if len(sentences) < self.num_sentences:
            # Shuffle the order of the sentences
            random.shuffle(sentences)
            
            # Process random sentences with various operations
            self._shuffle_words_in_random_sentence(sentences)
            self._duplicate_random_sentence(sentences)
            self._duplicate_words_in_random_sentence(sentences)
            
        return sentences
    
    def _shuffle_words_in_random_sentence(self, sentences: List[str]) -> None:
        """Shuffle words in a randomly selected sentence."""
        if random.random() < self.shuffle_sentence_prob and sentences:
            sentence_index = random.choice(range(len(sentences)))
            sentence = sentences[sentence_index]
            
            # Split into words
            words = sentence.split()
            if len(words) > 1:
                # Shuffle the words, preserving the first one (capital)
                to_shuffle = words[1:]
                random.shuffle(to_shuffle)
                words[1:] = to_shuffle
                sentences[sentence_index] = ' '.join(words)
    
    def _duplicate_random_sentence(self, sentences: List[str]) -> None:
        """Add duplicate sentences randomly."""
        if random.random() < self.duplicate_sentence_prob and sentences:
            sentence_to_repeat = random.choice(sentences)
            # Repeat the sentence multiple times
            for _ in range(self.duplicate_count):
                sentences.append(sentence_to_repeat)
    
    def _duplicate_words_in_random_sentence(self, sentences: List[str]) -> None:
        """Duplicate words in random sentences."""
        if random.random() < self.duplicate_word_prob and sentences:
            sentence_index = random.choice(range(len(sentences)))
            words = sentences[sentence_index].split()
            if words:
                word_to_duplicate = random.choice(words)
                # Add the word multiple times
                for _ in range(self.duplicate_count):
                    words.append(word_to_duplicate)
                sentences[sentence_index] = ' '.join(words)
    
    def generate_variation(self) -> Optional[Dict[str, Any]]:
        """
        Generate a single lorem ipsum variation.
        
        Returns:
            Dictionary with the generated variation or None if generation failed
        """
        # Generate base text
        base_text = self.generate_base_text()
        sentences = base_text.split('. ')
        
        # Process until we have enough sentences
        while len(sentences) < self.num_sentences:
            sentences = self.process_sentences(sentences)
        
        # Join the sentences back into a string
        text = '. '.join(sentences)
        
        # Check if this text has been generated before
        if text in self.unique_texts:
            return None
        
        # Add to tracking sets
        self.unique_texts.add(text)
        
        # Tokenize and potentially truncate
        return self.tokenize_and_truncate(text)
    
    def generate_all_variations(self) -> List[Dict[str, Any]]:
        """
        Generate all variations according to specified parameters.
        
        Returns:
            List of all generated variations
        """
        self.variations = []
        self.unique_texts = set()
        self.generated_count = 0
        
        print(f"Generating {self.total_samples} lorem ipsum variations...")
        
        while self.generated_count < self.total_samples:
            variation = self.generate_variation()
            if variation:
                self.variations.append(variation)
                self.generated_count += 1
                if self.generated_count % 100 == 0:
                    print(f"Generated {self.generated_count}/{self.total_samples} variations")
        
        # Save to file
        self.save_to_file()
        
        return self.variations
    
    def save_to_file(self) -> None:
        """Save all variations to the specified output file."""
        with open(self.output_file, 'w') as jsonl_file:
            for item in self.variations:
                jsonl_file.write(json.dumps(item) + "\n")
        
        print(f"Successfully saved {len(self.variations)} variations to {self.output_file}")
    
    @classmethod
    def get_config_preset(cls, preset_name: str) -> Dict[str, Any]:
        """
        Get predefined configuration presets for different token sizes.
        
        Args:
            preset_name: Name of the preset configuration
            
        Returns:
            Dictionary with preset configuration parameters
        """
        presets = {
            "1500_tokens": {
                "lorem_paragraphs": 3,
                "num_sentences": 110,
                "max_tokens": 1500
            },
            "3000_tokens": {
                "lorem_paragraphs": 5,
                "num_sentences": 200,
                "max_tokens": 3000
            },
            "4000_tokens": {
                "lorem_paragraphs": 7,
                "num_sentences": 300,
                "max_tokens": 4000
            },
            "5000_tokens": {
                "lorem_paragraphs": 15,
                "num_sentences": 400,
                "max_tokens": 5000
            }
        }
        
        return presets.get(preset_name, {})

