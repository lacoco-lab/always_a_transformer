import random
from utils import set_seed
import numpy as np
from torch.utils.data import Dataset
import json


class UniqueCopyDataset(Dataset):
    def __init__(self, tokenizer, num_instances: int = 100, seed: int = 0, length_range: tuple[int, int] = [10, 50],
                 sep_token: str = ">", n_shot: int = 3, fewshot_sep_token: str = "#", vocab: list = None):
        super().__init__()
        set_seed(seed)
        if vocab is None:
            self.vocab = (
                [chr(letter_ascii) for letter_ascii in range(ord('a'), ord('z') + 1)]
            )
        else:
            self.vocab = vocab
        assert len(tokenizer.encode(sep_token, add_special_tokens=False)) == 1, (sep_token, tokenizer.encode(sep_token, add_special_tokens=False))

        assert tokenizer.eos_token is not None

        if tokenizer.bos_token is not None:
            bos_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
            if bos_id is None or bos_id == tokenizer.unk_token_id:
                tokenizer.bos_token = None

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        space_id = tokenizer.encode(" ", add_special_tokens=False)[0]


        self.data = []
        for i in range(num_instances):
            fewshots = []
            for fewshot in range(n_shot):
                length = random.randint(length_range[0], length_range[1])
                fewshots.append(np.random.choice(self.vocab, size=length, replace=False))
            fewshots_tokens = [
                tokenizer.convert_ids_to_tokens([
                    c for c in tokenizer.encode(" " + " ".join(copy_string), add_special_tokens=False)
                    if c != space_id
                ]) for copy_string in fewshots
            ]
            assert all([len(fewshots_tokens[shot]) == len(fewshots[shot]) for shot in range(n_shot)]), (fewshots_tokens, fewshots)
            instance_no_bos_eos = fewshot_sep_token.join([
                "".join(copy_string_tokens) + sep_token + "".join(copy_string_tokens)
                for copy_string_tokens in fewshots_tokens
            ])
            instance = (tokenizer.bos_token if tokenizer.bos_token else "") + instance_no_bos_eos + tokenizer.eos_token
            instance_tokens = (
                ([tokenizer.bos_token] if tokenizer.bos_token else []) + \
                    [token for fewshot_tokens in fewshots_tokens for token in fewshot_tokens + [sep_token] + fewshot_tokens + [fewshot_sep_token]][:-1] + \
                        [tokenizer.eos_token])
            instance_token_ids = tokenizer.convert_tokens_to_ids(instance_tokens)

            len_of_test = len(fewshots[-1])
            beginning_of_first_part = sum([len(s) * 2 for s in fewshots[:-1]]) + (n_shot - 1) * 2 + (1 if tokenizer.bos_token else 0)
            beginning_of_second_part = sum([len(s) * 2 for s in fewshots[:-1]]) + (n_shot - 1) * 2 + (1 if tokenizer.bos_token else 0) + len(fewshots[-1]) + 1

            label_no_pad_bos_eos = "".join(fewshots_tokens[-1]) 
            label = tokenizer.pad_token * (beginning_of_second_part - 1) + label_no_pad_bos_eos + tokenizer.pad_token * 2
            label_tokens = [tokenizer.pad_token] * (beginning_of_second_part - 1) + [token for token in fewshots_tokens[-1]] + [tokenizer.pad_token] * 2
            label_token_ids = tokenizer.convert_tokens_to_ids(label_tokens)

            assert len(instance_tokens) == len(label_tokens)
            self.data.append((
                instance,
                instance_token_ids,
                instance_tokens,
                label,
                label_token_ids,
                label_tokens,
                len_of_test,
                beginning_of_first_part,
                beginning_of_second_part,
            ))
    def __getitem__(self, item):
        return self.data[item]
    def __len__(self):
        return len(self.data)
    

class ReverseUniqueCopyDataset(Dataset):
    def __init__(self, tokenizer, num_instances: int = 100, seed: int = 0, length_range: tuple[int, int] = [10, 50],
                 sep_token: str = ">", n_shot: int = 3, fewshot_sep_token: str = "#", vocab: list = None):
        super().__init__()
        set_seed(seed)
        if vocab is None:
            self.vocab = (
                [chr(letter_ascii) for letter_ascii in range(ord('a'), ord('z') + 1)]
            )
        else:
            self.vocab = vocab
        assert len(tokenizer.encode(sep_token, add_special_tokens=False)) == 1, (sep_token, tokenizer.encode(sep_token, add_special_tokens=False))

        assert tokenizer.eos_token is not None

        if tokenizer.bos_token is not None:
            bos_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
            if bos_id is None or bos_id == tokenizer.unk_token_id:
                tokenizer.bos_token = None

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        space_id = tokenizer.encode(" ", add_special_tokens=False)[0]


        self.data = []
        for i in range(num_instances):
            fewshots = []
            for fewshot in range(n_shot):
                length = random.randint(length_range[0], length_range[1])
                fewshots.append(np.random.choice(self.vocab, size=length, replace=False))
            fewshots_tokens = [
                tokenizer.convert_ids_to_tokens([
                    c for c in tokenizer.encode(" " + " ".join(copy_string), add_special_tokens=False)
                    if c != space_id
                ]) for copy_string in fewshots
            ]
            assert all([len(fewshots_tokens[shot]) == len(fewshots[shot]) for shot in range(n_shot)]), (fewshots_tokens, fewshots)
            instance_no_bos_eos = fewshot_sep_token.join([
                "".join(copy_string_tokens) + sep_token + "".join(reversed(copy_string_tokens))
                for copy_string_tokens in fewshots_tokens
            ])
            instance = (tokenizer.bos_token if tokenizer.bos_token else "") + instance_no_bos_eos + tokenizer.eos_token
            instance_tokens = (
                ([tokenizer.bos_token] if tokenizer.bos_token else []) + \
                    [token for fewshot_tokens in fewshots_tokens for token in fewshot_tokens + [sep_token] + list(reversed(fewshot_tokens)) + [fewshot_sep_token]][:-1] + \
                        [tokenizer.eos_token])
            instance_token_ids = tokenizer.convert_tokens_to_ids(instance_tokens)

            len_of_test = len(fewshots[-1])
            beginning_of_first_part = sum([len(s) * 2 for s in fewshots[:-1]]) + (n_shot - 1) * 2 + (1 if tokenizer.bos_token else 0)
            beginning_of_second_part = sum([len(s) * 2 for s in fewshots[:-1]]) + (n_shot - 1) * 2 + (1 if tokenizer.bos_token else 0) + len(fewshots[-1]) + 1

            label_no_pad_bos_eos = "".join(reversed(fewshots_tokens[-1]))
            label = tokenizer.pad_token * (beginning_of_second_part - 1) + label_no_pad_bos_eos + tokenizer.pad_token * 2
            label_tokens = [tokenizer.pad_token] * (beginning_of_second_part - 1) + [token for token in reversed(fewshots_tokens[-1])] + [tokenizer.pad_token] * 2
            label_token_ids = tokenizer.convert_tokens_to_ids(label_tokens)

            assert len(instance_tokens) == len(label_tokens)
            self.data.append((
                instance,
                instance_token_ids,
                instance_tokens,
                label,
                label_token_ids,
                label_tokens,
                len_of_test,
                beginning_of_first_part,
                beginning_of_second_part,
            ))
    def __getitem__(self, item):
        return self.data[item]
    def __len__(self):
        return len(self.data)
    

class FinetuningDatasetSameAsNotFinetuning(Dataset):
    def __init__(self, tokenizer, num_instances: int = 100, seed: int = 0, length_range: tuple[int, int] = [10, 50],
                 n_shot: int = 3, dataset_name: str = "unique_copy", vocab: list = None):
        super().__init__()
        set_seed(seed)
        if vocab is None:
            self.vocab = (
                [chr(letter_ascii) for letter_ascii in range(ord('a'), ord('z') + 1)]
            )
        else:
            self.vocab = vocab

        sep_token = ">"

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        self.data = []
        for i in range(num_instances):
            fewshots = []
            for fewshot in range(n_shot):
                length = random.randint(length_range[0], length_range[1])
                fewshots.append(np.random.choice(self.vocab, size=length, replace=False))
            actual_instance = fewshots[-1]
            label_no_pad = "".join(actual_instance) if dataset_name == "unique_copy" else "".join(reversed(actual_instance))
            len_of_str = len(label_no_pad)
            instance = "".join(fewshots[-1]) + sep_token + label_no_pad
            label = tokenizer.pad_token * len_of_str + label_no_pad + tokenizer.pad_token
            beginning_of_second_part = len_of_str + 1
            beginning_of_first_part = 0
            self.data.append((
                instance,
                tokenizer.convert_tokens_to_ids(list(instance)),
                tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids(list(instance))),
                label,
                tokenizer.convert_tokens_to_ids([tokenizer.pad_token] * len_of_str + list(label_no_pad) + [tokenizer.pad_token]),
                tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids([tokenizer.pad_token] * len_of_str + list(label_no_pad) + [tokenizer.pad_token])),
                len_of_str,
                beginning_of_first_part,
                beginning_of_second_part
            ))
    def __getitem__(self, item):
        return self.data[item]
    def __len__(self):
        return len(self.data)