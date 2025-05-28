
import torch
import random
from copy import deepcopy
from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset, IterableDataset


class CustomTokenizer():
    def __init__(self, vocab: list[str]):
        normal_tkn_num = len(vocab) # each element is a token

        self.bos_token = "<bos>"
        self.sep_token = "<sep>"
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"
        self.bos_token_id = normal_tkn_num
        self.sep_token_id = normal_tkn_num + 1
        self.eos_token_id = normal_tkn_num + 2
        self.pad_token_id = normal_tkn_num + 3
        self.special_token_ids = [self.bos_token_id, self.sep_token_id, self.eos_token_id, self.pad_token_id]
        self.special_tokens = [self.bos_token, self.sep_token, self.eos_token, self.pad_token]
        assert all(t not in vocab for t in self.special_tokens)
        
        # self.vocab = {"0": 0, "1": 1}
        self.vocab = {t: i for i, t in enumerate(vocab)}
        self.vocab[self.bos_token] = self.bos_token_id
        self.vocab[self.sep_token] = self.sep_token_id
        self.vocab[self.eos_token] = self.eos_token_id
        self.vocab[self.pad_token] = self.pad_token_id

        self.vocab_inv = {v: k for k, v in self.vocab.items()}
        self.padding_side = "right"

    def convert_ids_to_tokens(self, ids: list[int], rm_special=False):
        if rm_special:
            return [self.vocab_inv[i] for i in ids if i not in self.special_token_ids]
        else:
            return list(map(lambda x: self.vocab_inv[x], ids))

    def __len__(self):
        return len(self.vocab)
    

class _CopyDataset(IterableDataset):
    """
    Generic copy-memory dataset.

    Parameters
    ----------
    unique : bool
        If True, source tokens are drawn **without** replacement
        (`random.sample`); otherwise with replacement (`random.choices`).
    reverse : bool
        If True, the target side is the *reverse* of the source sequence.
    """
    def __init__(
        self,
        tokenizer,
        length_range: tuple[int, int],
        max_test_length: int,
        *,
        unique: bool,
        reverse: bool,
        no_offset: bool
    ):
        super().__init__()
        self.tok = tokenizer
        self.range_min, self.range_max = max(1, length_range[0]), length_range[1]
        self.max_test_length = max_test_length
        self.unique = unique
        self.reverse = reverse
        self.no_offset = no_offset

        # ---- consistency checks ------------------------------------------------
        if unique and max_test_length != -1:
            # need enough distinct (non-special) tokens for unique sampling
            assert len(tokenizer) - 4 >= max_test_length
        assert (max_test_length >= self.range_max) or (max_test_length == -1)
        # ------------------------------------------------------------------------

    # -------------------------------------------------------------------------- #
    # main iterator                                                              #
    # -------------------------------------------------------------------------- #
    def __iter__(self):
        rng = random
        # The rest are reserved for BOS, SEP, EOS, PAD
        vocab_size = len(self.tok) - 4

        while True:
            # 1) sample source sequence -----------------------------------------
            seq_len = rng.randint(self.range_min, self.range_max)
            if self.unique:
                src = rng.sample(range(vocab_size), seq_len)
            else:
                src = rng.choices(range(vocab_size), k=seq_len)

            # 2) build instance --------------------------------------------------
            tgt = src[::-1] if self.reverse else src

            inst = [self.tok.bos_token_id, *src,
                    self.tok.sep_token_id, *tgt,
                    self.tok.eos_token_id]

            # 3) build label (mask source + sep) ---------------------------------
            label = deepcopy(inst)
            label[:seq_len + 2] = [self.tok.pad_token_id,] * (seq_len + 2)

            # 4) positional ids offset -------------------------------------------
            if self.no_offset is True or self.max_test_length == -1:
                offset = 0
            else:
                # In case of training set of APE, apply this, otherwise don't
                # Multiplying by 2 ; as we want to consider the output length as well.
                offset = rng.randint(0, (self.max_test_length - seq_len) * 2)
            pos_ids = list(range(offset, offset + len(inst)))

            yield inst, pos_ids, label


class _RetrievalDataset(IterableDataset):
    """
    Generic retrieval-memory dataset.

    Sequence layout
    ---------------
    [bos]  context_tokens  [sep]  query_token  answer_token  [eos]

    Only *answer_token* is supervised (all earlier positions are masked).

    Parameters
    ----------
    tokenizer          : a tokenizer exposing .bos_token_id, .sep_token_id,
                         .eos_token_id, .pad_token_id and __len__()
    length_range       : (min_len, max_len) – length of *context* only
    max_test_length    : int  (-1 == unlimited, see copy-datasets paper)
    unique_query       : bool – should the query appear exactly once?
    side               : {"left","right"} – answer is neighbour on this side
    occurrence         : {"first","last"} – which occurrence to supervise
                         (ignored if unique_query=True)
    """

    def __init__(
        self,
        tokenizer,
        length_range: Tuple[int, int],
        max_test_length: int,
        *,
        unique_query: bool,
        side: str,
        occurrence: str = "first",
        no_offset: bool = False
    ):
        super().__init__()

        if side not in {"left", "right"}:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        if occurrence not in {"first", "last"}:
            raise ValueError(f"occurrence must be 'first' or 'last', got {occurrence!r}")

        self.tok = tokenizer
        self.range_min, self.range_max = length_range
        if self.range_min < 4:
            self.range_min = 4
        self.max_test_length = max_test_length
        self.unique = unique_query
        self.side = side
        self.which = occurrence
        self.no_offset = no_offset

        # ----- true vocabulary = every id except the four specials -----
        specials = {
            tokenizer.bos_token_id,
            tokenizer.sep_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
        }
        self.vocab: List[int] = [i for i in range(len(tokenizer)) if i not in specials]

        # sanity check from copy-datasets paper
        if max_test_length != -1 and max_test_length < self.range_max:
            raise ValueError("max_test_length must be ≥ max context length or -1.")

    # ------------------------------------------------------------------ #
    def __iter__(self):
        # Each worker gets its own Random() instance → no duplicate streams
        seed = torch.initial_seed()
        rng = random.Random(seed)

        while True:
            ctx_len = rng.randint(self.range_min, self.range_max - 1)
            # ---------------- build context ---------------- #

            if self.unique:
                # Pick a random character for induction
                query = rng.choice(self.vocab)
                # Construct the string with all other characters
                not_q = [t for t in self.vocab if t != query]
                context = rng.choices(not_q, k=ctx_len - 1)
                # insert the single query at a position that has a neighbour
                if self.side == "left":
                    pos = rng.randint(1, ctx_len - 1)
                else:  # right
                    pos = rng.randint(0, ctx_len - 2)
                # Insert the character somewhere.
                context.insert(pos, query)

            else:
                # Unrestricted choice of architecture : but since vocab is small
                # the context should have multiple occurrences of some strings
                context = rng.choices(self.vocab, k=ctx_len)
                # choose a query
                query = rng.choice(context)
                # choose guaranteed positions first/last with valid neighbours
                valid_positions = [idx for idx, c in enumerate(context) if c == query]

                # add first and last valid indices for the query
                pos = valid_positions[0] if self.which == "first" else valid_positions[-1]
                if len(valid_positions) == 1 or (self.which == 'last' and (pos > len(context) - 2)):
                    # If there is only instance of the query OR we are in the last variant
                    # and we picked the last instance of the query to be the last entry in the context
                    continue
            # -------------- compute answer -------------- #
            answer = context[pos - 1] if self.side == "left" else context[pos + 1]

            # -------------- build the full sample -------------- #
            instance = [
                self.tok.bos_token_id,
                *context,
                self.tok.sep_token_id,
                query,
                answer,
                self.tok.eos_token_id,
            ]

            # mask everything except the answer token
            label = [self.tok.pad_token_id] * len(instance)
            label[-2] = answer                      # answer is supervised

            # position ids (à la copy-datasets)
            if self.max_test_length == -1 or self.no_offset is True:
                pos_ids = list(range(len(instance)))
            else:
                max_offset = (self.max_test_length - ctx_len - 2)
                offset = rng.randint(0, max(0, max_offset))
                pos_ids = list(range(offset, offset + len(instance)))

            yield instance, pos_ids, label



def get_dataset(dataset_type, tokenizer, length_range, max_test_length, is_rope):
    if dataset_type == 'UL':
        return _RetrievalDataset(tokenizer, length_range, max_test_length,
                                 unique_query=True,  side="left",  occurrence="first", no_offset=is_rope)
    elif dataset_type == 'UR':
        return _RetrievalDataset(tokenizer, length_range, max_test_length,
                                 unique_query=True,  side="right",  occurrence="first", no_offset=is_rope)
    elif dataset_type == 'NLFirst':
        return _RetrievalDataset(tokenizer, length_range, max_test_length,
                                 unique_query=False,  side="left",  occurrence="first", no_offset=is_rope)
    elif dataset_type == 'NRFirst':
        return _RetrievalDataset(tokenizer, length_range, max_test_length,
                                 unique_query=False,  side="right",  occurrence="first", no_offset=is_rope)
    elif dataset_type == 'NLLast':
        return _RetrievalDataset(tokenizer, length_range, max_test_length,
                                 unique_query=False,  side="left",  occurrence="last", no_offset=is_rope)
    elif dataset_type == 'NRLast':
        return _RetrievalDataset(tokenizer, length_range, max_test_length,
                                 unique_query=False,  side="right",  occurrence="last", no_offset=is_rope)
    elif dataset_type == 'UF':
        return _CopyDataset(tokenizer, length_range, max_test_length, unique=True, reverse=False, no_offset=is_rope)
    elif dataset_type == 'UB':
        return _CopyDataset(tokenizer, length_range, max_test_length, unique=True, reverse=True, no_offset=is_rope)
    elif dataset_type == 'NF':
        return _CopyDataset(tokenizer, length_range, max_test_length, unique=False, reverse=False, no_offset=is_rope)
    elif dataset_type == 'NB':
        return _CopyDataset(tokenizer, length_range, max_test_length, unique=False, reverse=True, no_offset=is_rope)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Eager evaluation dataset
# ──────────────────────────────────────────────────────────────────────────────
class EvalDataset(Dataset):
    """
    Materialise *num_data* items from a potentially infinite
    :class:`IterableDataset` so that validation runs deterministically.

    Parameters
    ----------
    d          : streaming / iterable dataset that yields (input, pos, label)
    num_data   : number of elements to keep
    """
    def __init__(self, d: IterableDataset, num_data: int) -> None:
        super().__init__()
        self.data: List[Tuple[Any, ...]] = []           # keeps exact tuples
        for i, item in enumerate(d):
            if i >= num_data:                           # stop once we have enough
                break
            self.data.append(item)

    # Dataset API -------------------------------------------------------------
    def __getitem__(self, idx: int):
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Custom collate-fn
# ──────────────────────────────────────────────────────────────────────────────
class CustomCollator:
    """
    Pads (input_ids, position_ids, labels) triples to the same length.

    * `input_ids`   → padded with ``pad_id``  
    * `labels`      → padded with ``pad_id`` then converted to -100
                      (ignored by `nn.CrossEntropyLoss`)
    * `position_ids`→ extended by repeating the last valid position index
    """

    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    # ---------------------------------------------------------------------- #
    def __call__(
        self,
        examples: List[Tuple[List[int], List[int], List[int]]],
    ) -> Dict[str, torch.Tensor]:

        # unpack the batch
        input_ids, pos_ids, labels = map(list, zip(*examples))
        max_len = max(len(seq) for seq in input_ids)    # longest sequence

        # ---------- input_ids (pad with PAD) --------------------------------
        for seq in input_ids:
            seq.extend([self.pad_id] * (max_len - len(seq)))
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        # Add attention mask. 
        attention_mask = (input_ids_tensor != self.pad_id).long()

        # ---------- labels (pad + mask) -------------------------------------
        for seq in labels:
            seq.extend([self.pad_id] * (max_len - len(seq)))
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        labels_tensor[labels_tensor == self.pad_id] = -100  # ignore-index

        # ---------- position_ids (repeat last index) ------------------------
        for seq in pos_ids:
            seq.extend([seq[-1]] * (max_len - len(seq)))
        pos_ids_tensor = torch.tensor(pos_ids, dtype=torch.long)
        # Look any where you want with the attention mask, no need to put it.
        return {
            "input_ids":     input_ids_tensor,
            "position_ids":  pos_ids_tensor,
            "attention_mask": attention_mask, 
            "labels":        labels_tensor,
        }
