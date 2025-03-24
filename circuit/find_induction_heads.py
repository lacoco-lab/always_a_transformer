from transformer_lens.head_detector import detect_head
from transformer_lens import HookedTransformer
import torch

from .prompt_utils import *
from .intervention_utils import *
from .model_utils import *
from .extract_utils import *

from collections import Counter
import pickle
import argparse
import os

def generate_repeated_random_tokens(model, batch=1000, seq_len=50, seed=0):
    set_seed(seed)
    prefix = (torch.ones(batch, 1) * model.tokenizer.bos_token_id).long()
    rep_tokens_half = torch.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=torch.int64)
    rep_tokens = torch.cat([prefix, rep_tokens_half, rep_tokens_half], dim=-1).to(model.cfg.device)
    return rep_tokens

def find_random_induction_heads(model, batch=1000, seq_len=50, seed=0):
    rep_tokens = generate_repeated_random_tokens(model, batch, seq_len, seed)
    prompts = [model.tokenizer.decode(x) for x in rep_tokens]

    head_scores = detect_head(model, prompts, "induction_head", exclude_bos=False, exclude_current_token=False, error_measure="abs")

    results = dict()
    for layer, layer_scores in enumerate(head_scores):
        for head, score in enumerate(layer_scores):
            results[f'{layer}.{head}'] = score.item()

    return results

def find_eigenvalue_copying_heads(model):
    OV_circuit_all_heads = model.OV
    OV_circuit_all_heads_eigenvalues = OV_circuit_all_heads.eigenvalues 
    OV_copying_score = OV_circuit_all_heads_eigenvalues.sum(dim=-1).real / OV_circuit_all_heads_eigenvalues.abs().sum(dim=-1)

    results = dict()
    for layer, layer_scores in enumerate(OV_copying_score):
        for head, score in enumerate(layer_scores):
            results[f'{layer}.{head}'] = score.item()

    return results

def find_copying_heads(model, tokenizer, model_config):
    ## exclude fraction of frequent bpe tokens from random sequences
    frequent_excluded_ranks = int(0.04 * tokenizer.vocab_size)
    ## exclude both most and least frequent tokens
    rank_start, rank_end = frequent_excluded_ranks, tokenizer.vocab_size - frequent_excluded_ranks
    assert rank_start < rank_end and rank_end > 0
    rank_choice_list = np.arange(rank_start, rank_end)
    num_layers = model_config['n_layers']
    num_heads = model_config['n_heads']
    final = []

    with torch.no_grad():
        for seed in tqdm(range(100)):
            set_seed(seed)
            length = 4 * (seed * 2 + 25)
            generate_ranks = np.random.choice(rank_choice_list, size=length, replace=False)
            ## append a bos_token in the beginning to ensure normal model behaviour
            generate_ids = torch.tensor([tokenizer.bos_token_id] + generate_ranks)
            generate_ids = torch.unsqueeze(generate_ids, 0)
            input_shape = generate_ids.size()
            input_ids = generate_ids.view(-1, input_shape[-1]).to(model.device)

            hidden_states = model.gpt_neox.embed_in(input_ids)
            attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device) # TODO verify attention mask
            position_ids = attention_mask.long().cumsum(-1) - 1 # print position ids
            position_ids.masked_fill_(attention_mask == 0, 1) # TODO why they do this?

            copy_scores = torch.zeros((model_config['n_layers'], model_config['n_heads']))

            for layer_index in range(model_config['n_layers']):
                # Get the attention mechanism for the specific layer
                attention_layer = model.gpt_neox.layers[layer_index].attention
                
                query, key, value, present = attn_projections_and_rope(attention_layer, hidden_states=hidden_states, position_ids=position_ids)
                attn_output, attn_weights = attn(attention_layer, query, key, value)

                for head_index in range(model_config['n_heads']):
                    attn_output_i = attn_output[:, head_index]
                    attn_weights_i = attn_weights[:, head_index]

                    h, l, d = attn_output_i.shape
                    
                    w_i = attention_layer.dense.weight[:, head_index * d : (head_index + 1) * d]
                    # TODO not sure what to do with bias
                    attn_output_proj = torch.nn.functional.linear(attn_output_i, w_i)#, attention_layer.dense.bias / num_heads)

                    output = model.embed_out(attn_output_proj) # TODO how does layer norm work? separate this by computing layer mean std and aply to each compoennt
                    logits = torch.nn.functional.softmax(output, dim = -1)
                    
                    _, ind = torch.sort(attn_weights_i[0], dim = 1)
                    max_ind = ind[:, -1]
                    # max_ind = torch.argmax(attn_weights[0][0], dim=1)
                    c = 0
                    
                    for j in range(l):
                        c += 1
                        assert (max_ind[j] <= j)
                        ## tokens that can be attended to in the current time step ie 0 to j
                        attendable_input = input_ids[0][:(j+1)] # TODO check batch
                        ## logits of attendable tokens
                        attendable_logits = logits[0][j][attendable_input]
                        ## mean of the logits
                        mean_of_logits = attendable_logits.mean()
                        ## raise logits
                        raised_logits = attendable_logits - mean_of_logits
                        ## relu over raised logits
                        relu_raised_logits = torch.nn.functional.relu(raised_logits)
                        relu_raised_logit_max_ind = relu_raised_logits[max_ind[j]].item()
                        relu_raised_logit_all = relu_raised_logits.sum().item()
                        ## ratio of raised logit
                        copying_score = 0
                        ## edgecase: if all logits are of equal value then relu_raised_logit_all can be 0
                        if relu_raised_logit_all != 0:
                            copying_score = relu_raised_logit_max_ind / relu_raised_logit_all
                        copy_scores[layer_index][head_index] += copying_score
                    
                    copy_scores[layer_index][head_index] = copy_scores[layer_index][head_index] / c

            final.append(copy_scores.unsqueeze(0))
        final = torch.cat(final, dim=0)
        final = final.mean(dim=0)

    results = dict()
    for layer, layer_scores in enumerate(final):
        for head, score in enumerate(layer_scores):
            results[f'{layer}.{head}'] = score.item()

    return results

                    
def find_induction_heads(dataset, model, is_llama, n_icl_examples=10, n_trials=100):
    # Get the prompts
    prepend_bos = not is_llama

    prompts = []
    for i in range(n_trials):
        word_pairs = dataset['train'][np.random.choice(len(dataset['train']),n_icl_examples, replace=False)]
        word_pairs_test = dataset['valid'][np.random.choice(len(dataset['valid']),1, replace=False)]
        prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=prepend_bos)

        query = prompt_data['query_target']['input']
        token_labels, prompt_string = get_token_meta_labels(prompt_data, model.tokenizer, query)
        prompts.append(prompt_string)

    head_scores = detect_head(model, prompts, "induction_head", exclude_bos=False, exclude_current_token=False, error_measure="abs")

    results = dict()
    for layer, layer_scores in enumerate(head_scores):
        for head, score in enumerate(layer_scores):
            results[f'{layer}.{head}'] = score.item()

    return results



def main():
    print("Starting to find induction heads...")
    parser = argparse.ArgumentParser(description='Run a script using a config file and CLI arguments.')

    parser.add_argument('--dataset_name', help='Name of the dataset to be loaded', type=str, required=False)
    parser.add_argument('--model_name', default='attn-only-2l', help='Name of the model to load')
    parser.add_argument('--ckpt', type=int, default=None, help='Ckpt of the model to load', required=False)
    parser.add_argument('--seed', help='Randomized seed', type=int, required=False, default=42)
    parser.add_argument('--save_path_root', default='/scratch/users/kayoyin/icl-heads', help='save_path_root path')
    parser.add_argument('--n_shots', help="Number of shots in each in-context prompt", type =int, required=False, default=10)
    parser.add_argument('--n_trials', help="Number of in-context prompts to average over", type=int, required=False, default=100)
    parser.add_argument('--force', help='Force overwrite of existing files', type=int, required=False, default=0)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    datasets = args.dataset_name
    model_name = MODEL_NAME_DICT[args.model_name]
    ckpt = args.ckpt
    seed = args.seed
    save_path_root = args.save_path_root
    n_shots = args.n_shots
    n_trials = args.n_trials
    force = bool(int(args.force))
    set_seed(seed)
    print(f"Force: {force}")

    
    if ckpt is not None:
        tmp_path = f'{save_path_root}/tmp/induction-heads/{model_name.split("/")[-1].replace("-deduped", "")}-{ckpt}'
        output_path = f'{save_path_root}/rebut_outputs/heads/{model_name.split("/")[-1].replace("-deduped", "")}-{ckpt}'
    else:
        tmp_path = f'{save_path_root}/tmp/induction-heads/{model_name.split("/")[-1].replace("-deduped", "")}'
        output_path = f'{save_path_root}/rebut_outputs/heads/{model_name.split("/")[-1].replace("-deduped", "")}'


    # Load the dataset
    if datasets is None:
        datasets = ['ag_news_train', 'antonym_train', 'capitalize_train', 'capitalize_first_letter_train', 'commonsense_qa_train', 'country-capital_train', 'country-currency_train', 'english-french_train', 'english-german_train', 'english-spanish_train', 'landmark-country_train', 'lowercase_first_letter_train', 'national_parks_train', 'next_item_train', 'park-country_train', 'person-instrument_train', 'person-occupation_train', 'person-sport_train', 'present-past_train', 'product-company_train', 'sentiment_train', 'singular-plural_train', 'synonym_train', 'adjective_v_verb_3_train', 'adjective_v_verb_5_train', 'animal_v_object_3_train', 'animal_v_object_5_train', 'choose_first_of_3_train', 'choose_first_of_5_train', 'choose_last_of_3_train', 'choose_last_of_5_train', 'choose_middle_of_3_train', 'choose_middle_of_5_train', 'color_v_animal_3_train', 'color_v_animal_5_train', 'concept_v_object_3_train', 'concept_v_object_5_train', 'conll2003_location_train', 'conll2003_organization_train', 'conll2003_person_train', 'fruit_v_animal_3_train', 'fruit_v_animal_5_train', 'object_v_concept_3_train', 'object_v_concept_5_train', 'verb_v_adjective_3_train', 'verb_v_adjective_5']
    else:
        datasets = [datasets]

    is_llama = False
    if 'llama' in model_name.lower():
        hfmodel, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device)
        model = HookedTransformer.from_pretrained(model_name, hf_model=hfmodel, torch_dtype=torch.bfloat16, fold_ln=False, center_writing_weights=False, center_unembed=False, tokenizer=tokenizer)
        is_llama = True
    elif ckpt is not None:
        # model = HookedTransformer.from_pretrained(model_name, checkpoint_value=ckpt).to(device)
        model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device, ckpt=ckpt)
    else:
        # For copy only
        # model = HookedTransformer.from_pretrained(model_name).to(device) 
        model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device=device)

    model.eval()

    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print("Model loaded successfully!")

    copying_results = find_copying_heads(model, tokenizer, model_config)

    with open(f'{output_path}/copy_zerobias.pkl', 'wb') as f:
        pickle.dump(copying_results, f)

    return # Copy only

    random_results = find_random_induction_heads(model, seed=seed)

    with open(f'{output_path}/random_induction.pkl', 'wb') as f:
        pickle.dump(random_results, f)

    mean_results = dict()
    count = 0
    for dataset_name in datasets:
        print(f"Running on dataset: {dataset_name}")
        

        if os.path.exists(f'{tmp_path}/{dataset_name}.pkl') and not force:
            results = pickle.load(open(f'{tmp_path}/{dataset_name}.pkl', 'rb'))
        else:
            dataset = load_dataset(dataset_name, seed=0)
            results = find_induction_heads(dataset, model, is_llama, n_shots, n_trials)

            with open(f'{tmp_path}/{dataset_name}.pkl', 'wb') as f:
                pickle.dump(results, f)

        mean_results = {k: mean_results.get(k, 0) + results.get(k, 0) for k in set(mean_results) | set(results)}
        count += 1

    mean_results = {k: v / count for k, v in mean_results.items()}

    



    with open(f'{output_path}/induction.pkl', 'wb') as f:
        pickle.dump(mean_results, f)



    print("Finished finding induction heads successfully!")

if __name__ == "__main__":
    main()