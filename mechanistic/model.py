from functools import partial
from functools import partial
import torch


class ModelWithHooks:
    def __init__(self, model, task, remove_only_either_key_or_value, logger):
        self.model = model
        self.model.eval()
        self.logger = logger
        self.device = self.model.device
        assert task in ["unique_copy", "reversed_unique_copy"]
        self.task = task
        self.remove_only_either_key_or_value = remove_only_either_key_or_value

        self.hooks = {
            "attn": {
                layer: None
                for layer in range(len(self.model.transformer.h))
            },
        }

        self.activations = {
            layer: {
                "q": {},
                "k": {},
                "v": {}
            }
            for layer in range(len(self.model.transformer.h))
        }

        self.split_size = {}
        self.ln_f = self.model.transformer.ln_f
        for layer in range(len(self.model.transformer.h)):
            self.split_size[layer] = self.model.transformer.h[layer].attn.split_size
            self.hooks["attn"][layer] = self.model.transformer.h[layer].attn.register_forward_hook(partial(
                self.attn_hook, layer=layer
            ))

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def do_not_remove_anything(self, token_ids, save_activation):
        self.remove_edges = []
        self.remove_edges_to = {}
        self.save_activation = save_activation
        self.remove_with_zeros = False
    
    def remove_prev_token_heads(self, token_ids, len_of_str, beginning_of_first_part, beginning_of_second_part, save_activation, remove_with_zeros):
        previous_token_heads = [(prev_token, next_token) for prev_token, next_token in zip(range(beginning_of_second_part - 1), range(1, beginning_of_second_part))]
        self.remove_edges = previous_token_heads
        self.remove_edges_to = {
            to_token: [remove_from for remove_from, remove_to in self.remove_edges if remove_to == to_token]
            for to_token in range(len(token_ids))
        }
        self.logger("remove_prev_token_heads", self.remove_edges)
        self.save_activation = save_activation
        self.remove_with_zeros = remove_with_zeros
        if self.remove_only_either_key_or_value:
            self.remove_qkv = ["v"]
        else:
            self.remove_qkv = ["v", "k"]
    
    def remove_induction_heads(self, token_ids, len_of_str, beginning_of_first_part, beginning_of_second_part, save_activation, remove_with_zeros):
        if self.task == "unique_copy":
            induction_heads = [(prev_token, next_token) for prev_token, next_token in zip(range(beginning_of_first_part + 1, beginning_of_second_part + 1), range(beginning_of_second_part, beginning_of_second_part + len_of_str))]
            assert all([token_ids[head[0] - 1] == token_ids[head[1]] for head in induction_heads]), (induction_heads, len_of_str, beginning_of_second_part, induction_heads)
        elif self.task == "reversed_unique_copy":
            induction_heads = [(prev_token, next_token) for prev_token, next_token in zip(range(beginning_of_second_part - 1, beginning_of_first_part, -1), range(beginning_of_second_part, beginning_of_second_part + len_of_str))]
            assert all([token_ids[head[0] - 1] == token_ids[head[1]] for head in induction_heads]), (induction_heads, len_of_str, beginning_of_second_part, induction_heads)
        self.remove_edges = induction_heads
        self.remove_edges_to = {
            to_token: [remove_from for remove_from, remove_to in self.remove_edges if remove_to == to_token]
            for to_token in range(len(token_ids))
        }
        self.logger("remove_induction_heads", self.remove_edges)
        self.save_activation = save_activation
        self.remove_with_zeros = remove_with_zeros
        if self.remove_only_either_key_or_value:
            self.remove_qkv = ["k"]
        else:
            self.remove_qkv = ["v", "k"]

    def remove_antiinduction_heads(self, token_ids, len_of_str, beginning_of_first_part, beginning_of_second_part, save_activation, remove_with_zeros):
        if self.task == "unique_copy":
            induction_heads = [(prev_token, next_token) for prev_token, next_token in zip(range(beginning_of_first_part, beginning_of_second_part), range(beginning_of_second_part, beginning_of_second_part + len_of_str))]
            assert all([token_ids[head[0]] == token_ids[head[1]] for head in induction_heads]), (induction_heads, len_of_str, beginning_of_second_part, induction_heads)
        elif self.task == "reversed_unique_copy":
            induction_heads = [(prev_token, next_token) for prev_token, next_token in zip(range(beginning_of_second_part - 2, beginning_of_first_part - 1, -1), range(beginning_of_second_part, beginning_of_second_part + len_of_str))]
            assert all([token_ids[head[0]] == token_ids[head[1]] for head in induction_heads]), (induction_heads, len_of_str, beginning_of_second_part, induction_heads)
        self.remove_edges = induction_heads
        self.remove_edges_to = {
            to_token: [remove_from for remove_from, remove_to in self.remove_edges if remove_to == to_token]
            for to_token in range(len(token_ids))
        }
        self.logger("remove_antiinduction_heads", self.remove_edges)
        self.save_activation = save_activation
        self.remove_with_zeros = remove_with_zeros
        if self.remove_only_either_key_or_value:
            self.remove_qkv = ["v"]
        else:
            self.remove_qkv = ["v", "k"]

    def attn_hook(self, module, input, output, layer):
        self.logger("layer", layer, "remove_edges_to", self.remove_edges_to)
        if not self.remove_edges and not self.save_activation:
            return output
        
        outputs = []

        for to_token in range(input[0].shape[1]):
            self.logger("to_token", to_token)
            remove_hook = self.model.transformer.h[layer].attn.c_attn.register_forward_hook(partial(
                self.c_attn_hook, layer=layer, remove_activations=self.remove_edges_to, running_for_token=to_token, remove_with_zeros=self.remove_with_zeros,
                save_activation = self.save_activation
            ))
            result_for_token = module.forward(input[0])[0]
            outputs.append(result_for_token[:, to_token, :])
            remove_hook.remove()

        return torch.stack(outputs, dim=1), None

    def c_attn_hook(self, module, input, output,
                    layer, remove_activations, running_for_token, remove_with_zeros, save_activation):
        # input.shape == (bsz, seqlen, hidden_size)
        # output.shape == torch.vstack((bsz, seqlen, embed_dim), (bsz, seqlen, embed_dim), (bsz, seqlen, embed_dim))
        # output = query_states, key_states, value_states

        output_activations = []
        for i, activation in enumerate(["q", "k", "v"]):
            initial_activation = module.forward(input[0]).split(self.split_size[layer], dim=2)[i]
            if remove_activations and activation in self.remove_qkv:
                for remove_what in remove_activations[running_for_token]:
                    self.logger(f"removing {remove_what} to {running_for_token} in {activation}")
                    initial_activation[:, remove_what, :] = (0. if remove_with_zeros else self.activations[layer][activation][running_for_token][:, remove_what, :].to(self.device))
            if save_activation:
                self.activations[layer][activation][running_for_token] = initial_activation.detach().cpu()
            output_activations.append(initial_activation)

        return torch.cat(output_activations, dim=2)

    def remove_hooks(self):
        for layer in range(len(self.model.transformer.h)):
            self.hooks["attn"][layer].remove()


class BigModelWithHooks:
    def __init__(self, model, task, remove_only_either_key_or_value, logger):
        self.model = model
        self.model.eval()
        self.logger = logger
        self.device = self.model.device
        assert task in ["unique_copy", "reversed_unique_copy"]
        self.task = task
        self.remove_only_either_key_or_value = remove_only_either_key_or_value

        self.hooks = {
            "attn": {
                layer: None
                for layer in range(len(self.model.model.layers))
            },
        }

        self.activations = {
            layer: {
                "q": {},
                "k": {},
                "v": {}
            }
            for layer in range(len(self.model.model.layers))
        }

        for layer in range(len(self.model.model.layers)):
            self.hooks["attn"][layer] = self.model.model.layers[layer].self_attn.register_forward_hook(partial(
                self.attn_hook, layer=layer
            ), with_kwargs=True)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def do_not_remove_anything(self, token_ids, save_activation):
        self.remove_edges = []
        self.remove_edges_to = {}
        self.save_activation = save_activation
        self.remove_with_zeros = False
        self.len_of_input = len(token_ids)
    
    def remove_prev_token_heads(self, token_ids, len_of_str, beginning_of_first_part, beginning_of_second_part, save_activation, remove_with_zeros):
        previous_token_heads = [(prev_token, next_token) for prev_token, next_token in zip(range(beginning_of_second_part - 1), range(1, beginning_of_second_part))]
        self.remove_edges = previous_token_heads
        self.remove_edges_to = {
            to_token: [remove_from for remove_from, remove_to in self.remove_edges if remove_to == to_token]
            for to_token in range(len(token_ids))
        }
        self.logger("remove_prev_token_heads", self.remove_edges)
        self.save_activation = save_activation
        self.remove_with_zeros = remove_with_zeros
        self.len_of_input = len(token_ids)
        if self.remove_only_either_key_or_value:
            self.remove_qkv = ["v"]
        else:
            self.remove_qkv = ["v", "k"]
    
    def remove_induction_heads(self, token_ids, len_of_str, beginning_of_first_part, beginning_of_second_part, save_activation, remove_with_zeros):
        if self.task == "unique_copy":
            induction_heads = [(prev_token, next_token) for prev_token, next_token in zip(range(beginning_of_first_part + 1, beginning_of_second_part + 1), range(beginning_of_second_part, beginning_of_second_part + len_of_str))]
            assert all([token_ids[head[0] - 1] == token_ids[head[1]] for head in induction_heads]), (induction_heads, len_of_str, beginning_of_first_part, beginning_of_second_part)
        elif self.task == "reversed_unique_copy":
            induction_heads = [(prev_token, next_token) for prev_token, next_token in zip(range(beginning_of_second_part - 1, beginning_of_first_part, -1), range(beginning_of_second_part, beginning_of_second_part + len_of_str))]
            assert all([token_ids[head[0] - 1] == token_ids[head[1]] for head in induction_heads]), (induction_heads, len_of_str, beginning_of_first_part, beginning_of_second_part, list(range(beginning_of_second_part - 1, beginning_of_first_part, -1)), list(range(beginning_of_second_part, beginning_of_second_part + len_of_str)))
        self.remove_edges = induction_heads
        self.remove_edges_to = {
            to_token: [remove_from for remove_from, remove_to in self.remove_edges if remove_to == to_token]
            for to_token in range(len(token_ids))
        }
        self.logger("remove_induction_heads", self.remove_edges)
        self.save_activation = save_activation
        self.remove_with_zeros = remove_with_zeros
        self.len_of_input = len(token_ids)
        if self.remove_only_either_key_or_value:
            self.remove_qkv = ["k"]
        else:
            self.remove_qkv = ["v", "k"]

    def remove_antiinduction_heads(self, token_ids, len_of_str, beginning_of_first_part, beginning_of_second_part, save_activation, remove_with_zeros):
        if self.task == "unique_copy":
            induction_heads = [(prev_token, next_token) for prev_token, next_token in zip(range(beginning_of_first_part, beginning_of_second_part), range(beginning_of_second_part, beginning_of_second_part + len_of_str))]
            assert all([token_ids[head[0]] == token_ids[head[1]] for head in induction_heads]), (induction_heads, len_of_str, beginning_of_first_part, beginning_of_second_part)
        elif self.task == "reversed_unique_copy":
            induction_heads = [(prev_token, next_token) for prev_token, next_token in zip(range(beginning_of_second_part - 2, beginning_of_first_part - 1, -1), range(beginning_of_second_part, beginning_of_second_part + len_of_str))]
            assert all([token_ids[head[0]] == token_ids[head[1]] for head in induction_heads]), (induction_heads, len_of_str, beginning_of_first_part, beginning_of_second_part)
        self.remove_edges = induction_heads
        self.remove_edges_to = {
            to_token: [remove_from for remove_from, remove_to in self.remove_edges if remove_to == to_token]
            for to_token in range(len(token_ids))
        }
        self.logger("remove_antiinduction_heads", self.remove_edges)
        self.save_activation = save_activation
        self.remove_with_zeros = remove_with_zeros
        self.len_of_input = len(token_ids)
        if self.remove_only_either_key_or_value:
            self.remove_qkv = ["v"]
        else:
            self.remove_qkv = ["v", "k"]

    def attn_hook(self, module, input, input_kwargs, output, layer):
        self.logger("layer", layer, "remove_edges_to", self.remove_edges_to)
        if not self.remove_edges and not self.save_activation:
            return output  # Return the original output to maintain the forward pass
        
        self.logger("attn input", input)
        self.logger("attn input kwargs", input_kwargs)
        
        outputs = []

        for to_token in range(self.len_of_input):
            self.logger("to_token", to_token)
            remove_hooks = []
            remove_hooks.append(self.model.model.layers[layer].self_attn.q_proj.register_forward_hook(partial(
                self.c_attn_hook, layer=layer, remove_activations=self.remove_edges_to, running_for_token=to_token, remove_with_zeros=self.remove_with_zeros,
                save_activation = self.save_activation, current_activation="q"
            )))
            remove_hooks.append(self.model.model.layers[layer].self_attn.k_proj.register_forward_hook(partial(
                self.c_attn_hook, layer=layer, remove_activations=self.remove_edges_to, running_for_token=to_token, remove_with_zeros=self.remove_with_zeros,
                save_activation = self.save_activation, current_activation="k"
            )))
            remove_hooks.append(self.model.model.layers[layer].self_attn.v_proj.register_forward_hook(partial(
                self.c_attn_hook, layer=layer, remove_activations=self.remove_edges_to, running_for_token=to_token, remove_with_zeros=self.remove_with_zeros,
                save_activation = self.save_activation, current_activation="v"
            )))
            result_for_token = module.forward(**input_kwargs)[0]
            outputs.append(result_for_token[:, to_token, :])
            for remove_hook in remove_hooks:
                remove_hook.remove()

        return torch.stack(outputs, dim=1), None

    def c_attn_hook(self, module, input, output,
                    layer, remove_activations, running_for_token, remove_with_zeros, save_activation, current_activation):
        initial_activation = output
        if remove_activations and current_activation in self.remove_qkv:
            for remove_what in remove_activations[running_for_token]:
                self.logger(f"removing {remove_what} to {running_for_token} in {current_activation}")
                initial_activation[:, remove_what, :] = (0. if remove_with_zeros else self.activations[layer][current_activation][running_for_token][:, remove_what, :].to(self.device))
        if save_activation:
            self.activations[layer][current_activation][running_for_token] = initial_activation.detach().cpu()

        return initial_activation

    def remove_hooks(self):
        for layer in range(len(self.model.model.layers)):
            self.hooks["attn"][layer].remove()