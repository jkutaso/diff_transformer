#%%
from train_model import train_both_models, save_metrics, ExperimentConfig
from transformer_lens.utils import tokenize_and_concatenate
from transformers import GPT2TokenizerFast
import torch as t
import datasets
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = t.device('mps' if t.backends.mps.is_available() else 'cuda' if t.cuda.is_available() else 'cpu')
print(device)

first_experiment_config = ExperimentConfig(
    n_layers = 2,
    d_model = 64,
    d_head = 32,
    n_heads = 4,
    n_ctx = 256,
    batch_size = 16,
    epochs = 10,
    max_steps_per_epoch = 100,
    lr = 1e-3,
    weight_decay = 1e-2
)

all_experiment_configs = []
d_model_list = [64, 128, 256]
n_layers_list = [2, 4, 8]
n_heads_list = [4, 8]
for d_model in d_model_list:
    for n_layers in n_layers_list:
        for n_heads in n_heads_list:
            experiment_config = first_experiment_config.copy()
            experiment_config.d_model = d_model
            experiment_config.n_layers = n_layers
            experiment_config.n_heads = n_heads
            all_experiment_configs.append(experiment_config)
#%%
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train").remove_columns("meta")
tokenized_dataset = tokenize_and_concatenate(dataset, tokenizer, device, max_length = experiment_config.n_ctx, column_name="text", add_bos_token=True, num_proc=4)
dataset_dict = tokenized_dataset.train_test_split(test_size=1000)
#%%
for i, experiment_config in enumerate(all_experiment_configs):
    print(f"Training model {i+1} of {len(all_experiment_configs)}")
    base_cfg = experiment_config.get_config_base(tokenizer)
    diff_cfg = experiment_config.get_config_diff(tokenizer)
    diff_metrics, base_metrics = train_both_models(experiment_config, tokenizer, dataset_dict, device)
    save_metrics(base_metrics, diff_metrics, experiment_config)
    # Clear GPU memory cache
    if t.cuda.is_available():
        t.cuda.empty_cache()
    # Clear MPS memory if using Apple Silicon
    if t.backends.mps.is_available():
        import gc
        gc.collect()
        t.mps.empty_cache()
# %%
