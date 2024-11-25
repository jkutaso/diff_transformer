
#%%
import torch as t
from diff_transformer import DiffTransformer, Config
from transformer import Transformer
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
import datasets
from transformer_lens.utils import tokenize_and_concatenate
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Any
from torch import Tensor
from jaxtyping import Int, Float
from tqdm import tqdm
import pandas as pd
import os
import json
#%%
@dataclass
class TransformerTrainingArgs():
    batch_size: int
    epochs: int
    max_steps_per_epoch: int
    lr: float
    weight_decay: float

def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], 
    tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:

    log_probs = logits.log_softmax(dim=-1)
    # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    log_probs_for_tokens = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

    return log_probs_for_tokens

@dataclass
class ExperimentConfig:
    n_layers: int
    d_model: int
    d_head: int
    n_heads: int
    n_ctx: int
    max_steps_per_epoch: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    
    def get_config_base(self, tokenizer) -> Config:
        return Config(
            debug=False,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_head=self.d_head,
            d_mlp=self.d_model * 4,
            n_layers=self.n_layers,
            n_ctx=self.n_ctx,
            d_vocab=tokenizer.vocab_size
        )
    
    def get_config_diff(self, tokenizer) -> Config:
        cfg = self.get_config_base(tokenizer)
        cfg.d_head = cfg.d_head // 2
        return cfg
    
    def get_training_args(self) -> TransformerTrainingArgs:
        return TransformerTrainingArgs(
            batch_size=self.batch_size,
            epochs=self.epochs,
            max_steps_per_epoch=self.max_steps_per_epoch,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
    def copy(self):
        return ExperimentConfig(**vars(self))
    
@dataclass
class TrainingMetrics:
    step: int
    epoch: int
    training_loss: float
    validation_accuracy: float


class TransformerTrainer:
    def __init__(self, args: TransformerTrainingArgs, model, experiment_config: ExperimentConfig, dataset_dict, device):
        super().__init__()
        self.model = model
        self.args = args
        self.optimizer = t.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.step = 0
        self.training_loss = []
        self.validation_accuracy = []
        self.experiment_config = experiment_config
        self.metrics: list[TrainingMetrics] = []
        self.dataset_dict = dataset_dict
        self.device = device
    def training_step(self, batch: dict[str, Int[Tensor, "batch seq"]]) -> Float[Tensor, ""]:
        '''
        Calculates the loss on the tokens in the batch, performs a gradient update step, and logs the loss.

        Remember that `batch` is a dictionary with the single key 'tokens'.
        '''
        # YOUR CODE HERE
        self.optimizer.zero_grad()
        tokens = batch['tokens'].to(self.device)
        logits = self.model(tokens)
        log_probs = get_log_probs(logits, tokens)
        loss = -log_probs.mean()
        loss.backward()
        self.optimizer.step()
        self.step += 1
        return loss


    def validation_step(self, batch: dict[str, Int[Tensor, "batch seq"]]):
        '''
        Calculates & returns the accuracy on the tokens in the batch (i.e. how often the model's prediction
        is correct). Logging should happen in the `train` function (after we've computed the accuracy for 
        the whole validation set).
        '''
        # YOUR CODE HERE
        tokens = batch['tokens'].to(self.device)
        logits = self.model(tokens)[:, :-1]
        accuracy = (logits.argmax(dim=-1) == tokens[:, 1:]).flatten()
        return accuracy
        


    def train(self):
        '''
        Trains the model, for `self.args.epochs` epochs. Also handles wandb initialisation, and early stopping
        for each epoch at `self.args.max_steps_per_epoch` steps.
        '''
        # YOUR CODE HERE
        progress_bar = tqdm(total = self.args.max_steps_per_epoch * self.args.epochs)

        for epoch in range(self.args.epochs):
            for i, batch in enumerate(self.train_loader()):
                loss = self.training_step(batch)
                progress_bar.update()
                
                if i >= self.args.max_steps_per_epoch:
                    break

                accuracy = t.cat([self.validation_step(batch) for batch in self.test_loader()])
                mean_accuracy = accuracy.float().mean().item()
                self.metrics.append(TrainingMetrics(
                    step=self.step,
                    epoch=epoch,
                    training_loss=loss.item(),
                    validation_accuracy=mean_accuracy
                ))
                progress_bar.set_description(f"Epoch {epoch+1} - Step {i+1} - Loss {loss:.4f} - Accuracy {mean_accuracy:.4f}")
                self.training_loss.append(loss.item())
                self.validation_accuracy.append(mean_accuracy)

    def train_loader(self) -> DataLoader:
        '''Returns train loader (as in code above).'''
        return DataLoader(self.dataset_dict["train"], batch_size=self.args.batch_size, shuffle=True, num_workers=16, pin_memory=True)


    def test_loader(self) -> DataLoader:
        '''Returns test loader (as in code above).'''
        return DataLoader(self.dataset_dict["test"], batch_size=self.args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

def train_both_models(experiment_config: ExperimentConfig, tokenizer, dataset_dict, device):

    base_cfg = experiment_config.get_config_base(tokenizer)
    diff_cfg = experiment_config.get_config_diff(tokenizer)
    training_args = experiment_config.get_training_args()
    diff_model = DiffTransformer(diff_cfg).to(device)
    diff_trainer = TransformerTrainer(training_args, diff_model, experiment_config, dataset_dict, device)
    diff_trainer.train()

    base_model = Transformer(base_cfg).to(device)
    trainer = TransformerTrainer(training_args, base_model, experiment_config, dataset_dict, device)
    trainer.train()

    return diff_trainer.metrics, trainer.metrics

def save_metrics(base_metrics: list[TrainingMetrics], diff_metrics: list[TrainingMetrics], experiment_config: ExperimentConfig):
    serializable_result = {
        "config": vars(experiment_config),
        "base_metrics": [vars(metric) for metric in base_metrics],
        "diff_metrics": [vars(metric) for metric in diff_metrics]
    }
    
    for i in range(100):
        results_path = f'results/results_{i}'
        if not os.path.exists(results_path):
            with open(results_path, 'w') as f:
                json.dump(serializable_result, f)
            return
#%%
if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = t.device('mps' if t.backends.mps.is_available() else 'cuda' if t.cuda.is_available() else 'cpu')
    print(device)
    #%%
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    #%%
    experiment_config = ExperimentConfig(
        n_layers = 2,
        d_model = 16,
        d_head = 8,
        n_heads = 4,
        n_ctx = 256,
        batch_size = 16,
        epochs = 10,
        max_steps_per_epoch = 3,
        lr = 1e-3,
        weight_decay = 1e-2
    )
    base_cfg = experiment_config.get_config_base(tokenizer)
    diff_cfg = experiment_config.get_config_diff(tokenizer)
    #%%
    dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train").remove_columns("meta")
    tokenized_dataset = tokenize_and_concatenate(dataset, tokenizer, device, max_length = base_cfg.n_ctx, column_name="text", add_bos_token=True, num_proc=4)
    dataset_dict = tokenized_dataset.train_test_split(test_size=1000)
    diff_metrics, base_metrics = train_both_models(experiment_config, tokenizer, dataset_dict, device)
    save_metrics(base_metrics, diff_metrics, experiment_config)
    
    
    #%%
    with open('results/results_0', 'r') as f:
        result = json.load(f)
    # %%
    result.keys()
