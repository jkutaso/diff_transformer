
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
import einops
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = t.device('mps' if t.backends.mps.is_available() else 'cuda' if t.cuda.is_available() else 'cpu')
print(device)
#%%
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer
#%%

base_cfg = Config(
    debug=False, 
    d_model=256, 
    n_heads=4, 
    d_head=64, 
    d_mlp=1024, 
    n_layers=2, 
    n_ctx=256, 
    d_vocab=tokenizer.vocab_size
)
diff_cfg = base_cfg.copy()
diff_cfg.d_head = base_cfg.d_head // 2
diff_cfg
#%%
dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train").remove_columns("meta")
print(dataset)
print(dataset[0]['text'][:100])
#%%
tokenized_dataset = tokenize_and_concatenate(dataset, tokenizer, device, max_length = base_cfg.n_ctx, column_name="text", add_bos_token=True, num_proc=4)
print(tokenized_dataset)
#%%
@dataclass
class TransformerTrainingArgs():
    batch_size = 16
    epochs = 10
    max_steps_per_epoch = 100
    lr = 1e-3
    weight_decay = 1e-2

args = TransformerTrainingArgs()

dataset_dict = tokenized_dataset.train_test_split(test_size=1000)
train_loader = DataLoader(dataset_dict["train"], batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
test_loader = DataLoader(dataset_dict["test"], batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)
# %%
first_batch = train_loader.dataset[:args.batch_size]

print(first_batch.keys())
print(first_batch['tokens'].shape)
#%%
def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], 
    tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:

    log_probs = logits.log_softmax(dim=-1)
    # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    log_probs_for_tokens = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

    return log_probs_for_tokens

class TransformerTrainer:
    def __init__(self, args: TransformerTrainingArgs, model):
        super().__init__()
        self.model = model
        self.args = args
        self.optimizer = t.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.step = 0
        self.training_loss = []
        self.validation_accuracy = []


    def training_step(self, batch: dict[str, Int[Tensor, "batch seq"]]) -> Float[Tensor, ""]:
        '''
        Calculates the loss on the tokens in the batch, performs a gradient update step, and logs the loss.

        Remember that `batch` is a dictionary with the single key 'tokens'.
        '''
        # YOUR CODE HERE
        self.optimizer.zero_grad()
        tokens = batch['tokens'].to(device)
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
        tokens = batch['tokens'].to(device)
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
                progress_bar.set_description(f"Epoch {epoch+1} - Step {i+1} - Loss {loss:.4f} - Accuracy {mean_accuracy:.4f}")
                self.training_loss.append(loss.item())
                self.validation_accuracy.append(mean_accuracy)

    def train_loader(self) -> DataLoader:
        '''Returns train loader (as in code above).'''
        return DataLoader(dataset_dict["train"], batch_size=self.args.batch_size, shuffle=True, num_workers=16, pin_memory=True)


    def test_loader(self) -> DataLoader:
        '''Returns test loader (as in code above).'''
        return DataLoader(dataset_dict["test"], batch_size=self.args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

# %%
diff_model = DiffTransformer(diff_cfg).to(device)
diff_trainer = TransformerTrainer(args, diff_model)
diff_trainer.train()
# %%
base_model = Transformer(base_cfg).to(device)
trainer = TransformerTrainer(args, base_model)
trainer.train()

# %%
len(trainer.validation_accuracy)
# %%
results_df = pd.DataFrame({'base_training_loss': trainer.training_loss, 'base_validation_accuracy': trainer.validation_accuracy, 'diff_training_loss': diff_trainer.training_loss, 'diff_validation_accuracy': diff_trainer.validation_accuracy})
results_df.to_csv(f'results/example_results.csv')
# %%


plt.figure(figsize=(12,8))

plt.subplot(2,1,1)
plt.plot(results_df['base_training_loss'], label='Base Model')
plt.plot(results_df['diff_training_loss'], label='Diff Model') 
plt.title('Training Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2,1,2)
plt.plot(results_df['base_validation_accuracy'], label='Base Model')
plt.plot(results_df['diff_validation_accuracy'], label='Diff Model')
plt.title('Validation Accuracy') 
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
