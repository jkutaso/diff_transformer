#%%
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os
from train_model import ExperimentConfig
from transformers import GPT2TokenizerFast
from transformer import Transformer
from diff_transformer import DiffTransformer
#%%

# %%
def get_results_df(results):
    base_metrics_df = pd.DataFrame(results['base_metrics'])
    diff_metrics_df = pd.DataFrame(results['diff_metrics'])
    combined_df = pd.merge(base_metrics_df, diff_metrics_df, on=['step', 'epoch'], suffixes=('_base', '_diff'))
    return combined_df

#%%
# Define exponential decay function
def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

def fit_exp_decay(combined_df, column_name):
    # Fit exponential decay to training loss
    x_data = combined_df.step.values
    y_data = combined_df[column_name].values
    popt, pcov = curve_fit(exp_decay, x_data, y_data)

    # Generate fitted curve
    x_fit = np.linspace(min(x_data), max(x_data), 100)
    y_fit = exp_decay(x_fit, *popt)

    return x_fit, y_fit

#%%


#%%
# Calculate total parameters in diff_model
def count_params(config, tokenizer):
    experiment_config = ExperimentConfig(**config)
    base_cfg = experiment_config.get_config_base(tokenizer)
    diff_cfg = experiment_config.get_config_diff(tokenizer)
    diff_model = DiffTransformer(diff_cfg)
    base_model = Transformer(base_cfg)
    total_params_diff = sum(p.numel() for p in diff_model.parameters())
    total_params_base = sum(p.numel() for p in base_model.parameters())
    # Break down parameters by section for diff model
    attention_params_diff = sum(p.numel() for name, p in diff_model.named_parameters() if 'attn' in name)

    # Break down parameters by section for base model
    attention_params_base = sum(p.numel() for name, p in base_model.named_parameters() if 'attn' in name)
    del diff_model, base_model
    return total_params_diff, attention_params_diff, total_params_base, attention_params_base

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
# Load all results files and store combined dataframes
all_results = {}
configs = {}
keys = []
for filename in os.listdir('results'):
    if filename.startswith('results_'):
        with open(f'results/{filename}', 'r') as f:
            results = json.load(f)
            all_results[int(filename.split('_')[1])] = get_results_df(results)
            configs[int(filename.split('_')[1])] = results['config']
            keys.append(int(filename.split('_')[1]))
all_results.keys()
#%%
per_model_data = pd.DataFrame(columns = ['key', 'total_params_diff', 'attention_params_diff', 'total_params_base', 'attention_params_base', 'final_training_loss_base', 'final_training_loss_diff', 'final_validation_accuracy_base', 'final_validation_accuracy_diff'])

for key in keys[1:]:
    print(key)
    total_params_diff, attention_params_diff, total_params_base, attention_params_base = count_params(configs[key], tokenizer)
    combined_df = all_results[key]
    final_training_loss_base = combined_df.training_loss_base.iloc[-1]
    final_training_loss_diff = combined_df.training_loss_diff.iloc[-1]
    final_validation_accuracy_base = combined_df.validation_accuracy_base.iloc[-1]
    final_validation_accuracy_diff = combined_df.validation_accuracy_diff.iloc[-1]
    new_row = pd.DataFrame({
        'key': [key],
        'total_params_diff': [total_params_diff],
        'attention_params_diff': [attention_params_diff], 
        'total_params_base': [total_params_base],
        'attention_params_base': [attention_params_base],
        'final_training_loss_base': [final_training_loss_base],
        'final_training_loss_diff': [final_training_loss_diff],
        'final_validation_accuracy_base': [final_validation_accuracy_base],
        'final_validation_accuracy_diff': [final_validation_accuracy_diff]
    })
    per_model_data = pd.concat([per_model_data, new_row], ignore_index=True)

#%%
def get_linear_regression(x, y):
    """
    Fits a linear regression to x and y data using normal equations.
    Returns a tuple of (slope, intercept).
    """
    # Reshape x to 2D array if it's 1D
    x = x.values.reshape(-1) if hasattr(x, 'values') else x.reshape(-1)
    
    # Calculate means
    x_mean = x.mean()
    y_mean = y.mean()
    
    # Calculate slope and intercept using normal equations
    numerator = ((x - x_mean) * (y - y_mean)).sum()
    denominator = ((x - x_mean) ** 2).sum()
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    return slope, intercept

slope_diff, intercept_diff = get_linear_regression(per_model_data.total_params_diff, per_model_data.final_validation_accuracy_diff)
slope_base, intercept_base = get_linear_regression(per_model_data.total_params_base, per_model_data.final_validation_accuracy_base)
#%%

plt.scatter(per_model_data.total_params_diff, per_model_data.final_validation_accuracy_diff, label='Diff')
plt.scatter(per_model_data.total_params_base, per_model_data.final_validation_accuracy_base, label='Base')
plt.plot(per_model_data.total_params_diff, slope_diff * per_model_data.total_params_diff + intercept_diff, 'r--', label='Diff')
plt.plot(per_model_data.total_params_base, slope_base * per_model_data.total_params_base + intercept_base, 'g--', label='Base')
plt.legend()
plt.show()


#%%
all_results[0]
#%%
with open('results/results_15', 'r') as f:
    example_results = json.load(f)

example_results.keys()
combined_df = get_results_df(example_results)
combined_df.head()
x_fit_base, y_fit_base = fit_exp_decay(combined_df, 'training_loss_base')
x_fit_diff, y_fit_diff = fit_exp_decay(combined_df, 'training_loss_diff')

#plt.plot(combined_df.step, combined_df.training_loss_base, label='Base')
plt.plot(x_fit_base, y_fit_base, 'r--', label='Base fit')
#plt.plot(combined_df.step, combined_df.training_loss_diff, label='Diff')
plt.plot(x_fit_diff, y_fit_diff, 'g--', label='Diff fit')
plt.legend()
plt.show()
# %%
combined_df.groupby('epoch').mean()
# %%
combined_df