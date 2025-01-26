from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from mingpt.model import GPT
from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer

set_seed(3407)

use_mingpt = True
model_type = 'gpt2'
device = 'cpu'

tokenizer = BPETokenizer()
model = GPT.from_pretrained(model_type, save_activations_enabled=True, patch_embeddings_enabled=False)

# ship model to device and set to eval mode
model.to(device)
model.eval()


@dataclass
class Experiment:
    clean_text: str
    corrupted_text: str
    token_1: str
    token_2: str


def generate(prompt: str, num_samples=1, steps=1, layer_to_patch=None, embedding_to_patch=None):
    # tokenize the input prompt into integer input sequence
    x = tokenizer(prompt).to(device)

    tokens = [tokenizer.decode(torch.tensor([token])) for token in x[0]]

    # we'll process all desired num_samples in a batch, so expand out the batch dim
    x = x.expand(num_samples, -1)

    # forward the model `steps` times to get samples, in a batch
    y = model.generate(x, max_new_tokens=steps, layer_to_patch=layer_to_patch, embedding_to_patch=embedding_to_patch)

    return y, tokens, model.last_token_logits


def build_heatmap(input_length: int, test: Experiment):
    number_of_layers = 12

    heatmap = np.zeros((number_of_layers, input_length))
    model.save_activations_enabled = False
    model.patch_embeddings_enabled = True

    for i in range(number_of_layers):
        for j in range(input_length):

            # Run model over the corrupted text
            # and inject the activations from the clean run
            y, tokens, last_token_logits = generate(prompt=test.corrupted_text, layer_to_patch=i, embedding_to_patch=j)

            index_token_1 = tokenizer(test.token_1).item()
            index_token_2 = tokenizer(test.token_2).item()

            logit_of_token_1 = last_token_logits[0][index_token_1]
            logit_of_token_2 = last_token_logits[0][index_token_2]

            # Compute the difference between the logits of the two tokens
            heatmap[i][j] = logit_of_token_1 - logit_of_token_2

    df = pd.DataFrame(
        heatmap,
        index=[f'layer {i}' for i in range(heatmap.shape[0])],
        columns=[f'(embed {i}) {tokens[i]}' for i in range(heatmap.shape[1])]
    )

    cmap = sns.color_palette("Blues", as_cmap=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=False, fmt="f", cmap=cmap, cbar_kws={'label': 'Magnitude'})

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    experiments = [
        Experiment(
            clean_text="Michelle Jones was a top-notch student. Michelle",
            corrupted_text="Michelle Smith was a top-notch student. Michelle",
            token_1=" Jones",
            token_2=" Smith"
        ),
        Experiment(
            clean_text="When John and Mary went to the store, John gave the bag to",
            corrupted_text="When John and Mary went to the store, Mary gave the bag to",
            token_1=" John",
            token_2=" Mary"
        ),
        Experiment(
            clean_text="Jessica Jones was a top-notch student. Michelle",
            corrupted_text="Michelle Smith was a top-notch student. Jessica",
            token_1=" Jones",
            token_2=" Smith"
        ),
        Experiment(
            clean_text=" Elon Musk co-founded Tesla and revolutionized the electric car industry. Elon",
            corrupted_text="Steve Jobs co-founded Tesla and revolutionized the electric car industry. Elon",
            token_1=" Musk",
            token_2=" Jobs"
        ),
        Experiment(
            clean_text="a b c",
            corrupted_text="d e f",
            token_1=" a",
            token_2=" d"
        ),
    ]

    number_of_layers = 12
    for experiment in experiments:
        input_length = tokenizer(experiment.clean_text).shape[-1]

        # Run model over the clean text to save the activations
        y1, tokens1, last_token_logits1 = generate(prompt=experiment.clean_text)

        index = tokenizer(experiment.token_1).item()
        logit_of_jones = last_token_logits1[0][index]

        probabilities = torch.softmax(last_token_logits1.squeeze(), dim=-1)

        top_probs, top_indices = torch.topk(probabilities, k=10)
        top_tokens = [(tokenizer.decode(torch.tensor([idx])), prob.item()) for idx, prob in zip(top_indices, top_probs)]

        for token, prob in top_tokens:
            print(f"Token: '{token}', Probability: {prob:.4f}")

        build_heatmap(input_length, experiment)
