from datasets import load_dataset, load_from_disk
from hmmlearn import hmm
import numpy as np
from pandas.core.common import random_state
from sklearn.preprocessing import LabelEncoder

dataset = load_from_disk("conll2003")
nerTags = dataset["validation"][:3]['ner_tags']
tokens = dataset["validation"][:3]['tokens']
states = ["Other", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
observations = set()
maxLen = 0
observations.add(" ")
for token in tokens:
    maxLen = max(maxLen, len(token))
    for word in token:
        observations.add(word)
observations = list(sorted(observations))
# print("Observations:", observations)

le = LabelEncoder()
encoded_data = le.fit_transform(observations)
# print("Encoded data:", encoded_data)

X = []
for token in tokens:
    encoded_tokens = le.transform(token)
    X.append(np.pad(encoded_tokens, (0, maxLen-len(encoded_tokens)), mode='constant', constant_values=0).tolist())

print(X)
print(tokens)
print(nerTags)

n_components = len(states)
n_observations = len(observations)
# print(n_components)
# print(n_observations)

model = hmm.CategoricalHMM(n_components=n_components, random_state=42)

model.fit(X)

# Example: Dry, Wet, Dry
# Map observations to numerical indices (0 for Dry, 1 for Wet)
# observation_sequence = np.array([[0], [1], [0], [1], [0]])
observation_sequence = np.array([[2], [2], [2], [4]])

log_likelihood, hidden_states = model.decode(observation_sequence)
print("Log-likelihood of observations:", log_likelihood)
print("Most likely hidden states sequence:", [states[s] for s in hidden_states])



if __name__ == '__main__':
    # dataset = load_dataset("lhoestq/conll2003")
    # dataset.save_to_disk("conll2003")
    # ---------------------------------------------------------
    pass
