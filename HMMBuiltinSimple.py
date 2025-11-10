from datasets import load_dataset, load_from_disk
from hmmlearn import hmm
import numpy as np

states = ["Sunny", "Rainy"]
observations = ["Dry", "Wet"]
n_components = len(states)
n_observations = len(observations)

# Initial probabilities
startprob = np.array([0.6, 0.4])  # 60% chance of starting Sunny, 40% Rainy

# Transition probabilities
transmat = np.array([[0.7, 0.3],  # Sunny -> Sunny (0.7), Sunny -> Rainy (0.3)
                     [0.4, 0.6]])  # Rainy -> Sunny (0.4), Rainy -> Rainy (0.6)

# Emission probabilities
emissionprob = np.array([[0.9, 0.1],  # Sunny -> Dry (0.9), Sunny -> Wet (0.1)
                         [0.2, 0.8]])  # Rainy -> Dry (0.2), Rainy -> Wet (0.8)

model = hmm.CategoricalHMM(n_components=n_components, random_state=42)
model.startprob_ = startprob
model.transmat_ = transmat
model.emissionprob_ = emissionprob

# Example: Dry, Wet, Dry
# Map observations to numerical indices (0 for Dry, 1 for Wet)
observation_sequence = np.array([[0], [1], [0]])

log_likelihood, hidden_states = model.decode(observation_sequence)
print("Log-likelihood of observations:", log_likelihood)
print("Most likely hidden states sequence:", [states[s] for s in hidden_states])

X, Z = model.sample(n_samples=5)
print("Generated observations:", [observations[x[0]] for x in X])
print("Generated hidden states:", [states[z] for z in Z])

if __name__ == '__main__':
    # dataset = load_dataset("lhoestq/conll2003")
    # dataset.save_to_disk("conll2003")
    # ---------------------------------------------------------
    pass