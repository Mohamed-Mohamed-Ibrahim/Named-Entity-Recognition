from HMM import HMMCustom
import numpy as np

import numpy as np
from sklearn.preprocessing import LabelEncoder
# Assume HMMCustom class is defined elsewhere and handles fit/decode.

# --- HMM PARAMETERS FOR SENTIMENT TAGGING PROBLEM ---

# 1. States (Tags: +, -, O)
TAGS = ['+', '-', 'O']
tag_to_index = {tag: i for i, tag in enumerate(TAGS)}
N_COMPONENTS = len(TAGS) # 3

# 2. Observations (Words from Table 1)
# IMPORTANT: The order of this list MUST match the column order in Table 1
OBSERVATIONS = [
    'and', 'awful', 'bitter', 'bread', 'coffee', 'delicious', 'smells', 'the', 'was'
]

# Use LabelEncoder to map words to indices (0 through 8)
le = LabelEncoder()
# Fit transforms the observations based on the order of the list
le.fit(OBSERVATIONS)

# Verify the mapping (optional, but good for debugging)
# print(f"coffee index: {le.transform(['coffee'])[0]}") # Should be 4
N_OBSERVATIONS = len(OBSERVATIONS) # 9

# Initial Probabilities (Table 2, row '∅')
startprob = np.array([0.3, 0.3, 0.4]) # [P(+), P(-), P(O)]

# Transition Probabilities (Table 2)
# Rows: From (+, -, O), Columns: To (+, -, O)
transmat = np.array([
    [0.4, 0.2, 0.4],  # From +
    [0.1, 0.5, 0.4],  # From -
    [0.2, 0.2, 0.6]   # From O
])

# Emission Probabilities (Table 1)
# Rows: Tag (+, -, O), Columns: Word (order matches OBSERVATIONS list)
emissionprob = np.array([
    # and awful bitter bread coffee delicious smells the was
    [0.0, 0.0, 0.1, 0.05, 0.05, 0.7, 0.05, 0.0, 0.05], # +
    [0.0, 0.7, 0.15, 0.0, 0.05, 0.0, 0.05, 0.0, 0.05], # -
    [0.2, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05, 0.2, 0.2]  # O
])

# --- HMM INITIALIZATION ---

model = HMMCustom(
    n_components=N_COMPONENTS,
    n_observations=N_OBSERVATIONS,
    startprob=startprob,
    transmat=transmat,
    emissionprob=emissionprob
)

# --- DECODING FOR {coffee, smells, bitter} (Problem b) ---

# Use the fitted LabelEncoder to transform the word sequence
word_sequence_b = ['coffee', 'bitter', 'smells', 'bitter']
word_sequence_b = ['the', 'bread', 'smells', 'delicious']
observation_sequence = le.transform(word_sequence_b)

# The resulting array is: [4, 6, 2] (matching the previous manual mapping)

# Convert indices back to words for printing (using inverse_transform)
ture_observation_sequence = le.inverse_transform(observation_sequence)


# Run Viterbi decoding (finds the MOST likely sequence)
log_likelihood, hidden_states = model.decode(observation_sequence)

print("\n--- HMM Problem (b) Test with LabelEncoder ---")
print("Observation sequence:", ture_observation_sequence)
print("Log-likelihood of observations (MOST LIKELY PATH):", log_likelihood)
print("Most likely hidden states sequence:", [TAGS[s] for s in hidden_states])