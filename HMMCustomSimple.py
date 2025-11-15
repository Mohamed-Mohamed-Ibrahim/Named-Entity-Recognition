from HMM import HMMCustom
import numpy as np

# NOTE: The LabelEncoder, load_from_disk, and CONLL-2003 specific parts are REMOVED
# because they conflict with the explicit HMM parameters you provided earlier.

# --- HMM PARAMETERS FOR SENTIMENT TAGGING PROBLEM ---

# 1. States (Tags: +, -, O)
TAGS = ['+', '-', 'O']
tag_to_index = {tag: i for i, tag in enumerate(TAGS)}
N_COMPONENTS = len(TAGS) # 3

# 2. Observations (Words from Table 1)
OBSERVATIONS = [
    'and', 'awful', 'bitter', 'bread', 'coffee', 'delicious', 'smells', 'the', 'was'
]
word_to_index = {word: i for i, word in enumerate(OBSERVATIONS)}
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
# Rows: Tag (+, -, O), Columns: Word (and, awful, bitter, bread, coffee, delicious, smells, the, was)
emissionprob = np.array([
    [0.0, 0.0, 0.1, 0.05, 0.05, 0.7, 0.05, 0.0, 0.05], # +
    [0.0, 0.7, 0.15, 0.0, 0.05, 0.0, 0.05, 0.0, 0.05], # -
    [0.2, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05, 0.2, 0.2]  # O
])

# --- HMM INITIALIZATION ---

# Create the HMM model using the provided parameters
model = HMMCustom(
    n_components=N_COMPONENTS,
    n_observations=N_OBSERVATIONS,
    startprob=startprob,
    transmat=transmat,
    emissionprob=emissionprob,
    # strategy="greedy"
)

# NOTE: Since all parameters are explicitly defined, model.fit(X, nerTags) is not needed
# and is skipped, as the parameters are set in the constructor.

# --- DECODING FOR {coffee, smells, bitter} (Problem b) ---

# Map observations to numerical indices (4=coffee, 6=smells, 2=bitter)
observation_sequence = np.array([
    word_to_index['coffee'],
    word_to_index['smells'],
    word_to_index['bitter']
])
# Convert indices back to words for printing
ture_observation_sequence = [OBSERVATIONS[i] for i in observation_sequence]


# Run Viterbi decoding (finds the MOST likely sequence)
# If your HMMCustom class has the decode method added from the previous response:
log_likelihood, hidden_states = model.decode(observation_sequence)

print("Observation sequence:", ture_observation_sequence)
print("Log-likelihood of observations (MOST LIKELY PATH):", log_likelihood)
print("Most likely hidden states sequence:", [TAGS[s] for s in hidden_states])
# The most likely sequence for {coffee, smells, bitter} is: (O, O, O)
# The probability calculation (not required by the prompt, but useful for testing):
# P(O) * P(O|O) * P(O|O) * P(coffee|O) * P(smells|O) * P(bitter|O)
# = 0.4 * 0.6 * 0.6 * 0.1 * 0.05 * 0.05 = 0.000036
# log(0.000036) approx -10.2319