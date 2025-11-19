from datasets import load_dataset, load_from_disk
import numpy as np
from sklearn.preprocessing import LabelEncoder

import numpy as np


class HMMCustom:
    def __init__(self, n_components, n_observations, strategy="viterbi"):
        self.n_components_ = n_components
        self.n_observations_ = n_observations
        self.strategy = strategy

        # Initialize with small epsilon to avoid log(0) later
        self.startprob_ = np.zeros(n_components)
        self.transmat_ = np.zeros((n_components, n_components))
        self.emissionprob_ = np.zeros((n_components, n_observations))

    def fit(self, X, y):
        # X: List of lists (sentences of observations)
        # y: List of lists (corresponding hidden states)

        # 1. Count occurrences
        for idx, sentence in enumerate(X):
            for i, word in enumerate(sentence):
                state = y[idx][i]

                # Emission: State -> Word
                self.emissionprob_[state, word] += 1

                if i == 0:
                    # Start Probability
                    self.startprob_[state] += 1
                else:
                    # Transition: Previous State -> Current State
                    prev_state = y[idx][i - 1]
                    self.transmat_[prev_state, state] += 1

                    # 2. Add Laplace Smoothing (add 1) to avoid zero division
        self.startprob_ += 1
        self.transmat_ += 1
        self.emissionprob_ += 1

        # 3. Normalize (probabilities sum to 1)
        self.startprob_ /= np.sum(self.startprob_)

        # Normalize rows (axis 1) so sum of outgoing probs = 1
        self.transmat_ /= np.sum(self.transmat_, axis=1, keepdims=True)
        self.emissionprob_ /= np.sum(self.emissionprob_, axis=1, keepdims=True)

        # 4. CONVERT TO LOG SPACE
        # We store logs so we can add them instead of multiplying
        with np.errstate(divide='ignore'):
            self.startprob_ = np.log(self.startprob_)
            self.transmat_ = np.log(self.transmat_)
            self.emissionprob_ = np.log(self.emissionprob_)

    def _viterbi(self, X):
        n_steps = len(X)
        if n_steps == 0:
            return -np.inf, []

        # m stores the max log-probability reaching state s at time t
        m = np.zeros((self.n_components_, n_steps))
        # parent stores the best previous state
        parent = np.zeros((self.n_components_, n_steps), dtype=int)

        # --- Initialization (Step 0) ---
        for s in range(self.n_components_):
            # log(start) + log(emission)
            m[s, 0] = self.startprob_[s] + self.emissionprob_[s, X[0]]

        # --- Recursion (Forward Step) ---
        for t in range(1, n_steps):
            for s in range(self.n_components_):  # Current state

                # Calculate transition from all previous states (s_prev) to current state (s)
                # vector operation: m[:, t-1] is all prev path probs
                # transmat_[:, s] is prob of moving from any prev -> s
                probs = m[:, t - 1] + self.transmat_[:, s] + self.emissionprob_[s, X[t]]

                # Find max probability and the state that produced it
                parent[s, t] = np.argmax(probs)
                m[s, t] = np.max(probs)

        # --- Termination ---
        best_path_log_prob = np.max(m[:, -1])
        last_state = np.argmax(m[:, -1])

        # --- Backtracking (Backward Step) ---
        best_path = [last_state]

        # Loop backwards from last step down to 1
        for t in range(n_steps - 1, 0, -1):
            prev_state = parent[best_path[-1], t]
            best_path.append(prev_state)

        # Reverse to get correct order
        return best_path_log_prob, list(reversed(best_path))

    def decode(self, X):
        if self.strategy == "viterbi":
            return self._viterbi(X)
        # Implement greedy similarly using log space if needed
        return None

if __name__ == '__main__':
    # dataset = load_dataset("lhoestq/conll2003")
    # dataset.save_to_disk("conll2003")
    # ---------------------------------------------------------

    n_sampels = 150
    random_state = 42

    dataset = load_from_disk("conll2003")
    # nerTags = dataset["validation"][100:n_sampels]['ner_tags']
    nerTags = dataset["validation"][:]['ner_tags']
    # tokens = dataset["validation"][100:n_sampels]['tokens']
    tokens = dataset["validation"][:]['tokens']
    states = ["Other", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    observations = set()
    maxLen = 0
    observations.add(" ")
    for token in tokens:
        maxLen = max(maxLen, len(token))
        for word in token:
            observations.add(word.lower())
    observations = list(sorted(observations))
    # print("Observations:", observations)

    le = LabelEncoder()
    encoded_data = le.fit_transform(observations)
    # print("Encoded data:", encoded_data)

    X = []
    for token in tokens:
        encoded_tokens = le.transform([word.lower() for word in token])
        X.append(encoded_tokens.tolist())

    # print(X)
    # print(tokens)
    print(nerTags)

    n_components = len(states)
    n_observations = len(observations)
    # print(n_components)
    # print(n_observations)

    model = HMMCustom(n_components=n_components, n_observations=n_observations)

    model.fit(X, nerTags)

    # Example: Dry, Wet, Dry
    # Map observations to numerical indices (0 for Dry, 1 for Wet)
    # observation_sequence = np.array([[0], [1], [0], [1], [0]])
    observation_sequence = np.array([150, 211, 111, 10])
    ture_observation_sequence = le.inverse_transform(observation_sequence)

    log_likelihood, hidden_states = model.decode(observation_sequence)
    print("Log-likelihood of observations:", log_likelihood)
    print("Most likely hidden states sequence:", [states[s] for s in hidden_states])
    print("Observation sequence:", ture_observation_sequence)