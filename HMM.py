from datasets import load_dataset, load_from_disk
import numpy as np
from sklearn.preprocessing import LabelEncoder


class HMMCustom:
    def __init__(self, n_components, n_observations, startprob=None, transmat=None, emissionprob=None):

        self.n_components_ = n_components
        self.n_observations_ = n_observations

        if startprob is None:
            self.startprob_ = np.zeros(n_components)
        else:
            self.startprob_ = startprob
        if transmat is None:
            self.transmat_ = np.zeros((n_components, n_components))
        else:
            self.transmat_ = transmat
        if emissionprob is None:
            self.emissionprob_ = np.zeros((n_components, n_observations))
        else:
            self.emissionprob_ = emissionprob

    def fit(self, X, y):

        # Get start & transition & emission probs
        for idx, sentence in enumerate(X):
            for i, word in enumerate(sentence):

                # Get start prob
                if i == 0:
                    self.startprob_[y[idx][i]] += 1
                # Get transition prob
                else:
                    self.transmat_[y[idx][i], y[idx][i-1]] += 1

                # Get emission prob
                self.emissionprob_[y[idx][i], word] += 1


    def decode(self, X):

        log_likelihood, hidden_states = 0, []
        prev_state = None

        # Greedy for now
        for i, word in enumerate(X):
            log_score = 0
            if i == 0:
                for state in range(self.n_components_):
                    res = self.startprob_[state]*self.emissionprob_[state][word]
                    log_score_idx = np.argmax(res)
                    log_score += np.log(res[log_score_idx])
                    hidden_states.append(state)
                    prev_state = state
            else:
                for state in range(self.n_components_):
                    res = self.startprob_[state][prev_state]*self.emissionprob_[state][word]
                    log_score_idx = np.argmax(res)
                    log_score += np.log(res[log_score_idx])
                    hidden_states.append(state)
                    prev_state = state

            log_likelihood += log_score
        return log_likelihood, hidden_states


if __name__ == '__main__':
    # dataset = load_dataset("lhoestq/conll2003")
    # dataset.save_to_disk("conll2003")
    # ---------------------------------------------------------

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
        X.append(np.pad(encoded_tokens, (0, maxLen - len(encoded_tokens)), mode='constant', constant_values=0).tolist())

    # print(X)
    # print(tokens)

    n_components = len(states)
    n_observations = len(observations)
    # print(n_components)
    # print(n_observations)

    model = HMMCustom(n_components=n_components, n_observations=n_observations)

    model.fit(X, nerTags)

    # Example: Dry, Wet, Dry
    # Map observations to numerical indices (0 for Dry, 1 for Wet)
    observation_sequence = np.array([[0], [1], [0], [1], [0]])

    log_likelihood, hidden_states = model.decode(observation_sequence)
    print("Log-likelihood of observations:", log_likelihood)
    print("Most likely hidden states sequence:", [states[s] for s in hidden_states])