from datasets import load_dataset, load_from_disk
import numpy as np
from sklearn.preprocessing import LabelEncoder


class HMMCustom:
    def __init__(self, n_components, n_observations, startprob=None, transmat=None, emissionprob=None,
                 strategy="viterbi"):

        self.n_components_ = n_components
        self.n_observations_ = n_observations
        self.strategy = strategy

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
                    self.transmat_[y[idx][i], y[idx][i - 1]] += 1

                # Get emission prob
                self.emissionprob_[y[idx][i], word] += 1

        # Get start & transition & emission probs
        for i in range(self.n_components_):
            self.startprob_[i] += 1
            for j in range(self.n_components_):
                self.transmat_[i, j] += 1
            for j in range(self.n_observations_):
                self.emissionprob_[i, j] += 1

        with np.errstate(divide='ignore', invalid='ignore'):
            self.startprob_ /= np.sum(self.startprob_)
            self.emissionprob_ /= (np.sum(self.emissionprob_, axis=1).reshape(-1, 1))
            # self.emissionprob_ /= ( np.sum(self.transmat_, axis=1).reshape(-1, 1) + self.startprob_.reshape(-1, 1) )
            self.transmat_ /= np.sum(self.transmat_, axis=1).reshape(-1, 1)
            # self.transmat_ /= np.sum(self.transmat_, axis=1)[:, np.newaxis]
        self.transmat_ = np.nan_to_num(self.transmat_)
        self.emissionprob_ = np.nan_to_num(self.emissionprob_)

        # print(self.startprob_)
        # print()
        # for x in self.transmat_:
        #     print(x)
        # print()
        # for x in self.emissionprob_:
        #     print(x)
        # print()

    def _greedy(self, X):
        log_likelihood, hidden_states = 0, []
        prev_state = None

        for i, word in enumerate(X):
            score = -1
            if i == 0:
                for state in range(self.n_components_):
                    prob = self.startprob_[state] * self.emissionprob_[state][word]
                    if prob > score:
                        score = prob
                        prev_state = state
            else:
                for state in range(self.n_components_):
                    prob = self.transmat_[state][prev_state] * self.emissionprob_[state][word]
                    if prob > score:
                        score = prob
                        prev_state = state

            hidden_states.append(prev_state)

            log_likelihood += score
            # print(score, prev_state)

        return log_likelihood, hidden_states

    def _viterbi(self, X):
        log_likelihood, hidden_states = 0, []

        n_steps = len(X)

        if n_steps == 0:
            return log_likelihood, hidden_states

        m = np.zeros((self.n_components_, n_steps))
        parent = np.ones((self.n_components_, n_steps), dtype=int)

        for i, word in enumerate(X):
            if i == 0:
                for state in range(self.n_components_):
                    # print(self.startprob_[state], self.emissionprob_[state][word])
                    prob = self.startprob_[state] * self.emissionprob_[state][word]
                    if prob > m[state, i]:
                        parent[state, i] = state
                        m[state, i] = prob
            else:
                for s1 in range(self.n_components_):  # prev state
                    for s2 in range(self.n_components_):  # cur  state
                        # print(self.transmat_[s2, s1], self.emissionprob_[s2, word], m[s1, i-1])
                        prob = self.transmat_[s2, s1] * self.emissionprob_[s2, word] * m[s1, i - 1]

                        if prob > m[s2, i]:
                            parent[s2, i] = s1
                            m[s2, i] = prob

        # print()
        # for x in m:
        #     print(x)
        # print()
        # for x in parent:
        #     print(x)

        mostLikelyStateIdx = np.argmax(m[:, -1])
        hidden_states.append(mostLikelyStateIdx)
        log_likelihood += m[mostLikelyStateIdx, -1]
        i = n_steps - 2

        if n_steps > 2:
            while i > 0:
                # put parent of state i
                hidden_states.append(parent[mostLikelyStateIdx, i + 1])
                # add likelihood of state i
                log_likelihood += m[parent[mostLikelyStateIdx, i + 1], i]
                mostLikelyStateIdx = hidden_states[-1]
                i -= 1
        if n_steps > 1:
            hidden_states.append(parent[mostLikelyStateIdx, 1])
            log_likelihood += m[hidden_states[-1], 0]

        return log_likelihood, reversed(hidden_states)

    def decode(self, X):

        if self.strategy == "viterbi":
            return self._viterbi(X)
        elif self.strategy == "greedy":
            return self._greedy(X)

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