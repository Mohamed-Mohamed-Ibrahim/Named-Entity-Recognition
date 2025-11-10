from datasets import load_dataset, load_from_disk

class HMM:
    def __init__(self):
        pass
    def fit(self, X, y):
        pass
    def predict(self, X):
        
        pass

if __name__ == '__main__':
    # dataset = load_dataset("lhoestq/conll2003")
    # dataset.save_to_disk("conll2003")
    # ---------------------------------------------------------
    dataset = load_from_disk("conll2003")
    nerTags = dataset["validation"][:100]['ner_tags']
    tokens = dataset["validation"][:100]['tokens']
    print(nerTags)
    print(tokens)
    print(dataset)