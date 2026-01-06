
# Named Entity Recognition

---

## 👥 Team Members

1. [**Mohamed Hassan**](https://github.com/mohassan5286)
2. [**Omar Hany**](https://github.com/OmarHani4306)
3. [**Mohamed Mohamed Ibrahim**](https://github.com/Mohamed-Mohamed-Ibrahim)

---

## 📋 Objective

The goal of this assignment is to implement a **Word2Vec** model using the Skip-Gram with Negative Sampling (SGNS) approach. Subsequently, the trained embeddings will be utilized for Named Entity Recognition (NER) using two different methods:

1. **Feed Forward Neural Network (FFNN)**
2. **Hidden Markov Model (HMM)** with Viterbi Decoding.

## 🏆 Results

We evaluated the sequence tagging models on the CoNLL-2003 test set.

| Model | Accuracy | Weighted F1-Score |
| --- | --- | --- |
| **FFNN** | **93.35%** | **0.9289** |
| **HMM** | **84.05%** | **0.8744** |

### Detailed Classification Reports

#### 1. Feed Forward Neural Network (FFNN)

**Test Loss:** `0.2449`

| Class | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| **0 (O)** | 0.9507 | 0.9877 | 0.9688 | 38,323 |
| **1 (B-PER)** | 0.8474 | 0.8108 | 0.8287 | 1,617 |
| **2 (I-PER)** | 0.8926 | 0.8054 | 0.8467 | 1,156 |
| **3 (B-ORG)** | 0.7866 | 0.5924 | 0.6758 | 1,661 |
| **4 (I-ORG)** | 0.7992 | 0.5102 | 0.6228 | 835 |
| **5 (B-LOC)** | 0.8826 | 0.7572 | 0.8151 | 1,668 |
| **6 (I-LOC)** | 0.7756 | 0.6187 | 0.6883 | 257 |
| **7 (B-MISC)** | 0.7091 | 0.4444 | 0.5464 | 702 |
| **8 (I-MISC)** | 0.6512 | 0.5185 | 0.5773 | 216 |
| **Total / Avg** | **0.9286** | **0.9335** | **0.9289** | **46,435** |

#### 2. Hidden Markov Model (HMM)

| Class | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| **0 (O)** | 0.9599 | 0.9226 | 0.9409 | 38,323 |
| **1 (B-PER)** | 0.8802 | 0.2863 | 0.4321 | 1,617 |
| **2 (I-PER)** | 0.8011 | 0.2474 | 0.3781 | 1,156 |
| **3 (B-ORG)** | 0.7750 | 0.4521 | 0.5711 | 1,661 |
| **4 (I-ORG)** | 0.6954 | 0.3772 | 0.4891 | 835 |
| **5 (B-LOC)** | 0.8350 | 0.7584 | 0.7948 | 1,668 |
| **6 (I-LOC)** | 0.7888 | 0.4942 | 0.6077 | 257 |
| **7 (B-MISC)** | 0.8771 | 0.5285 | 0.6596 | 702 |
| **8 (I-MISC)** | 0.6233 | 0.4213 | 0.5028 | 216 |
| **Total / Avg** | **0.9335** | **0.8405** | **0.8744** | **46,435** |

---

## 📂 Dataset

* **Source:** [lhoestq/conll2003](https://www.google.com/search?q=https://huggingface.co/datasets/lhoestq/conll2003) (HuggingFace).
* **Task:** Named Entity Recognition (NER).
* **Labels:** The dataset includes 9 labels:
* `0`: **O** (Other)
* `1`: **B-PER** (Begin Person) | `2`: **I-PER** (Inside Person)
* `3`: **B-ORG** (Begin Organization) | `4`: **I-ORG** (Inside Organization)
* `5`: **B-LOC** (Begin Location) | `6`: **I-LOC** (Inside Location)
* `7`: **B-MISC** (Begin Miscellaneous) | `8`: **I-MISC** (Inside Miscellaneous)



---

## 🏗️ Part 1: Word2Vec (Skip-Gram with Negative Sampling)

We implemented the Skip-Gram architecture to learn word embeddings from scratch.

### Key Steps

1. **Preprocessing:** Treated sentences independently and prepared data for Word2Vec training (no extensive cleaning required).
2. **Implementation:**
    - Implemented the **Skip-Gram** model with **Negative Sampling (SGNS)**.
    - Manually implemented the SGNS loss and gradient functions.
    - Optimized training loops to handle the full corpus until convergence.


3. **Visualization:**
    - Plotted training loss across sessions.
    - Visualized word embeddings to demonstrate **word analogy** capabilities.



---

## 🧬 Part 2: Sequence Tagging for NER

We implemented two methods for NER detection using the embeddings trained in Part 1.

### 2.1 Feed Forward Neural Network (FFNN)

* **Input:** Word embeddings from Part 1 (tested variations of center-word vs. context-word embeddings).
* **Architecture:** A standard Feed Forward Network projecting input embeddings to the 9 output classes.
* **Training:**
    - Used the Validation set for hyperparameter tuning.
    - Input shape matches embedding dimension; output shape matches number of classes (9).



### 2.2 Hidden Markov Model (HMM)

A statistical approach using Dynamic Programming.

* **Training:** Computed **Transition** and **Emission** probabilities using the training split.
* **Decoding:** Implemented the **Viterbi Algorithm** to find the most likely sequence of entity tags.
* **Comparison:** Compared performance against the Neural Network approach (see Results section above).

---

## ⚙️ Implementation Notes

* **Embeddings:** The SGNS model handles saving and loading embeddings for multi-session usage.
* **Evaluation Metrics:** All models were evaluated using Precision, Recall, and F1-Score on the test set.
* **Optimization:** The training process for Word2Vec was optimized to handle long convergence times.

---