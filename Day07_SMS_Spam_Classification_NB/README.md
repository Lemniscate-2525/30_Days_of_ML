# SMS Spam Classification

---

## Problem

Spam detection is a classical Natural Language Processing (NLP) problem where machine learning models automatically classify messages as spam or legitimate communication.

In this project, we build an **end-to-end probabilistic text classification pipeline** using **Multinomial Naive Bayes** and compare it with **Bernoulli Naive Bayes and Logistic Regression.**

Focus Areas:

- NLP Exploratory Data Analysis
- TF-IDF Feature Engineering
- Bayes Theorem and Multinomial Naive Bayes Mathematics
- Model Comparison
- Engineering Tradeoffs
- Failure Case Analysis
- Complexity Analysis

---

## Dataset

**Dataset:** SMS Spam Collection

- ~5572 SMS messages
- Binary classification
- Imbalanced dataset

Target Encoding:

| Label | Meaning |
|-------|---------|
| 0 | Spam |
| 1 | Ham |

---

## Text Exploratory Data Analysis

### Class Distribution

Ham messages form the majority class; spam is the minority. Accuracy alone is misleading here. Recall and F1 Score are the important metrics.

![Class Distribution](eda1.png)

---

### Message Length Distribution

Spam messages tend to be longer due to promotional content, links, and structured language. This length separation provides intuition about class separability even before modeling.

![Message Length Histogram](eda2.png)

---

### WordCloud Visualization

WordCloud shows token frequency visually. Spam messages cluster around words like `free`, `win`, `claim`, `call`. Ham messages contain conversational vocabulary with no strong promotional pattern.

#### Spam WordCloud

![Spam WordCloud](eda3.png)

#### Ham WordCloud

![Ham WordCloud](eda4.png)

---

## Text Preprocessing Pipeline

Steps:

1. Label encoding: `spam → 0`, `ham → 1`
2. TF-IDF Vectorization with `max_features = 3000` and English stopword removal
3. Stratified Train-Test Split (80/20)

TF-IDF internally handles tokenization, lowercasing, and stopword removal in one step.

---

## TF-IDF Mathematical Intuition

TF-IDF assigns importance to a word based on how frequently it appears in a document and how rare it is across all documents.

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\!\left(\frac{N}{\text{DF}(t)}\right)$$

Where:

- $t$ = a specific token (word) in the vocabulary
- $d$ = a specific document (SMS message)
- $\text{TF}(t, d)$ = term frequency; how many times token $t$ appears in message $d$
- $\text{DF}(t)$ = document frequency; the number of messages across the entire dataset that contain token $t$
- $N$ = total number of documents (messages) in the dataset
- $\log(\cdot)$ = natural logarithm; acts as a dampening factor so that extremely rare words are not over-weighted

A word like "free" appearing frequently in spam but rarely in ham gets a high TF-IDF weight. A word like "the" appearing everywhere gets a near-zero weight. This suppresses noise and amplifies the discriminative signal.

---

## Bayes Theorem

Naive Bayes is built on Bayes theorem:

$$P(y \mid x) = \frac{P(x \mid y)\, P(y)}{P(x)}$$

Where:

- $y$ = the class label (spam or ham)
- $x$ = the feature vector representing the message (TF-IDF weights of all tokens)
- $P(y \mid x)$ = posterior probability; the probability the message belongs to class $y$ given the observed words
- $P(x \mid y)$ = likelihood; how probable these specific words are if the message truly belongs to class $y$
- $P(y)$ = prior probability; how often class $y$ appears in the training data (e.g., 13% spam, 87% ham)
- $P(x)$ = evidence; the overall probability of observing these words, constant across all classes

Since $P(x)$ is constant across classes, the classifier simplifies to:

$$P(y \mid x) \propto P(x \mid y)\, P(y)$$

The model predicts the class with the highest posterior probability. $P(y)$ is the prior (how often spam appears in training data). $P(x \mid y)$ is the likelihood (how probable the observed words are given a class).

---

## Multinomial Naive Bayes: Model Physics

For text classification, the feature vector represents token frequencies (or TF-IDF weights).

The "Naive" assumption is that all words are conditionally independent given the class. This is never true in reality (words co-occur in structured sentences), but the independence assumption dramatically simplifies computation and works surprisingly well in practice for bag-of-words representations.

**Likelihood model:**

$$P(x \mid y) = \prod_{i} P(w_i \mid y)^{x_i}$$

Where:

- $w_i$ = the $i$-th word in the vocabulary
- $x_i$ = the count (or TF-IDF weight) of word $w_i$ in the current message
- $P(w_i \mid y)$ = the probability of seeing word $w_i$ in messages of class $y$, estimated from training data
- $\prod_{i}$ = product over all vocabulary words; each word contributes its own likelihood term

Each word contributes its class-conditional probability, raised to the power of how many times it appears. The model treats the message as a "bag of tokens" with no word order.

**Taking log to avoid numerical underflow:**

$$\text{score}(y) = \log P(y) + \sum_{i} x_i \cdot \log P(w_i \mid y)$$

Where:

- $\log P(y)$ = log prior; the log of how often class $y$ appears in training
- $x_i$ = count of word $i$ in the message (acts as a weight; frequent words contribute more)
- $\log P(w_i \mid y)$ = log-likelihood of word $i$ under class $y$; a large negative number for rare words, closer to zero for common ones
- $\sum_{i}$ = sum over all vocabulary words

**How the physics works in practice:** When you send the message "Congratulations, you have won a FREE prize, call now," the model computes:

- $\log P(\text{spam})$: the base rate of spam in training
- Adds $\log P(\text{"free"} \mid \text{spam})$: large value (free appears often in spam)
- Adds $\log P(\text{"won"} \mid \text{spam})$: large value
- Adds $\log P(\text{"call"} \mid \text{spam})$: large value

The spam score accumulates high values from multiple discriminative tokens simultaneously. The ham score accumulates low values for the same tokens. The model assigns spam.

---

## Laplace Smoothing ($\alpha$ Parameter)

Without smoothing, if a token never appears in training spam messages:

$$P(w \mid \text{spam}) = 0$$

This single zero multiplies through the entire product, making the posterior zero regardless of all other evidence. One unseen word kills the prediction.

Laplace smoothing adds a pseudocount $\alpha$ to every token:

$$P(w \mid y) = \frac{\text{count}(w, y) + \alpha}{\text{total\_words}_y + \alpha \cdot V}$$

Where:

- $w$ = the word being estimated
- $y$ = the class (spam or ham)
- $\text{count}(w, y)$ = how many times word $w$ appeared in training messages of class $y$
- $\alpha$ = smoothing parameter; the pseudocount added to every word (default: 1)
- $\text{total\_words}_y$ = total number of word occurrences across all training messages of class $y$
- $V$ = vocabulary size; the total number of unique words in the training set (3,000 here)

- Small $\alpha$ (e.g., 0.01): trusts training data more; rare tokens get very low but nonzero probability
- Large $\alpha$ (e.g., 10): pulls all probabilities toward uniform; the model becomes less data-driven
- Standard choice: $\alpha = 1$ (Laplace smoothing), which adds exactly one pseudocount per token per class

---

## Models Compared

### Bernoulli Naive Bayes

Bernoulli NB changes one thing relative to Multinomial NB: instead of using word counts or TF-IDF weights, it collapses every feature to a binary value; 1 if the word appeared at all in the message, 0 if it did not.

**Likelihood model:**

$$P(x \mid y) = \prod_{i} P(w_i \mid y)^{x_i} \cdot (1 - P(w_i \mid y))^{(1 - x_i)}$$

Where:

- $x_i \in \{0, 1\}$ = binary indicator; 1 if word $w_i$ is present in the message, 0 if absent
- $P(w_i \mid y)$ = probability that word $w_i$ appears in a message of class $y$, estimated from training
- $(1 - P(w_i \mid y))$ = probability that word $w_i$ is absent from a message of class $y$

The key difference from Multinomial: Bernoulli explicitly penalizes the absence of words too. If the word "free" almost always appears in spam, then a message that does not contain "free" gets a nudge toward ham, because the model says "spam messages usually have free; this one does not." Multinomial NB simply ignores absent words; Bernoulli uses them as evidence.

**In log-score form:**

$$\text{score}(y) = \log P(y) + \sum_{i} \left[ x_i \log P(w_i \mid y) + (1 - x_i) \log(1 - P(w_i \mid y)) \right]$$

Bernoulli NB outperforms Multinomial NB here (0.98 vs 0.97 accuracy) because spam classification depends more on whether a spam word appears at all than on how many times it appears. Seeing "free" once is enough signal; seeing it three times adds little extra.

---

### Gaussian Naive Bayes

Gaussian NB is used when features are continuous real-valued numbers rather than token counts. Instead of estimating $P(w_i \mid y)$ from counts, it assumes each feature follows a Gaussian (normal) distribution within each class.

**Likelihood model:**

$$P(x_i \mid y) = \frac{1}{\sqrt{2\pi\,\sigma_y^2}} \exp\!\left(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}\right)$$

Where:

- $x_i$ = the value of feature $i$ for the current message (e.g., message length, word count, punctuation count)
- $\mu_y$ = the mean of feature $i$ across all training messages of class $y$; estimated from data
- $\sigma_y^2$ = the variance of feature $i$ across all training messages of class $y$; estimated from data
- $\exp(\cdot)$ = the exponential function
- $\frac{1}{\sqrt{2\pi\sigma_y^2}}$ = normalization constant ensuring the distribution integrates to 1

The model learns $\mu_y$ and $\sigma_y^2$ for every feature and every class during training. At inference, it plugs the observed $x_i$ into the Gaussian formula to get the probability density, then multiplies across all features (or sums in log space):

$$\text{score}(y) = \log P(y) + \sum_{i} \log P(x_i \mid y)$$

**Why Gaussian NB is not used here:** TF-IDF features are sparse and non-negative; they are count-derived and highly skewed. They do not resemble Gaussian distributions. Gaussian NB would be a poor fit for text. It is more appropriate for tasks with genuinely continuous features such as sensor measurements, medical test values, or financial ratios. For SMS text classification, Multinomial and Bernoulli NB are the correct probabilistic choices.

---

**Multinomial NB vs Logistic Regression:** Logistic Regression learns a discriminative boundary directly via gradient descent. It is more flexible but slower to train, requires more data to generalize, and does not have a closed-form probabilistic interpretation from first principles. NB makes strong independence assumptions but compensates with extreme training speed and good calibration on text data.

---

## Evaluation Metrics

Spam detection prioritizes **Recall** for the spam class. A missed spam (false negative) is costlier than a false alarm (false positive) in most real-world deployments. F1 score balances precision and recall when both matter.

### Multinomial NB

| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Spam) | 0.98 | 0.83 | 0.90 | 149 |
| 1 (Ham) | 0.97 | 1.00 | 0.99 | 966 |
| **Accuracy** | | | **0.97** | 1115 |
| Macro Avg | 0.98 | 0.91 | 0.94 | 1115 |
| Weighted Avg | 0.98 | 0.97 | 0.97 | 1115 |

ROC AUC: **0.9864**

### Bernoulli NB

| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Spam) | 1.00 | 0.87 | 0.93 | 149 |
| 1 (Ham) | 0.98 | 1.00 | 0.99 | 966 |
| **Accuracy** | | | **0.98** | 1115 |
| Macro Avg | 0.99 | 0.93 | 0.96 | 1115 |
| Weighted Avg | 0.98 | 0.98 | 0.98 | 1115 |

### Logistic Regression

| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Spam) | 1.00 | 0.71 | 0.83 | 149 |
| 1 (Ham) | 0.96 | 1.00 | 0.98 | 966 |
| **Accuracy** | | | **0.96** | 1115 |
| Macro Avg | 0.98 | 0.86 | 0.90 | 1115 |
| Weighted Avg | 0.96 | 0.96 | 0.96 | 1115 |

ROC AUC: **0.9839**

---

## Confusion Matrix (Multinomial NB)

![Confusion Matrix](cf3.png)

- **True Negatives (spam correctly caught):** 123
- **False Positives (ham flagged as spam):** 26
- **False Negatives (spam that slipped through):** 2
- **True Positives (ham correctly passed):** 964

Only 2 spam messages slipped through; 26 ham messages were incorrectly flagged. The model is conservative on spam, erring toward false alarms rather than missed spam.

---

## ROC Curve Comparison

![ROC Curve](roc3.png)

Both Multinomial NB and Logistic Regression achieve near-identical ROC curves. Both are well into the top-left corner (high TPR, low FPR), confirming strong separation of the two classes across all thresholds.

---

## Top Spam Indicative Tokens

Model coefficients (log-probabilities) reveal the most influential tokens in spam classification.

![Top Spam Words](res.png)

Tokens like `free`, `call`, `txt`, `win`, and `claim` have the highest log-probability under the spam class, confirming the WordCloud intuition.

---

## Engineering Tradeoffs

| Model | Training Speed | Inference Latency | Memory | Expressiveness |
|-------|---------------|------------------|--------|----------------|
| Multinomial NB | Extremely Fast | Very Low | Low | Limited |
| Bernoulli NB | Very Fast | Low | Low | Limited |
| Logistic Regression | Moderate | Very Low | Medium | Moderate |

---

## Failure Case Analysis

**Conversational spam:** Messages like "Hey, I saw you won something last week" contain spam-adjacent tokens (won) embedded in conversational structure. The bag-of-words model cannot distinguish a casual mention from a genuine spam trigger because it ignores word order and sentence context entirely.

**Ham containing promotional tokens:** A legitimate promotional SMS from a known brand, such as "Your free delivery from Amazon has shipped," contains tokens like "free" that the model associates strongly with spam. The model flags it based on token identity, not sender trust or conversational coherence.

**Word order and syntax are invisible:** Multinomial NB represents every message as a bag of tokens. The sentence "This is not a scam" and "This is a scam" would produce nearly identical feature vectors (differing only in the presence of "not," which is often removed as a stopword). Negation, conditionals, and grammatical structure are completely lost.

**Obfuscated spam vocabulary:** Spammers adapt to detection by using character substitution ("fr33," "c@ll now," "W1N") or synonym rotation ("complimentary" instead of "free"). If the training data does not contain these variants, the model assigns them low spam probability and misses the signal.

**Context and sender identity absent:** A message saying "Call me when you're free" is indistinguishable from a spam trigger on the word "free" alone without knowing the sender. Naive Bayes has no mechanism for encoding relationship context, message history, or sender reputation.

**Class imbalance effect on recall:** With ~87% ham in training data, the prior $P(\text{ham})$ is much larger than $P(\text{spam})$. This pushes posterior probabilities toward ham for borderline cases. The 26 false positives (ham flagged as spam) and 2 false negatives (spam passing through) reflect this prior imbalance.

---

## Time Complexity

**Training:** $O(n \cdot d)$

Where:

- $n$ = number of training documents (SMS messages); 4,457 here (80% of 5,572)
- $d$ = vocabulary size; 3,000 features after TF-IDF filtering

Training simply counts token frequencies per class; one pass through the data. Measured training time: **NB: 0.00203s, LR: 0.02816s**. Logistic Regression is 14x slower due to iterative gradient descent.

**Prediction:** $O(d)$

Where $d$ = vocabulary size (3,000). For each message, compute the log-probability score by summing over $d$ vocabulary terms. Constant in $n$; the model does not slow down as the dataset grows.

---

## Space Complexity

The model stores the vocabulary and the class-conditional log-probabilities: two matrices of shape $(C \times V)$ where:

- $C$ = number of classes; 2 here (spam and ham)
- $V$ = vocabulary size; 3,000 features

Total storage: $2 \times 3{,}000 = 6{,}000$ floats. TF-IDF uses a sparse matrix representation because most documents contain only a small subset of the 3,000-token vocabulary; actual memory usage is far lower than the dense worst case.

---

## Inference Latency

| Model | Inference Latency per Message |
|-------|------------------------------|
| Multinomial NB | $1.25 \times 10^{-6}$ s (~1.25 µs) |
| Logistic Regression | $7.26 \times 10^{-7}$ s (~0.73 µs) |

Both models operate at microsecond latency. Naive Bayes is marginally slower than Logistic Regression at inference because it sums log-probabilities over the vocabulary, while Logistic Regression computes a single dot product. Both are suitable for real-time filtering at any practical message volume.
