# Understanding Word2Vec with Negative Sampling

Word embeddings have become one of the most important ideas in natural language processing. They allow us to take words, which are symbolic, and map them into vectors of numbers that capture their meanings. Words that appear in similar contexts like orange and apple, end up being close to one another in this vector space.

In earlier posts, we saw how the Skip-Gram model creates embeddings by treating the prediction of nearby words as a supervised learning problem. While Skip-Gram works well, it originally relied on a softmax objective, which quickly becomes computationally expensive when you have a large vocabulary. Negative sampling was introduced as a clever solution to this problem. It makes learning much faster while still producing high-quality embeddings.

---

## The Problem with Softmax

The Skip-Gram model tries to predict the probability of every possible word in the vocabulary being the correct target word. This requires computing a softmax over tens or even hundreds of thousands of words. Each training step would mean updating every word in the vocabulary, which is far too slow.

With negative sampling, instead of asking, “What’s the probability distribution over all words?”, it reframes the question: “Given this pair of words, should they go together or not?” This small change transforms the problem into something far more efficient.

---

## Positive and Negative Examples

Suppose we have the pair (orange, juice). These words commonly appear together, so this is treated as a positive example. We label it with a 1.

Now imagine the pair (orange, king). These words rarely appear together, so this becomes a negative example, labeled 0.

Training data is built exactly this way. We start by sampling a center word in this case, orange. We then look within a small window of text (say, ±10 words) and pick one of its real neighbors, such as juice. That gives us one positive pair. After that, we generate multiple negative pairs by keeping orange but randomly choosing other words from the dictionary, like king, book, or of. All of these are labeled 0.

The end result is a supervised dataset where the model must distinguish between real center-target pairs and random, unlikely pairs.

To see how this works, imagine the sentence:

“I drank fresh orange juice this morning.”

If we take the word orange as the center word and look at its nearby neighbors within a small window, one of the words we might pick is juice. Since juice genuinely appears next to orange in the sentence, the pair (orange, juice) becomes a positive example, labeled 1.

Now suppose we keep orange as the center word but instead randomly pick another word from the dictionary, like king. The pair (orange, king) has nothing to do with our sentence, so it becomes a negative example, labeled 0.

If C is the maximum window size to pick positive examples or true targets from and k is the number of negative samples, 

for each center word, we don’t just pick one target word and stop. Instead, we can dynamically decide how many positive pairs to generate. Imagine we set a maximum window size of C = 5. For the word orange, we might randomly pick a window size of 3. That means we will take orange and generate three positive examples from its actual neighbors, such as (orange, juice), (orange, fresh), and (orange, this).

For each of these positives, we then sample k negative words from the vocabulary, like book or river, to form pairs such as (orange, book) and (orange, river). Even if one of these randomly chosen words happens to co-occur with orange elsewhere in the corpus, in this training step it still serves as a negative.

We repeat this process for every center word in the sentence. If the word drank appears and its sampled window size is 5, then we gather five positive pairs for drank, and again, each of them is matched with k negative examples.

By continuously mixing true neighbors with negative samples, the model is forced to sharpen its understanding of which words genuinely belong together. This constant contrast between “real” and “fake” pairs is what gradually sculpts the embeddings so that meaningful words cluster close together, while unrelated ones are pushed apart.

---

## Turning the Problem into Logistic Regression

With this dataset in hand, the model is trained as a set of binary classification problems. Each training example is a pair of words, and the task is to predict whether they should be labeled 1 (they appear together) or 0 (they are random).

Mathematically, this is modeled with a logistic regression. The probability that a pair of words belongs together is computed by applying the sigmoid function to the dot product of their embedding vectors. If the result is close to 1, the model believes the words go together. If it’s close to 0, it believes they do not.

Instead of one massive softmax over the entire vocabulary, training now involves updating only k+1 classifiers per example (where k is the number of negative samples): one positive pair and k negative pairs. If k = 4, that means just five updates per training step instead of tens of thousands. This is why negative sampling is dramatically more efficient.

### Understanding the objective 

The negative sampling objective is:

<div>
$$
J = \log \sigma\!\big( {v'_{w_O}}^{\top} v_{w_I} \big) \;+\; 
\sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} \Big[ \log \sigma\!\big( -{v'_{w_i}}^{\top} v_{w_I} \big) \Big]
$$
</div>

So the loss is just:

<div>
$$
\mathcal{L} = - J
$$
</div>

or explicitly:

<div>
$$
\mathcal{L} = - \Bigg( \log \sigma\!\big( {v'_{w_O}}^{\top} v_{w_I} \big) 
+ \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} \Big[ \log \sigma\!\big( -{v'_{w_i}}^{\top} v_{w_I} \big) \Big] \Bigg)
$$
</div>

**In words:**

* The **objective** $J$ is what we want to maximize (make real pairs score high, fake pairs score low).
* The **loss** $\mathcal{L}$ is the negative of that, so the optimizer can minimize it.


### Understand the Setting

* We have a center word (also called input word), let’s denote it $w_I$. Example: orange.
* We have a context word (also called output word), denoted $w_O$. Example: juice.
* We want the embeddings to capture the fact that *orange* and *juice* appear together often.

But instead of predicting the context word among the entire vocabulary (softmax), we do binary classification:

* Is $(w_I, w_O)$ a real pair? (positive example → label 1)
* Or is $(w_I, w)$ a fake pair with some random word $w$? (negative example → label 0)


### Define the Terms

* $v_{w_I}$: embedding vector of the input word (center word).
  → Think of this as the representation of *orange*.

* $v'_{w_O}$: embedding vector of the output word (context word).
  → Think of this as the representation of *juice* in the output space.

* $\sigma(x)$: sigmoid function,

  $$
  \sigma(x) = \frac{1}{1+e^{-x}}
  $$

  It squashes numbers into probabilities between 0 and 1.

* $P_n(w)$: the noise distribution, i.e. how we sample negative words.
  Usually proportional to word frequency raised to the power of $3/4$.

* $k$: number of negative samples per positive pair.

* $w_i$: one negative sample word, drawn from $P_n(w)$.

---

### Break the Formula

The objective has two parts:

#### (1) Positive example term:

<div>
$$
\log \sigma\!\big( {v'_{w_O}}^{\top} v_{w_I} \big)
$$
</div>

* Here, we take the dot product 
<div>
$$
{v'_{w_O}}^{\top} v_{w_I}
$$
</div>.

  * If orange and juice are related, their vectors should align → dot product is large.
  * Large dot product → $\sigma(\cdot) \approx 1$ → log probability ≈ 0 (high).

* Intuition: reward the model if it assigns high probability that real pairs are genuine.

---

#### (2) Negative examples term:

<div>
$$
\sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} \Big[ \log \sigma\!\big( -{v'_{w_i}}^{\top} v_{w_I} \big) \Big]
$$
</div>

* For each fake word $w_i$ (say king, book, the),

  * Compute dot product with orange: 

  <div>
  $$
  {v'_{w_i}}^{\top} v_{w_I}
  $$
  </div>

  * Put a minus sign in front: 

  <div>
  $$
  -{v'_{w_i}}^{\top} v_{w_I}
  $$.
  </div>

  * If the dot product was large, the negative sign flips it, pushing $\sigma(-\cdot)$ toward 0.

  * So the model is encouraged to assign low probability to random, unrelated pairs.

* Intuition: punish the model if it thinks random pairs like (orange, king) are real.

### Intuition of the Whole Objective

The loss is really saying:

* For the real pair (orange, juice):
  Maximize the probability that it’s classified as positive.

* For the fake pairs (orange, king), (orange, book), …:
  Maximize the probability that they’re classified as negative.

So, the embeddings are adjusted so that:

* Related words (orange, juice) have vectors that point in the same direction → large dot product.
* Unrelated words (orange, king) have vectors that are far apart → small (or negative) dot product.

### Why It Works

Each training step updates only:

* The embedding of the center word $v_{w_I}$.
* The embedding of the true context word $v'_{w_O}$.
* The embeddings of k sampled negative words $v'_{w_i}$.

That’s just $k+1$ updates, not the entire vocabulary → fast and scalable.

---


## How Do We Pick the Negative Samples?

Not all sampling strategies are equal. If we simply pick negative words based on how frequently they appear in the corpus, then extremely common words like *the*, *of*, and *and* will dominate the negative set. That isn’t useful, because the model will waste most of its effort learning to distinguish context words from these common fillers.

On the other hand, if we pick words uniformly at random, the distribution won’t reflect natural language very well either.

The creators of Word2Vec found a practical compromise: sample words according to their frequency raised to the power of three-quarters. This weighting gives common words some chance of being picked, but not overwhelmingly so. It’s a heuristic, not a theoretically perfect solution, but in practice it works very well.

This idea of sampling words based on their frequency raised to 3/4 is called the unigram^3/4 distribution

### Unigram distribution

* A unigram distribution is just the probability of each word occurring in your corpus.
* Formally, if you have a corpus of size $N$ and a word $w$ appears $count(w)$ times:

$$
P_{\text{unigram}}(w) = \frac{\text{count}(w)}{N}
$$

* So frequent words like “the” or “and” have high probability, rare words have low probability.


### Raising to the 3/4 power

According to the word2vec paper, they found that using the raw unigram frequencies as negative sampling weights gives too much emphasis to very frequent words, to reduce this bias, they use:

$$
P_n(w) = \frac{count(w)^{3/4}}{\sum_{w'} count(w')^{3/4}}
$$

That is, each word’s frequency is smoothed by raising it to the power 0.75.


### Intuition

* If you just used raw counts (unigram), “the” and “of” would dominate your negative samples.
* Raising to 3/4 reduces the dominance of very frequent words, but still keeps them more likely than rare words.
* Rare words get sampled less, which makes training more efficient and stable.


### Example

Suppose word counts in a tiny corpus:

| word | count |
| ---- | ----- |
| the  | 1000  |
| cat  | 10    |
| sat  | 5     |

* Unigram probabilities:

$$
P_{\text{unigram}}(\text{the}) = 1000 / 1015 \approx 0.985
$$

$$
P_{\text{unigram}}(\text{cat}) = 10 / 1015 \approx 0.00985
$$

$$
P_{\text{unigram}}(\text{sat}) = 5 / 1015 \approx 0.00493
$$

* Smoothed (3/4 power):

$$
1000^{0.75} \approx 177.8,\quad 10^{0.75} \approx 5.62,\quad 5^{0.75} \approx 3.34
$$

* Normalize:

$$
P_n(\text{the}) = 177.8 / (177.8+5.62+3.34) \approx 0.952
$$

$$
P_n(\text{cat}) = 5.62 / (177.8+5.62+3.34) \approx 0.030
$$

$$
P_n(\text{sat}) = 3.34 / (177.8+5.62+3.34) \approx 0.018
$$

Notice “the” is still most frequent, but cat and sat are much more likely than under raw unigram.

### why it matters

* Using this distribution gives better negative samples.
* Training converges faster and embeddings are higher quality.

---

## Subsampling Frequent Words

In large text corpora, some words appear far too, words like; *the*, *in*, *a*, *and*. These stop words don’t carry much meaning on their own, but because they occur millions of times, they dominate the training process.

For example:

* The co-occurrence of *France* and *Paris* is very meaningful (they are strongly related).
* The co-occurrence of *France* and *the* is not very meaningful (almost every word co-occurs with *the*).

If we keep training on *the* over and over again, the model wastes computation and learns little or no new information

To fix this, Word2Vec uses subsampling: it deliberately discards very frequent words during training. This reduces noise, speeds up learning, and allows the model to focus more on rare but informative words.

### The Subsampling Formula

The probability of discarding a word $w_i$ is given by:

$$
P(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}}
$$

where:

* $f(w_i)$ = frequency of word $w_i$ in the corpus (relative frequency, not raw count),
* $t$ = threshold (typically around $10^{-5}$),
* $P(w_i)$ = probability of removing the word from the training set.

### Intuition

* For frequent words (*the*, *and*), $f(w_i)$ is large → discard probability $P(w_i)$ is close to 1 → they are dropped most of the time.
* For rare words (*France*, *quantum*), $f(w_i)$ is small → discard probability $P(w_i)$ is close to 0 → they are almost always kept.

This way, the model learns more from meaningful co-occurrences rather than being overwhelmed by high-frequency, low-information words.