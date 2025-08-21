# Word2Vec - Hierarchical Softmax

Training a full neural language model is powerful but computationally expensive. Word2Vec was designed to be simpler, faster, and just as effective at producing high-quality embeddings.

The key idea: instead of predicting entire sentences, we turn the problem into a much simpler task; predicting nearby words.

---

## The Skip-Gram Model

Let’s say you have this sentence:

> I drank a glass of orange juice this morning.

Suppose we choose the word orange as our context word.
Now, Word2Vec asks: “Which words are likely to appear near ‘orange’?”

Within a window (say, ±5 words), possible target words could be:

* juice (1 word after)
* glass (2 words before)
* my (maybe 3 words before, depending on the sentence)

So, training pairs might look like this:

* (orange → juice)
* (orange → glass)
* (orange → my)

This gives us a supervised learning problem:
Given a context word, predict a randomly chosen target word from its neighborhood.

---

## Turning Words into Vectors

To feed words into our model, we start with a one-hot vector (a binary vector where one position is “on” for the chosen word).

But one-hot vectors are too sparse and don’t capture meaning. So, we multiply them by an embedding matrix to get a dense vector representation.

* Context word → embedding vector (E<sub>c</sub>)
* Feed E<sub>c</sub> into a softmax classifier
* Output: probabilities for every word in the vocabulary

In other words: given orange, the model should assign high probability to juice and lower probability to unrelated words like car.

---

## The Training Objective

In the original implementation, the standard softmax loss was used:

$$
\text{Loss} = - \log P(\text{target word} \mid \text{context word})
$$

As training progresses, the embedding vectors adjust so that words appearing in similar contexts (like orange and apple) end up close to each other in vector space. This is how semantic meaning naturally emerges from the training objective.

---

## The Computational Problem

Here’s the catch:

If your vocabulary has 10,000 words, the softmax step requires summing over all 10,000 words for every training example.

If your vocabulary has 1 million words (not unusual in real data), this becomes painfully slow.

---

## Hierarchical Softmax to the Rescue

One clever solution is hierarchical softmax.

Instead of predicting among all words directly, we arrange words in a binary tree:

* First, decide if the word is in the left or right half of the vocab.
* Then split further (quarter vocab, eighth vocab, …)
* Keep going until you reach the exact word.

Each decision is just a binary classification.

This reduces computation from O(V) (linear in vocabulary size) to O(log V) (logarithmic).

And here’s the trick: frequent words like *the* or *of* are placed higher in the tree (so they require fewer steps), while rare words like *durian* are placed deeper.

This makes training efficient without losing accuracy.

---

## Connecting My Code to the Original Word2Vec Loss

In the original Word2Vec paper, the loss is written as the standard softmax negative log-likelihood:

$$
\text{Loss} = - \log P(\text{target word} \mid \text{context word})
$$

This makes sense because, conceptually, we’re asking: “Given this context word, how likely is the correct target word compared to all other words?”

But in my implementation of hierarchical softmax, that probability is factorized into a sequence of binary decisions along the Huffman tree (each internal node decides “go left” vs. “go right”).

That means instead of one big multiclass softmax, we now have a sequence of binary logistic regressions.

And here’s the key point:

* At each node, the training objective is simply:

  $$
  \text{Loss(node)} = - \big[y \log(\sigma(s)) + (1-y)\log(1-\sigma(s))\big]
  $$

  where \$y\$ is whether the path goes left/right, and \$\sigma(s)\$ is the sigmoid of the dot product.
* This is exactly the Binary Cross Entropy (BCE) loss.

So, when you sum these BCE losses along the path from the root to the leaf (target word), you’re reconstructing the same objective as the original softmax formulation; just computed efficiently through a binary tree instead of over the full vocabulary.


---

## Huffman Trees for Hierarchical Softmax

In practice, the tree used for hierarchical softmax is often built using a Huffman tree.

Why Huffman? Because Huffman coding is a way of assigning short codes to frequent items and longer codes to rare items. This makes it a natural fit for word frequency in language:

* Common words (the, of, and) get short paths in the tree.
* Rare words (durian, semaphore) get longer paths.

During training, each word is represented by a unique path from the root to its leaf node. Instead of predicting the entire vocabulary in one shot, the model predicts the sequence of binary decisions along this path.

For example, if the word juice corresponds to the Huffman code `0101`, the model just learns to follow that sequence of left/right decisions in the tree.

This has two major benefits:

1. Efficiency; Training time scales with the length of the path, which is about $\log(V)$, not $V$.
2. Frequency-aware optimization; Since frequent words have shorter codes, they require fewer updates, which matches how often they appear in text.

In other words, Huffman trees make hierarchical softmax not just efficient, but also smart about word frequency.

---


## CBOW: Another Variant

Alongside Skip-Gram, the original Word2Vec paper also introduced CBOW (Continuous Bag of Words).

* Skip-Gram: predict nearby words from a single center word
* CBOW: predict the center word from nearby words

Both methods work, but Skip-Gram is often preferred when you have lots of data and want high-quality embeddings.

## CODE AND EXPLANATION

You can explore the notebook here:

- 📘 <a href="https://github.com/Tony-Ale/Notebooks/blob/main/Word2Vec_Hierarchical_Softmax.ipynb" target="_blank">View on GitHub</a>  
- 🚀 <a href="https://colab.research.google.com/github/Tony-Ale/Notebooks/blob/main/Word2Vec_Hierarchical_Softmax.ipynb" target="_blank">Open in Colab</a>
---

### Download and load the dataset


```python
import os
import zipfile
import urllib.request

# Download the dataset
url = 'http://mattmahoney.net/dc/text8.zip'
filename = 'text8.zip'

if not os.path.exists(filename):
  print("Downloading text8...")
  urllib.request.urlretrieve(url, filename)

# Extract the dataset
with zipfile.ZipFile(filename) as f:
  text = f.read(f.namelist()[0]).decode('utf-8')

print(f"First 300 characters:\n{text[:300]}")
```

    First 300 characters:
     anarchism originated as a term of abuse first used against early working class radicals including the diggers of the english revolution and the sans culottes of the french revolution whilst the term is still used in a pejorative way to describe any act that used violent means to destroy the organiz


---

### Sub-sample and filter tokens

$$
P(\text{keep}(w)) \=\ \min\left(1, \sqrt{\frac{t}{f(w)}}\right)
$$

where

* $t$ is the threshold (e.g. $10^{-5}$)
* $f(w) = \frac{\text{count}(w)}{\text{total tokens}}$ is the normalized frequency of word $w$.

This means:

* Very frequent words (like *the*, *is*, *and*) have large $f(w)$, so $\frac{t}{f(w)}$ becomes small → lower probability of keeping them.
* Rare words have small $f(w)$, so the probability goes close to 1 → they are almost always kept.

In practice, this reduces the dominance of common words in training and speeds up learning.


```python
from collections import Counter
import nltk
from nltk.corpus import stopwords
import random

nltk.download('stopwords')

def subsample_tokens(tokens, threshold=1e-5):
  subsampled_tokens = []
  word_counts = Counter(tokens)
  total_counts  = sum(word_counts.values())

  for token in tokens:
    normalized_freq = word_counts[token]/total_counts
    p_keep = (threshold/normalized_freq)**0.5

    p_keep = min(1.0, max(0.0, p_keep)) # clamp [0, 1]

    if random.random() < p_keep:
      subsampled_tokens.append(token)

  return subsampled_tokens

# English stop words
stop_words = set(stopwords.words('english'))

# Building vocab
tokens = text.split()
print(f"Total tokens: {len(tokens)}")

# Filter out stop words
filtered_tokens = [token for token in tokens if token.lower().strip() not in stop_words]
print(f"Total filtered tokens: {len(filtered_tokens)}")

subsampled_tokens = subsample_tokens(filtered_tokens)
print(f"Total subsampled tokens: {len(subsampled_tokens)}")

word_freq =  Counter(subsampled_tokens)
print(f"Unique words: {len(word_freq)}")

vocab = {word:idx for idx, (word, _) in enumerate(word_freq.items())}

inv_vocab = {idx:word for word, idx in vocab.items()}
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!


    Total tokens: 17005207
    Total filtered tokens: 10890638
    Total subsampled tokens: 4130359
    Unique words: 253702

---

### Create Vocab

```python
# Filter rare words
vocab_size = 10000

most_common = word_freq.most_common(vocab_size)#[:-1000-1:-1]
vocab = {word:idx for idx, (word, _) in enumerate(most_common)}
inv_vocab = {idx:word for word, idx in vocab.items()}
print(f"Unique words: {len(vocab)}")

# filter tokens to keep only those in vocab
tokens = [token for token in tokens if token in vocab]
print(f"Filtered tokens: {len(tokens)}")

# filtered word freq
filtered_word_freq = {word:freq for word, freq in most_common}
print(f"Filtered word freq: {len(filtered_word_freq)}")
```

    Unique words: 10000
    Filtered tokens: 9170125
    Filtered word freq: 10000

---

### Generate Pairs

```python
def generate_skipgram_pairs(tokens:list, window_size=8, max_len=1_000_000):
  pairs = []

  for i, center_word in enumerate(tokens):
    if center_word not in vocab:
      continue

    start_idx = max(0, i - window_size)
    end_idx = min(len(tokens), i + window_size + 1)

    for j in range(start_idx, end_idx):

      if i != j:
        context_word = tokens[j]
        if context_word in vocab:
          pairs.append((vocab[center_word], vocab[context_word]))
          if len(pairs) >= max_len:
            return pairs

  return pairs
```


---
### Implement Huffman Tree

```python
import heapq
from collections import defaultdict

# note that words are only leaf nodes
class HuffmanNode:
  def __init__(self, freq, word=None, left=None, right=None, idx=None):
    self.freq = freq
    self.word = word # None for internal nodes
    self.left = left
    self.right = right
    self.idx = idx # unique index for internal nodes

  def __lt__(self, other):
    return self.freq < other.freq

def build_huffman_tree(word_freq):
  heap = []
  word_to_leaf = {}

  # initialize heap with leaf nodes
  for word, freq in word_freq.items():
    node = HuffmanNode(freq, word)
    heapq.heappush(heap, node)
    word_to_leaf[word] = node

  next_internal_idx = len(word_freq) # internal node indices start after word indices

  # build tree

  while len(heap) > 1:
    node1 = heapq.heappop(heap)
    node2 = heapq.heappop(heap)

    parent = HuffmanNode(
        freq=node1.freq + node2.freq,
        left=node1,
        right=node2,
        idx=next_internal_idx
    )
    next_internal_idx += 1
    heapq.heappush(heap, parent)

  root = heap[0]
  return root, word_to_leaf
```

---

### Extract huffman codes and paths

```python
def extract_codes_and_paths(root, vocab):
  word_idx_to_code = {}
  word_idx_to_path = {}
  max_path_len = 0

  def dfs(node, code, path):
    nonlocal max_path_len
    if node.word is not None:
      idx = vocab[node.word]
      word_idx_to_code[idx] = code.copy()
      word_idx_to_path[idx] = path.copy()
      max_path_len = max(max_path_len, len(path))
      return

    # Go left (0)
    dfs(node.left, code + [0], path + [node.idx])

    # Go right (1)
    dfs(node.right, code + [1], path + [node.idx])

  dfs(root, [], [])
  return word_idx_to_code, word_idx_to_path, max_path_len
```

---

### Helper function to create huffman codes 

```python
def create_huffman_codes(word_freq, vocab):
  root, _ = build_huffman_tree(word_freq)
  codes, paths, max_path_len = extract_codes_and_paths(root, vocab)
  return codes, paths, max_path_len
```

---

### Implement Hierarchical softmax

```python
import torch.nn.functional as F
def hierarchical_softmax_batched(hidden, target_word_idxs, codes, paths, node_embeddings, path_mask, vocab_size):
  """
  hidden: (B, H)  - batch of context vectors
  target_word_idxs: (B,) - not directly used here unless for lookup
  codes: (B, L) - 0 or 1 (padded)
  paths: (B, L) - internal node indices (original indices, padded with 0)
  node_weights: (num_nodes, H) - Tensor of node weights, ordered according to sorted original indices + 0 for padding
  path_mask: (B, L) - 1 for valid path steps, 0 for padding
  vocab_size: int - size of the vocabulary
  """
  B, L = paths.shape
  H = hidden.size(1)

  # Remap original paths indices to tensor indices
  # Original internal node indices start from vocab_size
  # Tensor indices start from 0 (for padding) and then 1 onwards for remapped internal nodes
  # Mapping: original_idx -> original_idx - vocab_size + 1 (if original_idx >= vocab_size), 0 if original_idx is padding (0)
  mapped_paths = torch.zeros_like(paths, dtype=torch.long)
  # Handle padding index (-1)
  mapped_paths[paths == 0] = 0
  # Handle internal node indices
  internal_node_mask = paths >= vocab_size
  mapped_paths[internal_node_mask] = paths[internal_node_mask] - vocab_size


  # Get internal node vectors for each sample in the batch
  # mapped_paths: (B, L) → internal_node_vecs: (B, L, H)
  internal_node_vecs = node_embeddings(mapped_paths)  # (B, L, H)


  # Expand hidden from (B, H) → (B, L, H) to align for dot product
  hidden_expanded = hidden.unsqueeze(1).expand(-1, L, -1)  # (B, L, H)

  # Dot product: (B, L)
  dot_scores = torch.sum(hidden_expanded * internal_node_vecs, dim=2)

  # Sigmoid
  probs = torch.sigmoid(dot_scores)  # (B, L)

  # Log probs
  codes = codes.float()
  losses = F.binary_cross_entropy(probs, codes, reduction='none')

  # Apply mask
  losses = losses * path_mask.float() # (B, L)

  # Normalize per sequence length
  total_losses = losses.sum(dim=1)/path_mask.sum(dim=1).clamp(min=1e-8)

  return total_losses.mean()  # scalar
```

---


### Pad codes and path

```python
import math
def pad_codes_and_paths(pairs, max_path_len, huffman_codes, huffman_paths, pad_idx=0, batch_size=32):

  """
  huffman_codes: dict[word_idx] = list of 0/1
  huffman_paths: dict[word_idx] = list of node indices
  """

  padded_paths = []
  padded_codes = []
  path_masks = []

  pairs_len = len(pairs)
  for idx, (_, t) in enumerate(pairs):
    path = huffman_paths[t]
    code = huffman_codes[t]

    # Padding
    pad_len = max_path_len - len(path)
    padded_path = path + [pad_idx] * pad_len
    padded_code = code + [0] * pad_len
    mask = [1] * len(path) + [0] * pad_len

    padded_paths.append(padded_path)
    padded_codes.append(padded_code)
    path_masks.append(mask)

    if (idx + 1) % batch_size == 0 or idx == pairs_len-1:
      start_index = (math.ceil((idx + 1)/batch_size) - 1) * batch_size
      batch = pairs[start_index: idx+1]
      center_word_indices, context_word_indices = zip(*batch)

      yield (torch.tensor(center_word_indices),
             torch.tensor(context_word_indices),
             torch.tensor(padded_codes, dtype=torch.long),
             torch.tensor(padded_paths, dtype=torch.long),
             torch.tensor(path_masks, dtype=torch.float)
             )

      padded_paths.clear()
      padded_codes.clear()
      path_masks.clear()

```

---

### Implement Model

```python
import torch
import torch.nn as nn
class Word2VecHS(nn.Module):
  def __init__(self, vocab_size, embedding_dim, paths):
    super().__init__()
    self.vocab_size = vocab_size
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    # Create a list of unique internal node indices, including 0 for padding
    internal_node_indices = sorted(list(set(idx for path in paths.values() for idx in path) | {0})) # Include 0 for padding. note that node indices starts at the end of vocab size


    self.node_embeddings = nn.Embedding(len(internal_node_indices), embedding_dim, padding_idx=0) # create embedding for each node
    self.dropout = nn.Dropout(0.6)

  def forward(self, codes, paths, center_words_indices, context_words_indices, path_mask):
    batch_size = center_words_indices.shape[0]
    embeddings = self.embedding(center_words_indices)
    embeddings = self.dropout(embeddings)


    batch_loss = hierarchical_softmax_batched(
        hidden=embeddings,
        target_word_idxs=context_words_indices,
        codes=codes,
        paths=paths,
        node_embeddings=self.node_embeddings, # Pass the tensor
        path_mask=path_mask,
        vocab_size=self.vocab_size
      )

    return batch_loss
```

---

### Initialize Model

```python
codes, paths, max_path_len = create_huffman_codes(filtered_word_freq, vocab=vocab)
model = Word2VecHS(len(vocab), 300, paths)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```




    Word2VecHS(
      (embedding): Embedding(10000, 300)
      (node_embeddings): Embedding(10000, 300, padding_idx=0)
      (dropout): Dropout(p=0.6, inplace=False)
    )


---

```python
pairs = generate_skipgram_pairs(tokens, max_len=1_000_000)
#batches = generate_skipgram_batches(pairs, batch_size=1)
padded_batches = pad_codes_and_paths(pairs, max_path_len, codes, paths, batch_size=32)
```

---

```python
len(pairs)
```




    1000000


---

```python
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

---

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.5, threshold=0.001)
```

---

### Train Model

```python
epochs = 40
model.train()
for epoch in range(epochs):
  running_loss = 0.0
  batch_count = 0

  # Regenerate batches each epoch
  padded_batches = pad_codes_and_paths(pairs, max_path_len, codes, paths, batch_size=32)


  for center_batch, context_batch, codes_batch, paths_batch, path_mask_batch in padded_batches:
    center_batch = center_batch.to(device)
    context_batch = context_batch.to(device)
    codes_batch = codes_batch.to(device)
    paths_batch = paths_batch.to(device)
    path_mask_batch = path_mask_batch.to(device)


    optimizer.zero_grad()
    loss = model(codes_batch, paths_batch, center_batch, context_batch, path_mask_batch)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    batch_count += 1
  scheduler.step(running_loss/batch_count)

  print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/batch_count}, LR: {scheduler.get_last_lr()[0]}")
```

    Epoch 1/40, Loss: 11.909344181529045, LR: 0.001
    Epoch 2/40, Loss: 5.808248297450065, LR: 0.001
    Epoch 3/40, Loss: 3.807633185948372, LR: 0.001
    Epoch 4/40, Loss: 2.772666641082764, LR: 0.001
    Epoch 5/40, Loss: 2.152166331498146, LR: 0.001
    Epoch 6/40, Loss: 1.7515308139390946, LR: 0.001
    Epoch 7/40, Loss: 1.4858631509661675, LR: 0.001
    Epoch 8/40, Loss: 1.2952395974769593, LR: 0.001
    Epoch 9/40, Loss: 1.1529252299599648, LR: 0.001
    Epoch 10/40, Loss: 1.0489013797821998, LR: 0.001
    Epoch 11/40, Loss: 0.9673258307271003, LR: 0.001
    Epoch 12/40, Loss: 0.9046352724394798, LR: 0.001
    Epoch 13/40, Loss: 0.8542623966655731, LR: 0.001
    Epoch 14/40, Loss: 0.8157607867102623, LR: 0.001
    Epoch 15/40, Loss: 0.7810280789175034, LR: 0.001
    Epoch 16/40, Loss: 0.7547019110040665, LR: 0.001
    Epoch 17/40, Loss: 0.7325589862709045, LR: 0.001
    Epoch 18/40, Loss: 0.714932128390789, LR: 0.001
    Epoch 19/40, Loss: 0.6979182883553505, LR: 0.001
    Epoch 20/40, Loss: 0.6850562409195899, LR: 0.001
    Epoch 21/40, Loss: 0.6737167777647972, LR: 0.001
    Epoch 22/40, Loss: 0.6639752472081184, LR: 0.001
    Epoch 23/40, Loss: 0.6559887775287628, LR: 0.001
    Epoch 24/40, Loss: 0.6485856815314293, LR: 0.001
    Epoch 25/40, Loss: 0.6424151053433418, LR: 0.001
    Epoch 26/40, Loss: 0.6363362081794739, LR: 0.001
    Epoch 27/40, Loss: 0.6318357237992287, LR: 0.001
    Epoch 28/40, Loss: 0.6277442938332558, LR: 0.001
    Epoch 29/40, Loss: 0.6233037744064331, LR: 0.001
    Epoch 30/40, Loss: 0.6199342754230499, LR: 0.001
    Epoch 31/40, Loss: 0.6169203655724526, LR: 0.001
    Epoch 32/40, Loss: 0.6140976357069016, LR: 0.001
    Epoch 33/40, Loss: 0.610479837908268, LR: 0.001
    Epoch 34/40, Loss: 0.608790130194664, LR: 0.001
    Epoch 35/40, Loss: 0.6066960755081177, LR: 0.001
    Epoch 36/40, Loss: 0.6038744073171616, LR: 0.001
    Epoch 37/40, Loss: 0.6024134522185326, LR: 0.001
    Epoch 38/40, Loss: 0.6003890723233223, LR: 0.001
    Epoch 39/40, Loss: 0.5981555532755852, LR: 0.001
    Epoch 40/40, Loss: 0.5970449008154869, LR: 0.001


---

```python
# use model to predict next word
def predict_next_topk_words(word_idx, model=model, topk=5):
  topk_words = []
  model.eval()
  with torch.no_grad():
    last_embedding = model.embedding(torch.tensor([word_idx]).to(device))  # shape: (1, D)

    # Normalize embeddings to unit vectors
    normalized_embeddings = F.normalize(model.embedding.weight, dim=1)
    normalized_last = F.normalize(last_embedding, dim=1)

    # Compute cosine similarity
    cos_similarities = torch.matmul(normalized_last, normalized_embeddings.T).squeeze(0)

    topk = torch.topk(cos_similarities, k=topk)
    for i in topk.indices:
      topk_words.append(inv_vocab[i.item()])  # Convert index back to word

    return topk_words


```

---

### Test embedding

```python
last_word = "queen"
last_word_idx = vocab[last_word]

predictions = predict_next_topk_words(last_word_idx)
print(predictions)
```

    ['queen', 'elizabeth', 'governor', 'represented', 'parliament']

