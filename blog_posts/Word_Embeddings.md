# **Word Embeddings — Teaching Machines the Meaning of Words**

---
One of the core breakthroughs in NLP is something called word embeddings a way to represent words so that machines can understand relationships like:

> *Man is to Woman as King is to Queen.*

Sounds magical, right? let's go a bit deeper.
---

## **How We Used to Represent Words**

Before embeddings, we often used a one-hot vector.
Imagine you have a vocabulary of 10,000 words. Each word is a 10,000-long vector with a single "1" in its position and zeros everywhere else.

Example:

* man →  `[0, 0, 0, ..., 1, ..., 0]`  (1 at position 5391)
* woman → `[0, 0, 0, ..., 1, ..., 0]`  (1 at position 9853)

This sounds neat, but there’s a problem:
No two different words are “close” to each other. In fact:

* The dot product between any two different one-hot vectors = 0.
* The distance between any two one-hot vector is the same.

So, to the computer, apple and orange are just as unrelated as apple and king.

---

## **Why That’s a Problem**

Let’s say our language model learns:

> *I want a glass of orange juice* 

Now it sees:

> *I want a glass of apple \_\_\_*

It should guess juice, right?
But with one-hot vectors, the model doesn’t know apple is similar to orange, so it has to learn “apple juice” from scratch.

We need a way for words with similar meanings to be close in the computer’s mind.

---

## **Enter Word Embeddings**

Instead of a long vector of zeros and ones, what if we gave each word a list of features?
For example:

| Feature | man  | woman | king  | queen | apple | orange |
| ------- | ---- | ----- | ----- | ----- | ----- | ------ |
| Gender  | -1.0 | +1.0  | -0.95 | +0.97 | 0     | 0      |
| Royalty | 0    | 0     | 0.99  | 0.98  | 0     | 0      |
| Is Food | 0    | 0     | 0     | 0     | 1.0   | 1.0    |
| Age     | 0.2  | 0.2   | 0.85  | 0.85  | 0     | 0      |

Suddenly:

* apple and orange have very similar feature values.
* The model can generalize: if orange juice is common, apple juice probably is too.

In practice, we don’t manually define these features.
We let algorithms learn them automatically, often as 300-dimensional vectors.

---

## **A 300-Dimensional World**

Think of each word as a point in a 300D space (Hard to picture? Stick with me).
Words that are related end up close to each other:

* Fruits cluster together.
* Royal titles cluster together.
* Numbers cluster together.

If we use algorithms like t-SNE to compress these 300 dimensions into 2D for visualization, you can literally see the clusters like a language map.

---

## **Why It’s Called an Embedding**

We say a word is embedded because we take it from the abstract idea of “a word” and place it at a specific coordinate in this high-dimensional space.

Example:

* “orange” → a point in 300D space
* “apple” → another point, nearby

And this closeness means similarity.

---

## **Why It Matters**

Word embeddings revolutionized NLP because they:

* Capture meaning and relationships automatically.
* Allow models to generalize across similar words.
* Work even with relatively small training sets.

They’re the reason we can do analogy reasoning like:

```
king - man + woman ≈ queen
```

## From Features to Understanding

Imagine we’re doing Named Entity Recognition (NER); a task where the computer needs to spot names of people, places, or organizations in text.

Example sentence:

> *Sally Johnson is an orange farmer.*

If your model understands that orange farmer is a type of person, it can confidently label Sally Johnson as a person’s name.

In the old one-hot vector world, this would be tough. But with word embeddings, something magical happens: let’s say you train your model on the *orange farmer* example.

Now it sees:

> *Robert Lin is an apple farmer.*

Even if apple farmer never appeared in your training set, the model knows:

* apple is similar to orange
* farmer is the same word

Result? It can still figure out that Robert Lin is probably a person.

---

## Going Even Further: Unseen Words

Here’s where embeddings really shine.

What if your test sentence is:

> *Robert Lin is a durian cultivator.*

If durian and cultivator never appeared in your labeled data, a one-hot model would be lost.
But embeddings trained on massive amounts of text from the internet already know:

* durian is a fruit (like orange and apple)
* cultivator is similar to farmer

So your model can make the leap; durian cultivator is probably a person, even without ever seeing those exact words in training.

---

## Transfer Learning for NLP

Once you’ve learned how to create word embeddings, an exciting question comes up:
“Do I have to train them from scratch every time?”

The answer, thankfully is no.
This is where transfer learning comes in.

Instead of starting over for each new task, you can borrow knowledge from a model trained on a massive text corpus and apply it to your own problem.

1. Step 1; – Train word embeddings on a huge unlabeled text corpus (billions of words from books, articles, Wikipedia…).
   This is cheap you just need text.

2. Step 2; Use these embeddings in your actual NLP task (like NER), where you may have very little labeled data.

3. Optional; If you have enough labeled data for the target task, you can fine-tune the embeddings for even better results.

The key: knowledge from a big dataset (Step 1) transfers to a small dataset (Step 2).

---

## Where It Works Best

Word embeddings have been especially useful for:

* Named Entity Recognition (NER)
* Text summarization
* Co-reference resolution (figuring out when *he* refers to *John*)
* Parsing sentence structure

They’re less helpful for tasks like machine translation or language modeling when you already have tons of data specific to that task.

---

## The Big Picture

By replacing one-hot vectors with embeddings, we give algorithms:

* Better generalization to unseen examples.
* The ability to learn from far less labeled data.
* A bridge between meaning in the real world and numbers in a model.

---

## **How Word Embeddings Solve Analogies**

you now understand how word embeddings can make NLP systems smarter; especially when dealing with new words or small labeled datasets.
But there’s another fascinating property: they can solve analogies.

Even if analogies aren’t the most important NLP application, they’re a fantastic way to peek into what embeddings are really doing.

---

## The Famous Example: King – Man + Woman = ?

If I say:

> Man is to woman as king is to \_\_\_\_?

You probably answer queen instantly.
But can an algorithm figure this out automatically?

Surprisingly… yes.

---

## How It Works in Vector Space

Imagine each word is a point in space.
For simplicity, let’s say our embeddings are just 4 dimensions (in reality, they’re usually 50–300).

Let’s label them:

* e<sub>man</sub> → embedding for “man”
* e<sub>woman</sub> → embedding for “woman”
* e<sub>king</sub> → embedding for “king”
* e<sub>queen</sub> → embedding for “queen”

* If you subtract e<sub>woman</sub> from e<sub>man</sub>, you get a vector that roughly represents the concept of “male → female”.
* If you subtract e<sub>queen</sub> from e<sub>king</sub>, you get almost the exact same vector.

That means the difference between “man” and “woman” is nearly identical to the difference between “king” and “queen”.

---

## Turning This Into an Algorithm

To solve “man is to woman as king is to \_\_\_\_”:

1. Start with the vector for king, subtract man, and
add woman:
   e<sub>king</sub> - e<sub>man</sub> + e<sub>woman</sub>
2. Find the word whose embedding is most similar to this result.

When you do this, the answer with the highest similarity is queen.

---

## Measuring “Similarity” — Cosine Similarity

To compare vectors, we usually use cosine similarity:

<div>
\[
\text{similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \, \|\mathbf{v}\|}
\]
</div>


* If two vectors point in exactly the same direction → similarity = 1.
* If they’re at 90° → similarity = 0.
* If they’re in opposite directions → similarity = -1.

Why cosine?
It ignores vector length and focuses on direction; which often matters more for meaning.

You can use Euclidean distance, but cosine similarity is the go-to for word embeddings.

---

## What Else Can It Learn?

These vector relationships pop up all over language:

* Man : Woman :: Boy : Girl (gender difference)
* Ottawa : Canada :: Nairobi : Kenya (capital–country)
* Big : Bigger :: Tall : Taller (comparative forms)
* Yen : Japan :: Ruble : Russia (currency–country)

All of these patterns emerge just from training embeddings on a large text corpus, no human labels required.

---

## Why This Matters

Even if you’re not building an analogy-solving system, this property tells us something powerful:

> Embeddings capture semantic relationships in a way that can be manipulated with simple vector math.

And that’s why they’ve become such a foundational building block in modern NLP.

---

## The Embedding Matrix

When we talk about learning word embeddings, what we’re really doing is learning something called an embedding matrix. This matrix is at the heart of how algorithms represent words as numerical vectors that capture their meaning.

---

### Vocabulary and Dimensions

Suppose you have a vocabulary of 10,000 words. This vocabulary might include words like:

```
A, Aaron, Orange, Zulu
```

…and perhaps an unknown word token to represent any out-of-vocabulary terms.

If we’re building 300-dimensional embeddings, our embedding matrix $E$ will have:

* 300 rows → one for each dimension of the embedding space.
* 10,000 columns → one for each word in the vocabulary.

So $E$ is a 300 × 10,000 matrix.
Each column of $E$ corresponds to a word’s embedding.

---

### Example: The Word “Orange”

Let’s say the word Orange is the 6,257th word in our vocabulary.
In machine learning notation, we often represent a word using a one-hot vector.

For Orange:

$$
o_{6257} = [0, 0, 0, \dots, 1, \dots, 0]^T
$$

This is a 10,000-dimensional vector with the “1” at position 6,257.

---

### How the Embedding is Retrieved

If we multiply our embedding matrix $E$ (size 300 × 10,000) by this one-hot vector $o_{6257}$ (10,000 × 1), the result is:

* A 300 × 1 vector; the embedding for "Orange".

Why does this work?

* When multiplying, all elements in $o_{6257}$ are zero except the one at position 6,257.
* This multiplication selects the 6,257th column of $E$ directly.
* That column is exactly the 300-dimensional embedding vector for "Orange".

---

### Mathematical Notation

We can write:

$$
E \cdot o_{6257} = E_{6257}
$$

Where:

* $E_{6257}$ is the embedding vector (300 × 1) for "Orange".
* $E$ is the embedding matrix.
* $o_{6257}$ is the one-hot vector for "Orange".

---

### Intuition

Think of $E$ as a lookup table:

* Each column = the learned vector representation of a word.
* Multiplying by a one-hot vector just picks the right column.

This simple operation allows us to represent any word in the vocabulary as a dense, low-dimensional vector; one that captures semantic meaning, not just spelling.

---