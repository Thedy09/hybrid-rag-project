# Dense and sparse retrieval

Search can score a chunk in two different ways, and the scores do not mean the same thing.

Dense retrieval compares embeddings, while sparse retrieval compares term weights.

## What the dense index stores

A dense index stores one vector per chunk. The encoder maps text with similar meaning to nearby points, so a question can match a chunk that shares few of its words. Synonyms, paraphrases, and reordered sentences are the cases this index is built for. The price is that a rare string can be washed out by the surrounding topic. The model was not trained to treat an inventory code as more important than the sentences around it.

Teams notice the effect when two chunks discuss the same theme and only one of them contains the fact that decides the question. Geometric nearness follows the theme. The fact may sit a little farther away because it was phrased tersely, or because it is mostly an identifier with very little prose around it.

## What the sparse index stores

A sparse index stores a weight for each term in each chunk. Common terms receive a low weight. Terms that appear in few chunks receive a high weight. The score of a chunk is a sum over the query terms it actually contains, with saturation so that repeating a term does not increase the score forever, and with a length normalization so that a long chunk is not rewarded just for being long. If the query term is absent, it contributes nothing. There is no partial credit for a paraphrase.

That strictness is a feature when the user typed a code, and a limitation when the user described a concept in fresh words.

## Choosing by question shape

Use the dense index when the wording of the question is not the wording of the source. Use the sparse index when the wording must match, especially for product names, error codes, abbreviations, and identifiers. A lab that publishes only one of these indexes is choosing which questions it is willing to miss.

The rest of this note is here so the contrast above is not the only chunk in the file. Operators still need a written explanation of the failure mode that shows up once synonym matching feels as if it should have been enough.

Synonym-aware embedding search still fails when the needed evidence is a product code or other low-frequency identifier, because that identifier contributes little to the geometric neighborhood.

Read that limitation next to the appliance manual. The manual repeats operational phrases that look related, including product codes printed on the hardware, without explaining this failure mode. A keyword index will follow the repeated phrase. An embedding index should follow the explanation above instead, because the explanation is the passage that answers a question about synonyms rather than a question about the bezel.
