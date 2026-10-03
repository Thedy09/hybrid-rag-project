# Keyword search with BM25

BM25 is a sparse ranking function. It scores a chunk by the query terms that actually occur in it. The lab uses BM25Okapi, the variant in the `rank-bm25` package.

## Saturation and length

Term frequency saturates. The first time a query term appears, it adds a lot. The tenth time adds little. The saturation parameter k1 controls that curve. The lab leaves k1 at 1.5. Length normalization, controlled by b, stops a long chunk from winning just because it had more chances to mention a common word. The lab leaves b at 0.75. Inverse document frequency then lifts terms that are rare in the corpus and suppresses terms that appear everywhere.

No stemming is applied. "Ship" and "ships" are different terms. "Fail" and "fails" are different terms. A hyphenated identifier is kept as one term, so punctuation around it does not have to be typed exactly, but a plural still has to match.

## When keyword search is the better first stage

Keyword search earns its place when the query contains any of the following:

- Exact product names
- Error codes
- Technical abbreviations
- Identifiers
- Specific terminology that a paraphrase would drop

In those queries the rare term has a high inverse document frequency, and any chunk that lacks the term scores zero on it. A vector index can still return a fluent paragraph about the same theme. That paragraph is a near miss. BM25 will not prefer it over the chunk that contains the identifier, as long as the identifier was preserved by the tokenizer.

## What keyword search misses

A question written in fresh words can share almost no terms with the passage that answers it. BM25 then ranks a chunk that repeats the few overlapping words, even when those words are incidental. The bezel text in the appliance manual is an example of incidental repetition: it tells the operator to copy a product code into a ticket, and it says nothing about why a meaning-based index drops that code. If a question is really about synonyms, the manual's repeated phrase is a distractor, and the explanation lives in the note on dense and sparse retrieval.

## Building the index

The sparse index is rebuilt from the same chunks written into the vector store. Metadata filters are applied after scoring, by dropping chunks that do not match, so the inverse document frequency stays a property of the full corpus. An empty token list is an indexing bug, not a ranking result. The appliance manual documents that condition separately from the ranking formula.
