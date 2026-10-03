# Limitations of a vector-only pipeline

A basic retrieval pipeline embeds a question, fetches the nearest chunks, and asks a language model to answer from that shortlist. The baseline is useful, and it fails in ways that are easy to miss if the only inspection is a fluent answer.

These are the five structural limitations of a vector-only RAG pipeline: underweighted rare strings, overloaded chunks, no lexical backup, ignored metadata, and no joint query-document scorer.

## Underweighted rare strings

An embedding model is trained to place paraphrases near each other. A token that appears once in a corpus, such as a policy identifier or a product code, adds little mass to that geometry. The nearest neighbors are then chunks that share the topic, not chunks that contain the decisive token. A reviewer who reads only the top vector hit can cite a topical neighbor that never states the rule.

## Overloaded chunks

One vector represents the whole chunk. If a chunk mixes two policies, the vector sits between them and can be retrieved for both questions while answering neither cleanly. Smaller chunks reduce that mix, at the cost of splitting a procedure across boundaries. Overlap keeps a sentence that lands on a boundary from disappearing, but it does not give the embedding a second chance to represent each fact on its own.

## No lexical backup

When the question uses the same identifier as the source, a keyword index can require that identifier to be present. A vector-only index cannot make that requirement. It can only prefer chunks that point in a similar direction. That is acceptable for conceptual questions and brittle for codes, SKUs, and abbreviations.

## Ignored metadata

Nothing in a plain similarity search stops a retired policy from outranking the current one. The embedding does not know which file is in force unless the application filters metadata before it searches. File name, page, and document type are already on the chunk. Leaving them unused turns a controllable corpus into one flat pile.

## No joint query-document scorer

The first-stage model encodes the question and the chunk separately. It never reads them together. A joint scorer would compare the pair in one pass, but a traditional pipeline stops at the nearest-neighbor list. The separate encoders are what make the first stage fast enough to run over every chunk.

## Missing collection, in plain language

Operators sometimes ask what a vector error means and receive a generic explanation. The usual situation is that the named collection does not exist. Create the collection, rebuild the embeddings, and confirm the persist directory before retrying similarity search. Treat that situation as an operations failure. Look up the exact error identifier in the appliance manual rather than guessing from a topical paragraph like this one.

## How a team should respond

Add a keyword index beside the vector index, combine the two ranked lists, filter metadata, and only then apply a joint scorer. Those steps do not replace the embedding model. They repair the gaps above when the question is specific and the corpus is full of near misses. Conceptual questions can still be carried by the vector list. Identifier questions need the keyword list. A pipeline that keeps only one of those lists will look strong on the questions that match its bias and weak on the others.
