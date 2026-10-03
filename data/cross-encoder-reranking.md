# Joint scoring after fusion

The lab standard reorders the fused shortlist with a cross-encoder before any answer is generated. The model reads the question and one chunk as a pair and emits a relevance logit. Higher is better. The score is not a probability unless a later calibration says it is.

The checkpoint named by the lab standard is cross-encoder/ms-marco-MiniLM-L-6-v2.
The lab standard uses that cross-encoder to rerank fused chunks before an answer is written.

## Why this stage exists

The first-stage indexes encode chunks before the question arrives. They can be wrong in opposite directions, and fusion only knows about ranks. The cross-encoder is allowed to be slower because it scores about ten pairs rather than the whole corpus. It can demote a chunk that shares a rare term but does not answer the question, and it can promote a chunk whose rank was only middling in both lists but which actually contains the answer.

## What the lab keeps

Fusion produces ten chunks. The cross-encoder scores all ten and the pipeline keeps five. Those five are the only context passed to the answer step. A gold chunk that fusion placed at rank 8 can still become the context the model sees, if the pair score says it should. A gold chunk that never entered the fused ten cannot be recovered here. Re-ranking is not a second retrieval over the corpus.

## How to read a comparison

Compare the fused order with the re-ranked order on the same question. If the gold chunk moves up, the joint scorer earned its latency. If the order does not change, the first stage was already enough for that question. If the gold chunk leaves the top five, the joint scorer hurt that question and the example belongs in the error analysis, not in a claim that re-ranking always helps.

The lab records both orders in the evaluation table. Students should be able to point to a question where the cross-encoder changed the rank, and to a question where it did not.
