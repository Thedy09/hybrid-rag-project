# How combining the two lists helps

Running a vector search and a keyword search side by side is not yet a hybrid system. Hybrid search is what you do with the two lists.

Hybrid search improves retrieval by keeping a chunk that is strong in either the dense list or the sparse list, then ordering that union with a rank-based fusion rule.

## Why a weighted sum of scores is a poor fit

The dense score is a similarity. The keyword score is a sum of term weights. A similarity of 0.82 and a keyword score of 4.1 are not on a shared scale, so adding them, or averaging them, lets the list with the larger numeric range dominate. Normalizing each list helps only if the normalization is stable across queries. Short queries and long queries change the keyword score range. A rank-based rule avoids that argument entirely. It uses the position in each list and ignores the raw score.

## What the union is for

A chunk that is first in the keyword list and absent from the vector list is often an identifier hit. A chunk that is first in the vector list and absent from the keyword list is often a paraphrase hit. Dropping either list before the union deletes one of those cases. Keeping a fixed number from each list, then fusing, gives both cases a path into the shortlist that the answer model will see.

The lab keeps ten chunks from each list and ten chunks after the union. That budget is large enough for a near miss to survive and small enough for a joint scorer to read every survivor. If the gold chunk is outside both top tens, the union cannot invent it. Hybrid search repairs ranking. It does not repair a corpus that never contained the fact.

## What improves in practice

On conceptual questions, the vector list is often already right, and the union should not bury that hit under unrelated keyword matches. On identifier questions, the keyword list is often already right, and the union should carry that hit even when the vector list preferred a topical neighbor. The interesting questions are the ones where each list is partly right. The gold chunk is high in one list and merely present in the other. A rank-based rule promotes it above chunks that won only a single list.

This is also why the lab asks for a comparison table rather than a single demo query. One question can flatter either index. Ten questions, with a mix of paraphrases and exact identifiers, show which index was carrying the win.
