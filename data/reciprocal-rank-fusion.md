# Rank-based fusion

The Northwind retrieval standard recommends reciprocal rank fusion as the way to combine the two ranked lists. It does not add their raw scores.

The contribution of a chunk at zero-based rank r in one list is 1 divided by (constant + r + 1). The constant in that denominator is 60. The Northwind retrieval standard names this setting RRF-K60 because the constant is 60. The standard recommends that constant rather than a hand-tuned sum of the raw scores.

A chunk that is first in a list contributes 1/61. A chunk that is second contributes 1/62. A chunk that appears in both lists receives the sum of its two contributions. After every chunk in both lists has been scored, the chunks are sorted by that sum, high to low. The lab then keeps the first ten.

## Why the constant is large

A constant of 60 keeps the top of the list from completely drowning the rest. The first hit is worth more than the tenth, but not fifty times more. That matters when the correct chunk is rank 1 in the keyword list and only rank 8 in the vector list. Its keyword contribution is 1/61, and its vector contribution is still large enough to beat a chunk that won only one list at rank 1. If the constant were tiny, only the very top ranks would move the sum, and a strong second-list confirmation would not matter.

## Worked positions

Rank 0 contributes about 0.0164. Rank 1 contributes about 0.0161. The gap is small on purpose. Fusion is a voting rule over positions, not a second semantic model. It cannot tell that a rank-1 vector hit is off topic. It can only notice that another chunk was ranked by both indexes. A joint scorer, applied after this shortlist is cut to ten, is what reads the question and the chunk together.

## What the standard does not claim

RRF-K60 does not calibrate scores across queries, and it does not replace evaluation. The standard asks for a table that compares the vector list alone, the keyword list alone, the fused list, and the fused list after the joint scorer. A single lucky query is not evidence that the constant is right. If the gold chunk is outside the ten retrieved from each index, fusion never sees it, and changing the constant will not help.
