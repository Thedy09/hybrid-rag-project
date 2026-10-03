# Metadata filters and chunking

Filters and chunk boundaries decide what either index is allowed to see. They are not cosmetic fields on a search result.

## Filters run first

Metadata filters run before retrieval on both the dense index and the sparse index, so fusion never sees a chunk the filter rejected. The fields available on every chunk are the file name, the page number, the document type, and the chunk identifier. Document type is `md`, `txt`, or `pdf` in this corpus. Page numbers on text files are 1, because those files are not paginated. PDF pages keep the page number from the loader, shifted to be 1-based.

Apply a filter when the question is only meaningful inside one source, for example a question about the operations sheet, or when a document type is out of policy for the audience. Do not apply a filter to hide a weak index. If the filter removes the only chunk that contained the answer, every method will miss, and the comparison table will blame retrieval for an application choice.

People often ask which embedding checkpoint an appliance ships with. The answer is a product fact, not a chunking fact, and it has to be copied from the model card shipped with that appliance. This note does not name the checkpoint.

## Chunk standard POL-CHUNK-800

Standard POL-CHUNK-800 sets the chunk size to 800 characters and the overlap to 150 characters. The splitter prefers paragraph breaks, then line breaks, then sentence boundaries, and only then spaces. The goal is to avoid cutting a sentence when a coarser boundary is available. An 800 character chunk is long enough to hold a short procedure and short enough that one vector is not asked to represent an entire policy manual.

Overlap of 150 characters keeps a sentence that sits on a boundary inside two neighboring chunks. Fusion can then see the same fact twice under two chunk identifiers. That duplication is acceptable. Dropping the overlap is how a definition disappears between two chunks and becomes invisible to both indexes.

## What each chunk must retain

After splitting, every chunk carries:

- File name
- Page number
- Document type
- Chunk identifier

The identifier is stable for a given build: the file stem, the page, and a counter. Fusion uses it to notice that the dense list and the sparse list returned the same chunk. If the identifier were missing, fusion would fall back to the first hundred characters of the text, which can collide when overlap duplicates that opening.

Changing the chunk size changes the evaluation. Rebuild both indexes after any change. Comparing a new vector index with an old keyword index, or the reverse, measures the mismatch rather than the retrieval method.
