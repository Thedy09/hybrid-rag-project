# Northwind customer-answer policy

This policy covers what the company stores after a customer receives an answer from the retrieval application. It is a records policy. It does not define ranking formulas, model checkpoints, or appliance codes.

## General records

Operational logs that are not customer answers follow the general records schedule. Those logs include index build times, process exits, and disk watermarks. They are useful to platform engineers and they are not shown to customers. The general schedule tells operators to keep only what they still need for incident review, and to delete the rest on the ordinary cleanup cycle. It does not set the rule for answer text.

Reviewers sometimes confuse that schedule with the rule for customer answers because both refer to an audit log. The audit log is a single store with two kinds of rows. One kind records that a process ran. The other kind records the answer a customer was shown, together with the sources that were attached. Mixing those rows in a retention discussion is how a team keeps answers for too short a time, or keeps process logs for too long.

Customer answers are the rows that contain the generated text and the cited file names. They are the rows a later audit will read when someone asks why a customer was told a particular fact. Process logs cannot answer that question. A retention rule that mentions only "the audit log" without saying which rows is incomplete, and the sections below are the ones that draw the line.

The legal review for this policy asked for plain language. There is no code in the general section on purpose. Identifiers for the binding rules appear only in the sections that state the rule, so a keyword index can find the rule without also landing on this background.

## Binding retention rule

POL-RET-90 requires customer answers written to the audit log to be retained for 90 days.

The clock starts when the answer is shown to the customer, not when the index was built and not when the ticket was closed. Deleting those rows earlier breaks the policy even if the general records schedule would have allowed the deletion. Keeping them longer is permitted only under a legal hold, which is outside this document. The 90 day figure applies to the answer text and to the list of cited sources stored with it.

## Citation rule

POL-CITE-1 requires every generated answer to name the source file and the page number.

An answer that states a fact and cites nothing is incomplete under this policy, even if the fact is right. An answer that cites a file the retriever did not return is also incomplete, because the citation cannot be audited against the context. Page numbers for text and Markdown sources are recorded as page 1. PDF citations use the page that contained the chunk.

## Scope

These rules bind the customer-answer application. They do not bind internal engineering notes, and they do not set service targets for index builds. Service targets live on the operations sheet. Ranking settings live with the fusion standard. This policy only says what must be stored after an answer is delivered, and what that answer must cite.
