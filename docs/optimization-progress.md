# GraphRAG reliability and usability work

Objective: resolve the full project review, not merely pass the existing tests.
Existing unrelated working-tree changes must be preserved. Tests use temporary
knowledge bases; never rebuild real user data as a regression test.

## Verification checklist

- [x] Reject rebuilds with missing, unsupported or non-file sources instead of
  silently publishing a partial corpus. `tests/test_rebuild_scope.py`: 3 passed.
- [ ] Isolate SDK in-memory namespaces across index generations; test actual SDK
  stores with an existing corpus, rebuild, promotion and process restart.
- [ ] Validate document/chunk/vector/graph consistency before promotion; detect
  unexpected empty graphs without requiring entities in every document.
- [ ] Retain recoverable previous generations and provide verified rollback.
- [ ] Preserve durable manual graph edits independently of bounded display logs.
- [ ] Supply full budgeted text and graph evidence to answer generation.
- [ ] Resolve citations by stable chunk ID and document version/source location.
- [ ] Improve evidence relevance and contextual retrieval for follow-up questions.
- [ ] Align embedding inputs with chunk sizes; expose degradation and eliminate
  placeholder-vector success and destructive text normalization.
- [ ] Enforce extraction schema and scope after model output; fix code-block
  detection; add preview, exclusions, rejection reasons and rule version tracking.
- [ ] Provide corpus-wide entity search, progressive graph expansion, correct
  edge direction/path highlighting, scalable layout and accurate subset counts.
- [ ] Simplify navigation and separate global connections from knowledge-base
  rules; verify browser flows, loading/errors and document-to-evidence navigation.
- [ ] Improve rebuild read availability, cancellation/retry and backfill safety.
- [ ] Add representative retrieval/answer evaluation and quality/cost telemetry.
- [ ] Separate storage, task, extraction, retrieval and evidence responsibilities
  incrementally; establish SDK compatibility contract tests.
- [ ] Harden startup health identification, port handling and shutdown lifecycle.
- [ ] Document source parser limitations and preserve available source positions.
- [ ] Run complete backend/frontend tests, build, browser verification and a
  requirement-by-requirement completion audit.

Conditional features (community reports, OCR, graph database, multi-user RBAC)
require demonstrated product need; they are not mandatory additions merely to
match another GraphRAG architecture.

## 2026-09-08

Started implementation. Source preflight now fails with HTTP 409 and actionable
document details. Added regression coverage for a partially available corpus,
not just the entirely missing case. Namespace isolation and promotion safety
remain unresolved; do not describe rebuild data-loss risk as fixed yet.

Added a file-store SDK adapter that binds all twelve storages and pipeline
reservations to a fresh runtime scope before initialization, preserving legacy
disk paths and logical knowledge-base identity. A real SDK contract test proves
old processed IDs are eligible in the shadow service, JSON memory is isolated,
and promoted document/graph data can be read both by a reopened service and a
fresh interpreter. No model requests are used. This does not yet prove the full
indexing pipeline or promotion validation/rollback requirements.

Publication now reads persisted document/status/chunk/vector-ID records before
touching the active generation. It rejects corpus mismatch, missing records,
wrong chunk ownership, partial extraction and reported entities without a graph.
Legitimate text-only documents remain valid. Nine publication-gate tests added.
Complete graph provenance checks and an explicit
unexpected-empty-graph confirmation workflow remain to be implemented.

Vector publication validation now decodes the persisted float32 matrix and
checks dimensions, row count, unique nonempty IDs, finite values and nonzero
vectors. Checks cover chunk vectors and existing entity/relation vector stores.
Seven corrupt-matrix cases are covered, plus the real SDK flush format using
local deterministic embeddings (no network). Graph-vector coverage and embedding
signature consistency remain outstanding.

Prior generation backups are retained. Promotion refuses to overwrite an
existing backup. Rollback moves candidate files back instead of deleting them,
and no longer unconditionally deletes backups when rollback fails. A recovery
API/UI and process-crash journal are still outstanding; backup retention alone
does not satisfy the full recovery requirement.

Verification: full backend suite 234 passed / 10 skipped (live-server tests and
standalone E2E); after rollback refinements, 26 targeted tests passed.

Manual graph operations now have a separate atomic, unbounded chronological
journal. The UI audit remains capped at 200. Legacy retained entries are seeded
once on the next append; older already-truncated entries cannot be recovered.
Rebuild reads the journal and fails closed on corruption. Tests verify 205
operations across service reopen, legacy seeding without duplication and corrupt
history handling (24 index-consistency tests passed). This solves retention loss,
but a crash between the graph mutation and journaling still needs a transactional
intent/commit protocol; the full durability requirement remains open.

Citation identity now survives backend chat serialization and frontend typing.
Retrieval rank is no longer used as a source position; SDK reference numbers are
not treated as chunk IDs. The viewer uses stable ID, with unique excerpt matching
only for legacy citations. Missing/ambiguous references explicitly report stale
evidence instead of opening an arbitrary block. Frontend: 21 tests passed and
production build passed (existing >500 kB bundle warning remains). Explicit
document-version/source-location metadata and browser flow verification remain.

Answer text evidence is now separate from the 240-character UI excerpt and
shared by synchronous/streaming message construction. All returned sources are
eligible (removed the ten-citation/six-source cutoffs), with a 24,000-character
text budget distributed across sources. Internal answer_content is excluded
from public citation serialization. Regression verifies twelve sources and
evidence beyond character 240; nine citation/answer-quality tests passed.
This is not yet model-token-aware budgeting, and graph context still needs
explicit integration into generation; the full context requirement stays open.

Graph entities/relationships now pass through synchronous, streaming and retry
generation into the shared prompt builder. A separate 8,000-character budget
admits whole graph records with source IDs/file paths. Instructions distinguish
extracted relations from direct source evidence and prohibit fabricated source
numbers. Full suite before the final prompt assertion: 247 passed / 10 skipped.
Graph-only answers are still gated by citation availability; token-aware packing,
graph-only provenance and relevance evaluation remain open.

Lexical relevance now uses full retrieved evidence from all sources and excludes
filenames/paths. Regression covers matching MySQL filenames with unrelated body
text and relevant evidence beyond the UI excerpt/eighth source. Synchronous and
streaming chat now pass sanitized history into context retrieval's SDK query
parameters. This is plumbing, not proof of pronoun resolution: explicit query
rewriting and semantic relevance calibration remain outstanding.

Removed embedding error fallbacks that replaced a rejected text with shorter,
punctuation-stripped or placeholder content. Batch splitting preserves original
inputs; individual rejection now fails explicitly. Initial normalization retains
Unicode and command punctuation and rejects empty input. Existing initial
token/character truncation still needs replacement with aligned chunking and
versioned embedding policy; current corpus vectors are not automatically rebuilt.
24 index-consistency tests passed for this change.

KG structure filtering now preserves raw indentation and counts complete fenced
blocks (backticks and tildes), rather than stripping indentation before testing
it or counting only fence delimiters. Low-substance checks still apply, so
meaningful operations commands are not removed merely because they are code.
Five structural regressions plus functional tests: 18 passed. This fixes the
identified classifier bug, not the broader ontology/exclusion-policy workflow.

Evidence subgraphs now retain only edges whose endpoints survive node capping.
Critical-node markers are recalculated from visible edges rather than hidden
neighbors. Regressions cover a 50-neighbor graph capped to 24 nodes and
relationship-only evidence requiring generated endpoint nodes (2 passed).
Graph exploration layout, full-corpus search and visual direction/path semantics
remain open.

Strict extraction now enforces entity types after SDK parsing and before graph
merge. Disallowed entity records and relations without admitted endpoints are
removed; rejection counts are included in KG filter statistics. Case/whitespace
normalization is allowed, semantic coercion to a nearest type is not. Empty
strict whitelist fails explicitly. Assist/enhanced modes remain unchanged.
Relation-type schema enforcement, candidate review UI and exclusions remain
open. Policy plus index-consistency tests: 27 passed.

Strict endpoint validation now uses admitted entities across the entire parsed
document batch, avoiding removal of valid cross-chunk relations merely because
an endpoint was declared in another chunk. Blank whitelist entries cannot admit
untyped entities. Added regressions for both cases. Cross-document references
and explicit relation-type schema remain separate outstanding work.

Publication validation no longer skips persisted graph/vector checks when the
source manifest is empty: such workspaces may contain manually imported graph
knowledge. Returned counts now reflect that graph. Duplicate chunk references
cannot inflate a document's declared chunk count. Three regressions added;
publication and hardening tests: 35 passed.

Launcher checks application-specific backend/frontend content rather than any
HTTP 200, enforces Vite strictPort 5173, exposes a configurable 180-second wait,
and launches hidden with persistent stdout/stderr logs. README updated. Script
parser passed; updated API contract tests 7 passed. Full suite before contract
update had 265 passed / 10 skipped / 1 expected obsolete health assertion.
Live startup, exact checkout ownership, child-exit detection and stop lifecycle
remain unverified/outstanding; no real services were launched in this step.

Launcher now retains child process handles, propagates native exit codes and
fails immediately if either spawned service exits during readiness polling.
Windows PowerShell parser plus isolated null/running/exited handle checks pass;
the regression executes only the guard function, not the actual services.
Full live startup and safe stop lifecycle remain outstanding.

Rebuild preflight now validates original manifest filenames rather than taking
their basename. Blank and duplicate manifest names fail explicitly instead of
being silently skipped. Added ordinary manifest-consistency regressions for
empty/duplicate names; real knowledge bases were not changed.

PDF parser now records one-based page spans against raw_text, missing-text pages
and explicit ocr_performed=false. Handles close even on extraction failure.
Two reader-double tests verify blank-page numbering/offsets and cleanup. No PDF
artifact was generated or visually verified; real-file testing and persistence
through manifest/chunk/citation APIs remain outstanding.

Upload registration persists an allowlisted source_metadata snapshot including
page spans/OCR flags and the matching text SHA-256. Replacing a source rebuilds
this snapshot rather than carrying obsolete page offsets forward. Manifest
reopen/replacement regression and parser/index tests: 27 passed. Chunk-to-source
mapping and citation API location fields still need implementation.

Successful indexing snapshots source hash/page spans for the active generation.
Indexed chunk viewing now returns conservative source_location (offsets/pages)
only when the stored raw source hash agrees and the chunk occurs exactly once.
Stale, missing or ambiguous source yields null, not a guessed location. Exact
cross-page and stale/duplicate-text regressions plus index tests: 26 passed.
Citation location propagation, UI page navigation and real document QA remain.

Citation detail now displays verified source page numbers separately from chunk
ordinal, cancels pending loads on close/workspace change, and ignores late
responses. Added dialog semantics, Escape close, bounded keyboard focus and
focus return. Interaction regression verifies page display and late-response
behavior after close (4 chat tests passed). Prior full frontend suite 21 passed;
real browser visual QA and direct source-page navigation remain outstanding.

Top-level pages are lazy-loaded with an accessible loading fallback and a keyed
error boundary preserving navigation on chunk/render failures. Production build
entry JS reduced from ~515 kB to ~186 kB (QAChat separately ~184 kB); total code
is not claimed to shrink proportionally. The >500 kB chunk warning is gone.
Frontend tests 22 passed before adding boundary recovery test; real browser
network/performance measurement remains outstanding.

App now reports workspace/session loading and chat create/delete failures via an
alert. Failed or superseded session loads return null rather than an empty list,
so they cannot clear saved active-session selection; aborted workspace callbacks
cannot reset the current selection. Added offline load regression. Workspace
mutation async races and broader action retry/busy feedback remain outstanding.

Chat create/delete results now carry a view-generation guard, so switching
knowledge bases (including away-and-back) prevents stale mutations from replacing
the visible list/selection or showing old errors. Delete completion checks the
current selected chat rather than its old closure. Added delayed-create/switch
regression (2 App tests passed); broader workspace mutation races remain open.

Workspace refresh fallback selection and create/delete completion now respect
view generation. Creating a workspace while navigating elsewhere no longer
forces navigation back; deletion clears only the target's saved chat entry.
Delayed workspace-creation regression added: 3 App tests and build passed.
Delete-confirmation race browser coverage and busy/double-submit protection
remain outstanding.

GraphView no longer implies direction by default: arrows render only when the
graph metadata explicitly marks directed=true. Graph exploration passes this
metadata through. Directed/undirected SVG regression and production build pass.
Exact evidence-edge highlighting and layout/large-graph work remain open.

Evidence edge highlighting now receives actual relation endpoints rather than
inferring paths from co-occurring hit nodes. Undirected matches accept reversed
endpoints; directed matches preserve orientation. Both graph exploration and
answer evidence views use this contract. Two GraphView tests and build passed.
Multi-edge identity would require explicit edge IDs if a multigraph is added.

Force layout moved into a module Worker; the main thread renders an O(N) circle
immediately instead of running 300 pairwise iterations synchronously. Workers
terminate on graph change/unmount and stale results are ignored. Startup/runtime
Worker failures retain basic layout with explicit status. Worker asset is emitted
by Vite. This improves thread isolation, not the algorithm's O(N²) scaling;
large real-browser responsiveness and visual layout quality remain unverified.

Neighbor view now counts only displayed edges and resets on deselection or graph
replacement. Pointer release checks canvas capture ownership and lost capture
clears dragging. The interaction test initially exposed unhandled jsdom errors
despite passing assertions; fixed rather than treating that run as successful.
Full backend regression: 274 passed, 10 skipped. Frontend full regression before
the extra pointer ownership test: 31 passed; production build passed. Rebuild
crash recovery across filesystem replacements remains outstanding.

Publication now checks candidate embedding metadata before resetting the active
service: nonempty vectors require a model signature, matching workspace and
matching dimensions in every populated vector store. Missing metadata can no
longer silently retain the previous generation's model identity. Backup conflict
checks also precede active-service reset. Tests cover invalid/missing signatures,
valid signatures, and rejection before reset. This gate does not establish model
identity from vector values or implement crash recovery.

Rebuild preparation now rejects an existing candidate directory instead of
recursively deleting it, and validates task-directory containment. Failure paths
finalize but retain candidates/backups rather than deleting request-derived
paths. Regression verifies the old-index sentinel survives a repeated prepare
attempt (49 targeted tests passed). Interrupted-publication messaging no longer
claims active data was preserved without proof. Explicit recovery and retention
cleanup UI remain outstanding; failed candidates now consume disk until managed.

Startup now quarantines interrupted publication as phase=recovery_required
instead of ordinary failed/done. Workspace availability checks return structured
INDEX_RECOVERY_REQUIRED and new rebuilds honor that guard. Repeated startup
recovery does not erase quarantine, while other workspaces remain available.
32 scope/consistency tests passed. This is containment, not automatic recovery;
full endpoint coverage and a verified restore action remain outstanding.

Recovery-only guard now also protects document listing, statistics, chunk
preview, workspace deletion and global model configuration mutation. Normal
rebuilds retain their previous list/statistics behavior. Tests assert quarantined
requests fail before opening a service and model-signature mutation is blocked.

Publication writes publication.json before any generation artifact is moved,
including candidate/active/backup paths and initial existence flags. It records
committed only after all replacements and rolled_back only after restoration.
Existing journals are never overwritten. Fault-injection regression verifies
write-ahead ordering and rolled_back state (16 hardening tests passed; 42 combined
hardening/validation tests passed before the extra assertions). Startup does not
yet consume this journal for automatic restoration; path validation and durable
restore sequencing must precede that feature.

Startup can now finish a task interrupted after a fully committed publication:
verify journal identity, exact configured artifact paths, moved-candidate state,
active index contents and embedding signature, then persist succeeded without
reindexing. Incomplete/unknown journals remain quarantined. Disk-backed verifier
and altered-path rejection tests pass; this handles the post-commit task-status
gap, not partially moved generations or restoration of an old generation.

Source-empty rebuilds (including manually maintained graphs) now persist
replaying/committing phases before those operations, allowing post-commit restart
verification to recognize them. Failed manual replay does not publish and reports
the original index was retained, rather than incorrectly claiming it was cleared.
Success/failure branch regressions plus publication validation: 42 passed.

Added a standalone publication rollback engine with whole-generation existence
preflight, overlap rejection, no deletes and repeatable renames. Tests exercise
each replacement boundary, ambiguous later-artifact rejection before any move,
first-generation directory recovery and interruption during rollback. Caller
must supply validated owned paths and hold the writer lock. The engine is not
yet wired into startup/API recovery; no live knowledge-base data was moved.

The rollback engine is now used by the live publication exception path, replacing
per-artifact flag-based restoration. A rollback failure marks recovery_required;
the indexing exception handler preserves that phase instead of overwriting it
with done. Fault injection verifies both successful restoration and failed
restoration retaining the old backup/prepared journal (24 tests passed). Startup
restoration of prepared journals and the explicit recovery API remain pending.

Startup now restores prepared/rolled_back publications under the workspace lock,
using journal identity and exact configured paths before any move. It verifies
the restored index/embedding metadata before clearing quarantine, and marks the
rebuild failed rather than successful. Recovery-required tasks are retried on
startup; unknown/missing journals stay isolated. A disk-backed partial-swap test
verifies old/new generation preservation and repeatability (50 related tests
passed before the retry-filter extension). Real subprocess kill testing and
the explicit recovery API/UI remain outstanding.

Recovery engine now has abrupt-process-exit coverage at all six renames across
three generation artifacts. Each interrupted child exits without exception
cleanup; a fresh interpreter completes recovery and verifies old/new contents.
14 recovery tests passed. This is process-level rollback verification, not yet
a kill test of the full server publication/startup lifecycle or power-loss test.

Added POST index-tasks/{task_id}/recover scoped to the requested workspace.
Only quarantined rebuilds are accepted; concurrent recovery attempts serialize,
and active writers block recovery. Existing verification/restoration code is
reused. Failed recovery or status persistence retains in-memory quarantine.
19 recovery API/scope tests passed before the status-write guard. Frontend wiring
and complete HTTP/browser end-to-end recovery verification remain outstanding.

Knowledge-base management now mounts an independent top-level recovery panel,
listing all workspace recovery-required tasks regardless of age or document-list
failure. Recovery has disabled busy state, failure retry and verified-result
message; switching workspace aborts/ignores stale responses. The UI/UX skill
guided explicit states and keyboard focus styling. Production build passed;
real-browser end-to-end validation and automatic data refresh remain pending.

Recovery success now refreshes document data and updates the visible task result
without reloading the page. Refresh failures explicitly distinguish successful
index recovery from a page-data fetch failure. Interaction tests cover retry,
refresh failure and late recovery results after a workspace switch (3 passed).
Real browser integration remains unverified.

Browser inspection with the backend offline exposed a misleading graph empty
state (zero counts plus advice to index documents despite load failure). Fixed
unknown counts to dashes and added an explicit load-failure/retry panel; verified
the rendered browser accessibility tree after hot reload. This validates the
offline state, not populated graph layout or recovery end-to-end behavior.

Graph loading now tracks graph-fetch failure separately from governance/import
errors and waits for all concurrent requests to settle before ending loading.
Failed refresh labels retained data as stale. Workspace changes clear graph and
entity/relation selection. Regression covers successful empty data, stale refresh
and failed loading after a workspace switch; no old-workspace graph remains.

Strict extraction now enforces configured relationship categories through the
SDK keywords field (one exact category, case/outer-whitespace normalized), with
prompt guidance matching the post-extraction contract. Mixed/unknown/missing
categories are rejected rather than relabeled. Other extraction modes retain
their behavior. Task results expose policy rejection counts for indexing and
backfill; the UI separates entity type, endpoint and relation type rejections,
explicitly as records rather than unique entities. Two display regressions pass;
real-corpus precision/recall evaluation remains pending.

Extraction and backfill now capture one governance snapshot for prompt building,
post-extraction policy and persisted rule summary. Mid-extraction configuration
changes cannot alter that pass's policy. A versioned SHA-256 fingerprint of
effective guidance and constraint fields is stored in document metadata;
timestamp-only changes do not alter it. Snapshot-mutation and fingerprint
regressions added. Cross-document task-wide policy pinning and UI comparison of
indexed/current policy fingerprints remain outstanding.

Governance and reference writes are blocked while that workspace has queued or
running indexing/backfill tasks, including a post-upload-read recheck. New tasks
persist the effective policy fingerprint and check it when execution starts,
including restart resumption. Changed rules/references stop that task instead
of silently resuming under a different policy. Legacy tasks without fingerprints
remain compatible. Full backend regression: 323 passed, 10 skipped. External
file changes between documents still need task-wide immutable snapshot handling.

New tasks now persist config plus effective guidance, validate their fingerprint,
and pass that same snapshot to every index/backfill call. Restarted snapshot-aware
tasks do not reread reference content. The full snapshot is omitted from public
task responses. Legacy fingerprint-only tasks keep change-detection behavior.
Full backend suite: 323 passed, 10 skipped before the additional persisted-snapshot
validation/public-response regression. Real-corpus extraction evaluation remains.

Document policy fingerprints now expose tri-state kg_policy_stale and a visible
reindex warning. Backfill rejects known changed policies before graph access;
additive backfill cannot remove old-policy facts. Strict empty entity whitelists
are rejected at rule save, task creation and extraction context entry, rather
than after model extraction. Fast indexing still bypasses KG validation. API
validation returns INVALID_EXTRACTION_POLICY without persisting invalid rules.

Full backend regression after policy changes: 327 passed, 10 skipped. Template
application now preserves the workspace extraction mode and Other-type setting
instead of silently resetting strict mode to assist. Invalid template/policy
combinations return a structured 400 error. Regression checks preserved options.

The rule editor now explains strict filtering accurately, including optional
relation whitelist, explicit Other membership and reindex requirements. The
Other fallback checkbox is disabled in strict mode, where it has no effect;
mode buttons expose aria-pressed. Interaction regression verifies these states.

Answer evidence preparation now preserves code blocks, identifiers, comparison
operators and table syntax instead of destructively stripping technical sources.
Content budgeting uses max-min allocation: short sources release unused shares
to longer evidence, while total excerpt content stays within 24,000 characters.
11 allocation/citation tests passed, including retention of a formerly truncated
tail conclusion. This is a character content cap, not full-prompt token budgeting
or model-context-window enforcement; headers/history/graph still need accounting.

Evidence bodies now additionally share an 8,000-token cl100k_base estimator cap;
the 24,000-character cap remains. Unicode-safe prefix decoding avoids introducing
replacement characters and special-token spellings are treated as source data.
This estimator is explicitly not the tokenizer of every hosted model. Full
prompt overhead, history, output reservation and configurable model input limits
remain unimplemented; do not claim this prevents every context-window overflow.

Tokenizer initialization failure now falls back to a conservative UTF-8 byte
budget without breaking answer preparation. Failure is cached to avoid repeated
initialization attempts; Unicode prefix boundaries remain intact. 14 evidence/
citation tests passed, including offline initialization and fallback budgeting.

### Recovery with pending uploads

Rollback validation now distinguishes existing generations from newly rebuilt
corpora. A restored manifest may retain uploaded, unindexed documents without
requiring nonexistent index records. Entries marked unindexed but still carrying
an active document ID, chunk list or nonzero chunk count remain rejected. The
strict publication default is unchanged, and the manifest itself is not filtered
or rewritten. Tests cover mixed indexed/pending documents, pending-only disk
rollback, repeated recovery, and inconsistent references. Targeted checks: 50
passed; full backend suite: 343 passed, 10 skipped. No live knowledge base was
rebuilt or modified. Full prompt context budgeting remains outstanding.

### Complete outgoing answer budget guard

Answer settings now persist context_window (default 32,768, an application budget
and not an assertion about the selected model). The conversation settings panel
separates total input/output budget from maximum output and explains estimation.
Both answer paths count all final message text, including system/history/source
headers/graph/question, with a framing allowance and output reserve. Nonstream
retries check again after appending retry instructions. Over-budget requests are
not sent and return an explicit budget diagnostic instead of a citation fallback.
No instructions or evidence are silently truncated by this final guard.

Verification: backend 348 passed, 10 skipped; frontend build passed; frontend
suite 38 passed before adding the settings persistence regression, then all 5
QAChat tests passed including that regression. Model-specific tokenization,
automatic evidence/history packing to fit smaller budgets, and live-provider
boundary validation remain outstanding. The shared estimator is cl100k_base,
with a conservative UTF-8 byte fallback when initialization is unavailable.

### Actual SDK rebuild lifecycle integration

Added four offline integration cases combining real service indexing, SDK JSON
entity/relationship extraction, chunking, vector and graph persistence, shadow
preparation, publication/rollback and reopening. Only hosted LLM/embedding calls
are replaced with deterministic local responses. Both direct publication and the
real task worker are exercised, including a publication exception after swapping
the index directory but before publishing the manifest. The worker path reads
an uploaded text fixture through the actual document loader and writes actual
task JSON in the temporary directory.

Assertions prove that old graph/manifest remain intact before publication;
success replaces old entities and document content; failure restores the old
generation while retaining the new candidate; persisted task results match the
worker result; reopened graph relations still reference current document chunks.
Vector payloads and embedding metadata pass the actual publication validators.
All data paths are temporary; no user knowledge base or hosted model was used.

Verification: four integration cases passed, then full backend 352 passed,
10 skipped. This does not yet cover HTTP task creation, an actual server-process
kill during extraction/publication, power loss, or real-model extraction quality.

### Conversation settings save reliability

Replaced fire-and-forget debounce side effects inside a React state updater with
a scoped autosave hook. Saves are serialized per workspace/session and repeated
edits coalesce to the latest pending draft. Navigation flushes pending edits to
their captured owner; late responses cannot change the new conversation's save
status. Failed drafts survive navigation within the mounted chat page and can be
retried. Session loading does not replace an outstanding draft with stale saved
settings. UI now exposes pending/saving/saved/error states and an error badge when
the settings panel is closed. Controls are disabled while loading or answering.

Sending a question waits for existing settings writes (including their latest
pending payload) before sending its settings snapshot, avoiding an older PATCH
arriving after that snapshot. Tests cover serialization/coalescing, failures and
retry, scope change, unmount flush, missing session, sender waiting, and UI retry.
Frontend suite: 45 passed; production build passed. Browser-close durability,
cross-tab write conflict resolution, and recovery of drafts after a full page
unmount/reload are not provided by this in-memory queue.

### Embedding input-limit consistency

The SDK embedding descriptor now uses configured embed_max_tokens instead of a
hardcoded 480. New index signatures include both embed_max_chars and
embed_max_tokens, so changes to truncation settings cannot silently mix old and
new vector semantics under the same model name/dimension. Historical indexes
without complete input-limit metadata are explicitly incompatible and require
rebuilding; compatibility checks no longer invent metadata from current settings.
This changes legacy behavior but does not mutate or delete the existing index.

Truncation treats special-token spellings as ordinary source data, removes
incomplete UTF-8 suffixes instead of inserting replacement characters, and
rejects limits leaving an empty payload. Non-object metadata is reported as
incompatible instead of crashing the check. Verification: 359 backend tests
passed, 10 skipped before the final metadata-shape cases; all 10 focused
input-limit tests passed afterward. Full long-chunk embedding coverage and
chunk/embedding limit alignment remain outstanding; this change does not
claim to eliminate truncation or to provide model-specific token counting.

### Text chunk coverage within embedding limits

Preview and actual indexing now share an embedding-aligned recursive splitter.
After the requested semantic splitting, oversized pieces are split further at
Unicode character boundaries to satisfy the character cap and both SDK and
embedding-estimator token budgets. Every accepted prefix is measured; source
spans are translated and chunk order is rebuilt. The real SDK path uses its
documented custom chunking hook (an explicit R selector bypasses that hook),
restoring the prior hook afterward and preserving fast-mode KG skipping.

Real SDK tests capture the actual embedding boundary: English long paragraphs
and Chinese/emoji text cover every non-whitespace source character, persisted
chunks equal preview chunks, and normalized embedding payloads equal those
chunks rather than truncated prefixes. Existing complete/fast parity and rebuild
publication tests remain green. More chunks/model calls can result when input
limits are lower than the requested chunk size.

Verification: 12 focused tests passed; full backend 364 passed, 10 skipped.
No existing index was rebuilt. Existing indexes need rebuilding to receive the
new chunk boundaries. This establishes document-text chunk coverage, not precise
token counting for every hosted model, graph-description embedding coverage, or
an empirical retrieval quality improvement on a representative user corpus.

### Large graph layout computation

Layouts above 80 nodes now use a deterministic Barnes-Hut quadtree for repulsion;
smaller graphs retain exact pairwise behavior. The existing worker boundary is
unchanged, and no nodes/edges are sampled away. Self-containing cells are always
expanded, while coincident points are aggregated without deep recursion or
floating-point centroid self-force. Edge attraction remains exact.

Tests compare theta=0 against pairwise forces, check determinism/finite positions
and node preservation for a 400-node layout, and verify a 1,600-node fixture has
aggregate force error below 8% with fewer than one eighth the tree visits of
exact traversal. These are fixture-specific algorithm checks, not a wall-clock
speedup guarantee or visual acceptance. Full frontend: 49 tests passed;
production build passed. Populated-graph browser QA, default display density and
large-corpus progressive exploration remain outstanding.

### Automatic graph label density

Automatic labels now share screen-space overlap checking and a 32-label budget
across node names and selected relationships (at most eight relationship labels).
Focused nodes take precedence, followed by retrieval hits and nearby labels.
Zoom and hub selection no longer force every label on. Underlying nodes and
edges are preserved, with accessible node names, hover detail and the manual
all-label override still available. Text boxes use estimated font widths, so
browser rendering/visual QA remains necessary; this is not a pixel-perfect
overlap guarantee. Nine focused tests and the production build passed, including
an 80-node hub that previously rendered 111 automatic text labels.
