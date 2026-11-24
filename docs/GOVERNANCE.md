# GOVERNANCE — spec-first, reviewer checklist, spec lineage, pre-registration

This document is repo-owned and normative. It governs how specs are cited,
reviewed, and changed in this repository. It contains no product claims; it is
process only.

## 1. Spec-first rule

Any implementation PR that cites spec sections must reference a spec **merged
on the default branch (`main`)** at a citable version (commit or tag). Spec
changes either ride in the same PR as the build, or merge **before** the build
PR. A build may not cite a branch-only or open-PR spec as authoritative.

Rationale: a citation that only exists on a branch is unverifiable by reviewers
and by anyone reading `main` later.

## 2. Reviewer checklist item

Add to every implementation-PR review checklist:

> **Spec citations resolve on the default branch at the cited version.**
> The reviewer verifies that each cited spec section number exists on `main`
> at the cited version and that its content matches the claim the PR makes.
> Flag any divergence between the citation and the cited text.

## 3. Spec lineage map

Keep this map current as specs merge. Name the authoritative location per spec
family:

| Spec family | Authoritative location | Current state |
|---|---|---|
| Product scope & vendor amendment | `jonathanbarakat-crypto/sift-research` (default branch `main`) | v4 merged on `main` (`docs/SIFT-Product-Scope-v4-Cards-Vault-Folded.md`, superseding v3; v3 §12 hard rules still bind); v4.3 EODHD sole-vendor amendment merged on `main` (`docs/SIFT-Data-Vendor-Amendment-v4.3.md`, pull/13); canonical cross-repo map at `docs/SPEC-INDEX.md` |
| Engineering specs | `jonathanbarakat-crypto/sift` `docs/` (default branch `main`) | `docs/cards-decks-spec-v4.2.md` merged on `main` (pull/35; supersession header updated in pull/63) |

Rule: within a family, the default branch of the owning repo is the only
authoritative copy. Open PRs are drafts, not authority (see §1).

## 4. Pre-registration of gauntlet parameters

Gauntlet thresholds (tau, scope T, cost bounds) are **pre-registered
parameters**. Changes go through a versioned spec update and apply to
**future runs only** — never retroactively to rescue existing results.
Exploratory threshold proposals follow the spec-evolution discipline: gated,
with an against-case, and explicit user go-ahead before any supersession.
