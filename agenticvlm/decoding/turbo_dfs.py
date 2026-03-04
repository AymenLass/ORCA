"""Turbo DFS — Depth-First Search with score-guided pruning for constrained generation.

Adapted from the ARC competition reference implementation. The router treats
question-type classification as a **constrained generation task** — the model
generates short label tokens and Turbo DFS explores multiple likely
continuations via recursive depth-first search, pruning branches whose
cumulative negative log-likelihood exceeds ``-log(min_prob)``.

Key functions:
    * :func:`turbo_dfs` — the core recursive DFS on the token tree.
    * :func:`inference_turbo_dfs` — convenience wrapper that runs the initial
      forward pass then calls :func:`turbo_dfs`, returning scored candidates.

Paper parameters for the router:
    * ``min_prob = 0.02``  (accept sequences with probability ≥ 2 %)
    * ``max_new_tokens = 3``  (agent labels are short)
    * ``temperature = 0.9``
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import torch


def turbo_dfs(
    model: Any,
    logits: torch.Tensor,
    path: list[int],
    eos_token_id: int,
    max_new_tokens: int,
    max_score: float,
    max_score_greedy: float,
    temperature: float,
    score: float = 0.0,
    pos: int = 0,
    cache: Optional[list] = None,
) -> list[tuple[float, list[int], list[torch.Tensor]]]:
    """Recursively explore token continuations via depth-first search.

    At each position the routine considers **every** token in the
    vocabulary, pruning any branch whose cumulative NLL exceeds
    ``max_score`` (or ``max_score_greedy`` for the greedy token).
    When the EOS token is reached, the sequence is emitted.

    Args:
        model: The causal language model (with ``.device`` and callable
            for a single forward step with ``input_ids``,
            ``position_ids``, ``past_key_values``).
        logits: Logits tensor for the current position(s).  Shape
            ``(remaining_path + 1, vocab)`` or ``(vocab,)`` if only
            a single step remains.
        path: Pre-computed path of token ids to follow first (used to
            bias towards the greedy sequence before exploring
            alternatives).
        eos_token_id: End-of-sequence token id.
        max_new_tokens: Maximum remaining tokens to generate.
        max_score: NLL ceiling for non-greedy tokens.
        max_score_greedy: NLL ceiling for the greedy token.
        temperature: Sampling temperature applied to the logits.
        score: Cumulative NLL so far.
        pos: Current absolute position in the KV-cache.
        cache: Mutable single-element list holding ``past_key_values``.

    Returns:
        List of ``(cumulative_nll, suffix_tokens, logits_list)`` triples.
        ``suffix_tokens`` are in **reverse** order (callers must reverse).
    """
    if logits.dim() > 1 and logits.shape[0] > 1:
        current_logits, next_logits = logits[0], logits[1:]
    else:
        current_logits = logits[0] if logits.dim() > 1 else logits
        next_logits = None

    nll = -(current_logits / temperature).detach().float().log_softmax(-1).cpu().numpy()

    greedy_index = int(nll.argmin())
    nll_indexed: list[tuple[int, Any]] = list(enumerate(nll))

    # Follow the pre-computed path first (bias towards greedy sequence)
    if path:
        first_idx = path[0]
        nll_indexed[0], nll_indexed[first_idx] = nll_indexed[first_idx], nll_indexed[0]
        path = path[1:]

    suffixes: list[tuple[float, list[int], list[torch.Tensor]]] = []

    for token_id, token_nll in nll_indexed:
        next_score = score + token_nll
        allowed_max = max_score_greedy if token_id == greedy_index else max_score

        if next_score >= allowed_max:
            continue

        if token_id == eos_token_id:
            suffixes.append((next_score, [], []))
        elif max_new_tokens > 1:
            if next_logits is None:
                if pos < cache[0][0][0].shape[2]:
                    cache[0] = tuple(
                        tuple(c[:, :, :pos] for c in layer) for layer in cache[0]
                    )
                step_out = model(
                    input_ids=torch.full((1, 1), token_id, device=model.device),
                    position_ids=torch.full((1, 1), pos, device=model.device),
                    past_key_values=cache[0],
                )
                step_logits = step_out[0][0]  # unbatch
                cache[0] = step_out[1]
            else:
                step_logits = next_logits

            child_suffixes = turbo_dfs(
                model,
                logits=step_logits,
                path=path,
                eos_token_id=eos_token_id,
                max_new_tokens=max_new_tokens - 1,
                max_score=max_score,
                max_score_greedy=allowed_max,
                temperature=temperature,
                score=next_score,
                pos=pos + 1,
                cache=cache,
            )
            for suffix in child_suffixes:
                suffix[1].append(token_id)
                suffix[2].append(current_logits)
            suffixes.extend(child_suffixes)
        # else: max_new_tokens == 1 and not EOS → no suffix emitted

        # After the first token, we no longer have pre-computed forward logits
        next_logits = None

    return suffixes


def inference_turbo_dfs(
    model: Any,
    input_ids: torch.Tensor,
    eos_token_id: int,
    max_new_tokens: int = 3,
    min_prob: float = 0.02,
    min_prob_greedy: float = 1.0,
    temperature: float = 0.9,
    path: Optional[list[int]] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Run Turbo DFS decoding from an initial prompt.

    Performs a single forward pass over the full input to obtain logits
    and KV-cache, then calls :func:`turbo_dfs` to explore candidate
    continuations.

    Args:
        model: Causal language model.
        input_ids: Input token ids — shape ``(seq_len,)`` or ``(1, seq_len)``.
        eos_token_id: End-of-sequence token id.
        max_new_tokens: Maximum tokens to generate per candidate.
        min_prob: Minimum cumulative probability threshold.  Branches
            with probability below this are pruned.
        min_prob_greedy: Separate (looser) threshold for the greedy
            token.  Set to 1 to allow the greedy path unconditionally.
        temperature: Sampling temperature.
        path: Optional pre-computed token path to follow first.
        attention_mask: Optional attention mask (must be all-ones if
            provided — padding is not supported).

    Returns:
        Sorted list of ``(cumulative_nll, token_sequence, logits_array)``
        triples, lowest NLL first.  ``token_sequence`` is a 1-D numpy
        array of generated token ids.
    """
    assert attention_mask is None or attention_mask.all(), "Padding not supported"

    input_ids = torch.as_tensor(input_ids, device=model.device, dtype=torch.long)
    if input_ids.ndim == 2:
        input_ids = input_ids.squeeze(0)
    assert input_ids.ndim == 1, "Batching not supported"

    max_score = -np.log(min_prob)
    max_score_greedy = (
        (-np.log(min_prob_greedy)) if min_prob_greedy > 0 else float("inf")
    )
    max_score_greedy = max(max_score, max_score_greedy)

    if path is None:
        path = []
    if path and path[-1] == eos_token_id:
        path = path[:-1]

    with torch.no_grad():
        full_path = input_ids
        if path:
            full_path = torch.cat(
                [full_path, torch.as_tensor(path, device=model.device)]
            )
        out = model(input_ids=full_path.unsqueeze(0))
        logits = out[0][0, len(input_ids) - 1 :]  # (path_len+1, vocab)
        cache = out[1]

    raw = turbo_dfs(
        model,
        logits=logits,
        path=path,
        eos_token_id=eos_token_id,
        max_new_tokens=max_new_tokens,
        max_score=max_score,
        max_score_greedy=max_score_greedy,
        temperature=temperature,
        score=0.0,
        pos=len(input_ids),
        cache=[cache],
    )

    results = [
        (
            score_val,
            np.array(suffix[::-1]),
            (
                torch.stack(score_arr[::-1]).float().cpu().numpy()
                if score_arr
                else np.empty((0,), dtype=np.float32)
            ),
        )
        for score_val, suffix, score_arr in raw
    ]
    return sorted(results, key=lambda x: x[0])
