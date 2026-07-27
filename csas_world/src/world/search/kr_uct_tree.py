"""Multi-ply KR-UCT MCTS over the *real* curling game tree.

This is a genuine recursive Monte-Carlo Tree Search whose transition model is the
AUTHORITATIVE simulator (``world.env_bridge``) -- never the learned dynamics.  It
is built for the continuous curling action space, so it combines two standard
continuous-MCTS ingredients:

  * **Progressive widening** -- a node only acquires a new policy-sampled child
    action once its visit count is large enough (``len(children) <= k * N**alpha``,
    capped at ``max_children``).  Until then it must re-select among the children
    it already has.

  * **Kernel-regression UCT selection** -- among a node's existing children, each
    child's value estimate is regressed against its neighbours in (z-normalised)
    action space via ``env_bridge.kr_smooth`` (the csas KR-UCT smoother), which
    already folds in the UCT exploration bonus through ``uct_c`` and the effective
    kernel mass.  The argmax of the smoothed scores is selected.

Everything is policy/value-agnostic: the neural policy proposals and the leaf
rollout are passed in as callables (``sample_fn`` / ``rollout_value_fn``).

State is the canonical 24-vector; cond is ``[shot_norm, team_order, stone_block]``;
the to-move team's block is ``round(cond[2])``; action is ``[speed, angle, spin, y0]``.

Run the self-test (uniform policy + uniform-rollout value) with::

    cd /mnt/data/curling2/csas_world && PYTHONPATH=src:/mnt/data/curling2/csas_v3/src \\
    GNN_EDGE_SCALAR_MODE=button_visible_plus_curl_arc_reach_with_outgoing GNN_NODE_FEATURE_MODE=none \\
    GNN_RELEASE_NODE_MODE=three_plus_takeout_boundary GNN_EDGE_PRUNE_MODE=none JAX_PLATFORMS=cpu \\
    python3 -m world.search.kr_uct_tree
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from .. import env_bridge
from ..actions import clip_raw


# --------------------------------------------------------------------------- #
# Tree node
# --------------------------------------------------------------------------- #
class _Node:
    """One game position: (state, cond, horizon, to-move perspective).

    Per-child bookkeeping is stored in parallel lists indexed by child slot:
      * ``child_actions``  -- physical action [4] played from this node
      * ``children``       -- child ``_Node`` (None until first transitioned)
      * ``child_n``        -- visit count n_a
      * ``child_w``        -- value-sum W_a, accumulated in THIS node's perspective
    """

    __slots__ = ("state", "cond", "horizon", "persp", "terminal", "terminal_value",
                 "N", "child_actions", "children", "child_n", "child_w", "_action_keys")

    def __init__(self, state: np.ndarray, cond: np.ndarray, horizon: int, root_persp: int):
        self.state = np.asarray(state, dtype=np.float32)
        self.cond = np.asarray(cond, dtype=np.float32)
        self.horizon = int(horizon)
        self.persp = int(round(float(self.cond[2])))  # to-move team's block
        self.terminal = self.horizon <= 0
        # Terminal value is computed once, in ROOT perspective (set lazily).
        self.terminal_value: Optional[float] = None
        self.N = 0
        self.child_actions: List[np.ndarray] = []
        self.children: List[Optional["_Node"]] = []
        self.child_n: List[int] = []
        self.child_w: List[float] = []
        self._action_keys: set = set()

    # --- child management -------------------------------------------------- #
    def _add_action(self, action: np.ndarray) -> int:
        """Add a (clipped, deduped) child action; return its slot, or -1 if dup."""
        a = clip_raw(np.asarray(action, dtype=np.float32).reshape(4))
        key = tuple(np.round(a, 5).tolist())
        if key in self._action_keys:
            return -1
        self._action_keys.add(key)
        self.child_actions.append(a)
        self.children.append(None)
        self.child_n.append(0)
        self.child_w.append(0.0)
        return len(self.child_actions) - 1


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def mcts_search(
    x: np.ndarray,
    c: np.ndarray,
    horizon: int,
    shots_in_end: int,
    root_persp: int,
    *,
    sample_fn: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    rollout_value_fn: Callable[[np.ndarray, np.ndarray, int, int], float],
    action_mean: np.ndarray,
    action_std: np.ndarray,
    n_sims: int = 160,
    k_widen: float = 2.0,
    alpha_widen: float = 0.5,
    kernel_bw: float = 0.18,
    uct_c: float = 0.6,
    noise=None,
    rng: Optional[np.random.Generator] = None,
    max_children: int = 48,
    root_only: bool = False,
    max_depth: int = 0,
    root_candidates: Optional[np.ndarray] = None,
) -> Dict:
    """Run multi-ply KR-UCT MCTS from a root position.

    Parameters
    ----------
    x, c : root state (24,) and condition (3,).
    horizon : throws remaining at the root (>=1 means a throw happens here).
    shots_in_end : total deliveries this end (advances cond one ply).
    root_persp : block (0/1) whose end-margin we maximise; all returned values
        are expressed in THIS perspective.
    sample_fn(state, cond, n) -> (n,4) : neural policy action proposals.
    rollout_value_fn(state, cond, horizon, root_persp) -> float : on-policy MC
        rollout to TERMINAL + rule score, in ROOT perspective.
    action_mean, action_std : (4,) for z-normalising the KR kernel.
    n_sims, k_widen, alpha_widen, kernel_bw, uct_c, max_children : search knobs.
    noise : optional ``LocalNoise`` realising stochastic transitions.
    rng : numpy Generator (defaults to a fresh one).
    root_only : if True, run a DEPTH-1 KR-UCT bandit -- never expand grandchildren.
        Each visit (re-)executes the selected root action with FRESH execution noise and
        re-evaluates its leaf, so repeated visits average execution uncertainty for that
        intended action while the kernel pools evidence across neighbouring actions. This
        makes the returned ``n`` (and the kernel-effective count W(a)=Σ_b K(a,b)n_b derived
        from it) a value-allocated, noise-robust visit statistic over the root actions.
    max_depth : if > 0, cap the tree depth -- a node at depth ``max_depth`` is treated
        as a leaf for search purposes (no further descent, no fresh-child expansion past
        it); its value comes from ``rollout_value_fn`` exactly as for a horizon-driven
        leaf. Use ``max_depth=2`` for the 2-ply training-time operator (root → child →
        eval-via-leaf-fn). 0 disables the cap (horizon-bound tree, the default).
    root_candidates : optional (N,4) action set to PRE-SEED the root's children (e.g. a
        dense proposal from ``generate_candidates`` — policy + structured + diverse +
        uniform). Gives the root eventual-density coverage that policy-only progressive
        widening cannot; KR-UCT selection then allocates the sim budget adaptively over
        them (kernel regression pools evidence across neighbours, and the unvisited-first
        rule guarantees each seed at least one visit when n_sims >= N).

    Returns
    -------
    dict with keys ``actions`` [M,4], ``q`` [M] (root-persp Q per searched root
    action), ``n`` [M] visit counts, ``root_value`` float (root-persp value).
    """
    if rng is None:
        rng = np.random.default_rng()
    root_persp = int(root_persp)
    action_mean = np.asarray(action_mean, dtype=np.float64).reshape(4)
    action_std = np.asarray(action_std, dtype=np.float64).reshape(4)

    root = _Node(x, c, horizon, root_persp)
    if root_candidates is not None:
        for a in np.asarray(root_candidates, dtype=np.float32).reshape(-1, 4):
            root._add_action(a)   # dedupes internally

    for _ in range(int(n_sims)):
        _simulate(root, root_persp, shots_in_end,
                  sample_fn=sample_fn, rollout_value_fn=rollout_value_fn,
                  action_mean=action_mean, action_std=action_std,
                  k_widen=k_widen, alpha_widen=alpha_widen,
                  kernel_bw=kernel_bw, uct_c=uct_c,
                  noise=noise, rng=rng, max_children=max_children, root_only=root_only,
                  max_depth=int(max_depth), depth=0)

    # Root node's to-move IS root_persp, so child W is already in root perspective.
    m = len(root.child_actions)
    actions = (np.stack(root.child_actions, axis=0).astype(np.float32)
               if m else np.zeros((0, 4), dtype=np.float32))
    n_counts = np.asarray(root.child_n, dtype=np.int64)
    q = np.zeros(m, dtype=np.float64)
    for a in range(m):
        q[a] = root.child_w[a] / root.child_n[a] if root.child_n[a] > 0 else 0.0
    total_n = int(n_counts.sum())
    root_value = float((n_counts * q).sum() / total_n) if total_n > 0 else 0.0
    return {"actions": actions, "q": q, "n": n_counts, "root_value": root_value}


# --------------------------------------------------------------------------- #
# Recursion: one simulation (selection -> expansion -> evaluation -> backup)
# --------------------------------------------------------------------------- #
def _simulate(node: _Node, root_persp: int, shots_in_end: int, *,
              sample_fn, rollout_value_fn, action_mean, action_std,
              k_widen, alpha_widen, kernel_bw, uct_c, noise, rng, max_children,
              root_only: bool = False, max_depth: int = 0, depth: int = 0) -> float:
    """Recurse one MCTS simulation through ``node``; return the leaf value in
    ROOT perspective.  Standard MCTS: expand at most one fresh node per call,
    evaluate it by rollout (or terminal score), recurse only on already-expanded
    children, then back the value up.

    If ``max_depth > 0`` and the freshly-expanded child sits at depth ``max_depth``,
    its value is taken from ``rollout_value_fn`` (which is the value head when called
    with the value_leaf=true config branch) and we never recurse past it -- this is
    the 2-ply training-time operator.
    """
    # --- terminal leaf: end is over -> exact rule score (root perspective) --- #
    if node.terminal:
        if node.terminal_value is None:
            node.terminal_value = float(env_bridge.score_end(node.state, root_persp))
        node.N += 1
        return node.terminal_value

    # --- selection / progressive widening ---------------------------------- #
    a_idx = _select_child(node, action_mean=action_mean, action_std=action_std,
                          k_widen=k_widen, alpha_widen=alpha_widen,
                          kernel_bw=kernel_bw, uct_c=uct_c,
                          sample_fn=sample_fn, max_children=max_children)

    fresh_child = node.children[a_idx] is None
    if fresh_child:
        child = _transition(node, a_idx, shots_in_end, root_persp, noise=noise, rng=rng)
        node.children[a_idx] = child
        # Evaluate the freshly-expanded child (ROOT perspective):
        if child.terminal:
            # horizon==1 at this node -> child at horizon 0: exact end score.
            child.terminal_value = float(env_bridge.score_end(child.state, root_persp))
            child.N += 1
            value = child.terminal_value
        else:
            # First visit to a non-terminal child: evaluate via the leaf fn (either
            # an on-policy MC rollout to terminal, or the value head for value_leaf=true).
            child.N += 1
            value = float(rollout_value_fn(child.state, child.cond,
                                           child.horizon, root_persp))
    elif max_depth > 0 and depth + 1 >= max_depth:
        # Depth cap (2-ply training-time operator): we're at the cap; never recurse
        # into already-expanded grandchildren. Re-evaluate via the leaf fn on a fresh
        # transition so repeated visits accumulate value evidence at the cap.
        child = _transition(node, a_idx, shots_in_end, root_persp, noise=noise, rng=rng)
        if child.terminal:
            value = float(env_bridge.score_end(child.state, root_persp))
        else:
            value = float(rollout_value_fn(child.state, child.cond, child.horizon, root_persp))
    elif root_only:
        # Depth-1 KR-UCT bandit: re-execute the selected intended action with FRESH
        # execution noise and re-evaluate its leaf (never expand grandchildren). Repeated
        # visits thus average execution uncertainty for this action; child_w/child_n hold
        # the mean leaf value over its noisy realisations.
        child = _transition(node, a_idx, shots_in_end, root_persp, noise=noise, rng=rng)
        if child.terminal:
            value = float(env_bridge.score_end(child.state, root_persp))
        else:
            value = float(rollout_value_fn(child.state, child.cond, child.horizon, root_persp))
    else:
        # Recurse the tree on an already-expanded child.
        value = _simulate(node.children[a_idx], root_persp, shots_in_end,
                          sample_fn=sample_fn, rollout_value_fn=rollout_value_fn,
                          action_mean=action_mean, action_std=action_std,
                          k_widen=k_widen, alpha_widen=alpha_widen,
                          kernel_bw=kernel_bw, uct_c=uct_c,
                          noise=noise, rng=rng, max_children=max_children, root_only=root_only,
                          max_depth=max_depth, depth=depth + 1)

    # --- backup at THIS node (store W in this node's perspective) ----------- #
    node.N += 1
    node.child_n[a_idx] += 1
    node.child_w[a_idx] += value if node.persp == root_persp else -value
    return value


def _select_child(node: _Node, *, action_mean, action_std, k_widen, alpha_widen,
                  kernel_bw, uct_c, sample_fn, max_children) -> int:
    """Pick a child slot via progressive widening + KR-UCT selection."""
    n_children = len(node.child_actions)
    widen_cap = k_widen * (node.N ** alpha_widen)
    may_widen = (n_children == 0) or (n_children <= widen_cap and n_children < max_children)

    if may_widen:
        a_idx = _widen(node, sample_fn=sample_fn, action_mean=action_mean,
                       action_std=action_std)
        if a_idx >= 0:
            return a_idx  # a freshly-widened action is selected immediately
        # widening produced only duplicates -> fall through to KR-UCT select.

    # KR-UCT selection among existing children.
    actions = np.stack(node.child_actions, axis=0).astype(np.float64)
    mean_vals = np.array([node.child_w[a] / node.child_n[a] if node.child_n[a] > 0 else 0.0
                          for a in range(len(node.child_actions))], dtype=np.float64)
    # If any child is unvisited (n_a == 0), prefer it (treat as a fresh selection).
    unvisited = [a for a in range(len(node.child_actions)) if node.child_n[a] == 0]
    if unvisited:
        return unvisited[0]
    scores = env_bridge.kr_smooth(actions, mean_vals, action_mean, action_std,
                                  kernel_bw, uct_c)
    return int(np.argmax(scores))


def _widen(node: _Node, *, sample_fn, action_mean, action_std) -> int:
    """Sample one new policy action, clip + dedupe, add it; return its slot or -1.

    Ensures a node always ends up with at least one child: if the policy keeps
    proposing duplicates, retry a few times before giving up.
    """
    for _ in range(8):
        proposals = np.asarray(sample_fn(node.state, node.cond, 1),
                               dtype=np.float32).reshape(-1, 4)
        if proposals.shape[0] == 0:
            break
        slot = node._add_action(proposals[0])
        if slot >= 0:
            return slot
    # All duplicates and node already has children -> let caller KR-UCT select.
    if len(node.child_actions) > 0:
        return -1
    # Degenerate fallback: force-add a clipped proposal under a jittered key so a
    # node is never childless (dedupe of an empty set can only happen via NaNs).
    a = clip_raw(np.asarray(sample_fn(node.state, node.cond, 1),
                            dtype=np.float32).reshape(-1, 4)[0])
    node.child_actions.append(a)
    node.children.append(None)
    node.child_n.append(0)
    node.child_w.append(0.0)
    return len(node.child_actions) - 1


def _transition(node: _Node, a_idx: int, shots_in_end: int, root_persp: int, *,
                noise, rng) -> _Node:
    """Simulate the selected action from ``node`` (with optional noise), enforce
    legality, and build the resulting child node."""
    action = node.child_actions[a_idx]
    realized = action
    if noise is not None:
        realized = noise.sample_batch(action[None], 1).reshape(4)
    post = env_bridge.simulate_one(node.state, node.cond, realized)
    corrected, _illegal = env_bridge.apply_legality(node.state, post[None], node.horizon, node.cond)
    child_state = corrected[0]
    child_cond = env_bridge.next_condition(node.cond, shots_in_end)
    return _Node(child_state, child_cond, node.horizon - 1, root_persp)


__all__ = ["mcts_search"]


# --------------------------------------------------------------------------- #
# Self-test / unit test
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    import time

    from ..actions import ACTION_HIGH, ACTION_LOW

    rng = np.random.default_rng(0)

    # Synthetic root: a couple of stones near the house, two throws remaining.
    x = np.zeros(env_bridge.STATE_DIM, dtype=np.float32)
    # stone 0 (block 0) and stone 6 (block 1) placed near the button (raw/4095).
    x[0:2] = np.array([760.0, 820.0], dtype=np.float32) / 4095.0
    x[12:14] = np.array([740.0, 900.0], dtype=np.float32) / 4095.0
    c = np.array([0.5, 1.0, 0.0], dtype=np.float32)   # to-move block 0, has hammer
    horizon = 3
    shots_in_end = 16
    root_persp = int(round(float(c[2])))

    action_mean = 0.5 * (ACTION_LOW + ACTION_HIGH).astype(np.float64)
    action_std = (0.5 * (ACTION_HIGH - ACTION_LOW)).astype(np.float64)

    def sample_fn(state, cond, n):
        return rng.uniform(ACTION_LOW, ACTION_HIGH, size=(int(n), 4)).astype(np.float32)

    def rollout_value_fn(state, cond, h, persp):
        st, cc, hh = np.asarray(state, np.float32).copy(), np.asarray(cond, np.float32).copy(), int(h)
        while hh >= 1:
            a = rng.uniform(ACTION_LOW, ACTION_HIGH, size=(4,)).astype(np.float32)
            post = env_bridge.simulate_one(st, cc, a)
            corrected, _ = env_bridge.apply_legality(st, post[None], hh, cc)
            st = corrected[0]
            cc = env_bridge.next_condition(cc, shots_in_end)
            hh -= 1
        return float(env_bridge.score_end(st, persp))

    print("[self-test] warming JAX backend:", env_bridge.warm_jax())
    t0 = time.perf_counter()
    out = mcts_search(
        x, c, horizon, shots_in_end, root_persp,
        sample_fn=sample_fn, rollout_value_fn=rollout_value_fn,
        action_mean=action_mean, action_std=action_std,
        n_sims=40, rng=rng,
    )
    dt = time.perf_counter() - t0

    actions, q, n, root_value = out["actions"], out["q"], out["n"], out["root_value"]
    print(f"[self-test] elapsed = {dt:.2f}s for n_sims=40")
    print(f"[self-test] actions shape = {actions.shape}  q shape = {q.shape}  n shape = {n.shape}")
    print(f"[self-test] root M (searched root actions) = {len(actions)}")
    print(f"[self-test] sum(n) = {int(n.sum())}  (root expansion + tree backups)")
    print(f"[self-test] root_value = {root_value:.4f}")
    if len(q):
        best = int(np.argmax(n))
        print(f"[self-test] most-visited action idx={best} n={int(n[best])} q={q[best]:.4f}")
        print(f"[self-test] q range = [{q.min():.4f}, {q.max():.4f}]")

    # ----- assertions -----
    assert actions.ndim == 2 and actions.shape[1] == 4, actions.shape
    assert q.shape == (len(actions),), (q.shape, len(actions))
    assert n.shape == (len(actions),), (n.shape, len(actions))
    assert len(actions) >= 1, "root must have at least one child"
    # The root accrues exactly one child-visit per simulation.
    assert int(n.sum()) == 40, f"visit counts should sum to n_sims; got {int(n.sum())}"
    assert np.isfinite(root_value), root_value
    assert np.all(np.isfinite(q)), q
    # The most-visited root action should not be a low-Q one: its Q should be at
    # least the visit-weighted root value (KR-UCT concentrates visits on good arms).
    best = int(np.argmax(n))
    assert q[best] >= root_value - 1e-6, (q[best], root_value)
    print("[self-test] PASS")


if __name__ == "__main__":
    _self_test()
