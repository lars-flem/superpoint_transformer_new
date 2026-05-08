# Bugfixes Audit

Cross-reference of [BUGFIXES.md](BUGFIXES.md) against actual log evidence and the
preprocessing pipeline order, to determine which of the 10 fixes are still
load-bearing now that fix #5 (sparse-subtile `.skip` sentinel) catches sparse
inputs upstream.

## Method

For each fix, checked:
1. Whether its triggering error appears in any log file under `logs/train/` or
   `logs/eval/`.
2. Whether any such log was produced *after* `min_points_per_subtile` was added
   (so we can tell whether fix #5 alone would have prevented the crash).
3. Where the fix sits in the pipeline relative to the `.skip` filter in
   [src/datasets/base.py:895](src/datasets/base.py#L895).

The pre_transform pipeline (from [configs/datamodule/semantic/default_ezsp.yaml:32](configs/datamodule/semantic/default_ezsp.yaml#L32)):

```
SampleXYTiling → [.skip filter at base.py:895] → pre_transform:
  DataTo → SaveNodeIndex → GridSampling3D → PointFeatures →
  GroundElevation (RANSAC) → KNN → AdjacencyGraph → PretrainedCNN →
  GreedyContourPriorPartition → SegmentFeatures → RadiusHorizontalGraph
```

The `.skip` sentinel is written *before* `pre_transform` runs, so empty/sparse
clouds never reach `GridSampling3D`, `KNN`, `PointFeatures` etc. on a
sufficiently high `min_points_per_subtile`.

## Verdict per fix

| # | Verdict | Evidence |
|---|---|---|
| 1 | **STILL NEEDED** | The original crash on Mar 19 ([slurm_spt_viken2022_24202168.err:62-66](logs/train/slurm_spt_viken2022_24202168.err)) was via single-node levels, and fix 10 now short-circuits that codepath. But `cluster_radius_nn_graph` at [neighbors.py:642](src/utils/neighbors.py#L642) calls `scatter_nearest_neighbor` unconditionally with the edge_index from its own radius search — for multi-node levels where the radius returns zero pairs, that edge_index is empty and fix 1's guard is what prevents the crash. Fix 10 only covers `num_nodes < 2`, not "≥2 nodes but no in-radius pairs". |
| 2 | **STILL NEEDED** | Directly downstream of fix 10. `RadiusHorizontalGraph._process` at [graph.py:728-731](src/transforms/graph.py#L728-L731) iterates over **all** levels unconditionally and calls `_process_edge_features_for_single_level`, which calls `subedges(..., nag[i_level].edge_index, ...)` at [graph.py:761](src/transforms/graph.py#L761). On the levels where fix 10 set an empty edge_index (which happens repeatedly per the logs), this would crash without fix 2's guard. The two fixes form a pair: fix 10 emits the empty edge_index, fix 2 absorbs it. |
| 3 | **STILL NEEDED** | Crashed [slurm_spt_viken2022_24244300.err:473](logs/train/slurm_spt_viken2022_24244300.err) on Mar 29 with `min_points_per_subtile: 1000` confirmed in the corresponding `.out`. Crash is in `NAGBatch.from_nag_list` during training-time subgraph sampling, not preprocessing — fix 5 has no reach here. Different NAGs in a batch end up with different optional keys because fix 10 set empty edges on different levels per sample. |
| 4 | Probably redundant (cheap to keep) | Never observed in any log. Hierarchy collapse at preprocessing is prevented by fix 5; with hierarchy depth preserved, the SPT forward loop never overshoots. The fix is a one-line `break` guard, so cost of keeping it is near zero. |
| 5 | **REQUIRED — load-bearing** | Active in all post-Mar-25 runs; produces messages like `"Subtile '32-1-513-133-41__TILE_1-1_OF_2-2.h5' has only 184 points (< 1000), skipping"` in [slurm_spt_viken2022_24244300.out](logs/train/slurm_spt_viken2022_24244300.out). |
| 6 | **STILL NEEDED** | Crashed [slurm_spt_viken2022_24237158.err:65](logs/train/slurm_spt_viken2022_24237158.err) on Mar 26 with `min_points_per_subtile=1000` active. Root cause is XY-degeneracy of *ground candidates*, which is independent of total point count — a 50k-point subtile over flat water can still have collinear ground-class candidates. |
| 7 | Probably redundant | Both `GridSampling3D` and `QuantizePointCoordinates` run inside `pre_transform`, which fires *after* the `.skip` filter in [base.py:895](src/datasets/base.py#L895). With threshold ≥ 1, an empty cloud cannot reach them. Caveat: only safe if `min_points_per_subtile > 0` is enforced in configs. |
| 8 | **STILL NEEDED (current form)** | Span clamp + empty-input guard run *before* the size check, so they're necessary regardless of fix 5. The empty-tile fallback you already removed was firing constantly in older runs (e.g. `"SampleXYTiling: Requested tile (1, 1) is empty. Falling back to tile (0, 0) with 59931 points"` in [slurm_spt_viken2022_24237158.out](logs/train/slurm_spt_viken2022_24237158.out)) — those cases are now handled by fix 5. |
| 9 | Probably redundant | KNN, Inliers, Outliers, PointFeatures all run after the `.skip` filter. No log evidence of firing. |
| 10 | **REQUIRED — fires constantly** | `"NAG only has 1 node(s) at level=X. Skipping horizontal graph construction"` appears repeatedly across [24237158.out](logs/train/slurm_spt_viken2022_24237158.out), [24244300.out](logs/train/slurm_spt_viken2022_24244300.out), [24230418.out](logs/train/slurm_spt_viken2022_24230418.out), [24231097.out](logs/train/slurm_spt_viken2022_24231097.out) with `min_points_per_subtile=1000` active. Threshold cannot prevent hierarchy from collapsing to 1 node at deeper levels — that's a function of partitioning, not raw point count. |

## Summary

- **Definitely keep:** 1, 2, 3, 5, 6, 8 (current form), 10. Fixes 1 and 2 are both downstream guards still reachable on legitimate codepaths; fix 2 in particular fires every time fix 10 does.
- **Safe to remove:** 7, 9 (preconditioned out by fix 5).
- **Borderline:** 4. Probably redundant but the cost of keeping the one-line `break` guard is negligible.

## Orphan finding (not in BUGFIXES.md)

[src/utils/ground.py:115-131](src/utils/ground.py#L115-L131) contains an
undocumented workaround: when `len(pos) < 3`, it synthesizes 2 fake points by
copying the first point and offsetting XY by ±10. This is currently firing in
runs (`"RAnzac wont work i simply create two extra points..."` debug print
visible across viken `.out` logs). It's complementary to fix 6 (which handles
the `≥ 3` XY-degenerate case) and either belongs in BUGFIXES.md or should be
cleaned up if no longer needed.

## Log files referenced

| Log | Date | Crash type | `min_points_per_subtile` |
|---|---|---|---|
| [slurm_spt_viken2022_24202168.err](logs/train/slurm_spt_viken2022_24202168.err) | Mar 19 | Fix 1 (`scatter_nearest_neighbor` empty `torch.cat`) | not yet introduced |
| [slurm_spt_viken2022_24237158.err](logs/train/slurm_spt_viken2022_24237158.err) | Mar 26 | Fix 6 (RANSAC consensus) | 1000 (active) |
| [slurm_spt_viken2022_24244300.err](logs/train/slurm_spt_viken2022_24244300.err) | Mar 29 | Fix 3 (KeyError `'sub'` in batching) | 1000 (active) |
| [slurm_spt_viken2022_24230418.out](logs/train/slurm_spt_viken2022_24230418.out), [24231097.out](logs/train/slurm_spt_viken2022_24231097.out), [24237158.out](logs/train/slurm_spt_viken2022_24237158.out), [24244300.out](logs/train/slurm_spt_viken2022_24244300.out) | Mar 24-29 | Fix 10 warnings firing repeatedly | 1000 (active) |
