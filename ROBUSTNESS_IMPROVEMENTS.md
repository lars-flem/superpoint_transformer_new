# Robustness Improvements for Sparse and Degenerate Data

This document catalogues the changes made to the SuperPoint Transformer codebase
to make it robust to the kinds of degenerate inputs that arise when training on
national-scale Norwegian airborne LiDAR data (water tiles, narrow strips, sparse
subregions, and so on). The upstream code largely assumed well-conditioned point
clouds; the changes here extend it to handle the messier reality of unfiltered
LAS tiles. Compiled as part of the Master's thesis of Jakob Punnerud and Lars
Flem.

## Summary

A total of **10 robustness improvements** were made, addressing:
- Empty graph structures at deeper hierarchy levels
- Heterogeneous data batching when optional fields differ between samples
- Hierarchies shallower than the model's expected depth
- Sparse subtiles that should be skipped at preprocessing time
- Degenerate ground candidates that defeat RANSAC plane fitting
- Empty point clouds propagating through the transform pipeline
- Single-node NAG levels reaching horizontal graph construction
- Empty or degenerate XY tiling inputs

| # | Scenario | File | Approach |
|---|---|---|---|
| 1 | Empty tensor concatenation in `scatter_nearest_neighbor` | `src/utils/scatter.py` | Early guard returning empty tensors |
| 2 | Empty tensor concatenation in `subedges` | `src/utils/graph.py` | Early guard returning empty tensors |
| 3 | KeyError on missing optional fields during batching | `src/data/data.py` | Dynamic key exclusion |
| 4 | NAG hierarchy shallower than the model's down-stage loop expects | `src/models/components/spt.py` | Bounds check in forward pass |
| 5 | Sparse / near-empty subtiles | `src/datasets/base.py` | `.skip` sentinel + dataset-level filtering |
| 6 | RANSAC ValueError on XY-degenerate ground candidates | `src/utils/ground.py` | `try/except` with flat-plane fallback |
| 7 | Empty input to `GridSampling3D` / `QuantizePointCoordinates` | `src/transforms/sampling.py` | Early return with empty tensors |
| 8 | Division-by-zero / empty tile in `SampleXYTiling` | `src/transforms/sampling.py` | Span clamp + empty-input guard (empty-tile fallback later removed; superseded by #5) |
| 9 | Empty input to `KNN`, `Inliers`, `Outliers`, `PointFeatures` | `src/transforms/neighbors.py`, `point.py` | Early return unchanged |
| 10 | Single-node NAG level in horizontal graph construction | `src/transforms/graph.py` | Warn + emit empty `edge_index` |

---

## 1. Empty Edge Index Guard in scatter_nearest_neighbor

**File:** `src/utils/scatter.py`

**Scenario:**
At deeper hierarchy levels with very few superpoints, the radius-based graph
between segments can contain zero edges. `scatter_nearest_neighbor` chunks edges
for memory efficiency, so when `edge_index.shape[1] == 0` no chunks are
produced, the output list is empty, and the final concatenation fails:
```python
candidate = torch.cat([elt[0] for elt in out_list], dim=0)
# RuntimeError: torch.cat(): expected a non-empty list of Tensors
```

**Improvement:**
Added an early guard before the chunking logic that returns correctly-shaped
empty tensors instead:
```python
if edge_index.shape[1] == 0:
    candidate = torch.empty((0, points.shape[1]), dtype=points.dtype, device=points.device)
    candidate_idx = torch.empty((2, 0), dtype=edge_index.dtype, device=edge_index.device)
    return candidate, candidate_idx
```

---

## 2. Empty Edge Index Guard in subedges Function

**File:** `src/utils/graph.py`

**Scenario:**
Same family as #1. `subedges` computes superedge point pairs for the trimmed
graph. When `to_trimmed()` removes all edges (no edges remain after dropping
self-loops and duplicates), the same empty-list concatenation problem appears:
```python
edge_index = torch.cat([elt[0] for elt in out_list], dim=1)
# RuntimeError: torch.cat(): expected a non-empty list of Tensors
```

This path is reached every time #10 emits an empty `edge_index`, so the two
work as a pair.

**Improvement:**
Added an early return for the empty case after the trimming step:
```python
if edge_index.shape[1] == 0:
    ST_pairs = torch.empty((2, 0), dtype=torch.long, device=points.device)
    ST_uid = torch.empty((0,), dtype=torch.long, device=points.device)
    return edge_index, ST_pairs, ST_uid
```

---

## 3. Dynamic Key Exclusion in from_data_list

**File:** `src/data/data.py`

**Scenario:**
When batching samples whose hierarchies differ in structure, optional fields
like `'super_index'` or `'sub'` may exist on some samples and not on others.
PyG's `collate()` assumes every sample carries the same key set, so the missing
field raises:
```python
KeyError: 'sub'
```

This is partly a downstream consequence of #10: when one sample in a batch had
a single-node level that skipped horizontal graph construction, its key set
diverges from siblings that built a full graph.

**Improvement:**
Detect which optional keys are present in every sample and exclude the
remainder before delegating to PyG's collate:
```python
if exclude_keys is None:
    exclude_keys = []
else:
    exclude_keys = list(exclude_keys)

for k in data_list[0].to_dict().keys():
    if k not in exclude_keys and not all(k in d for d in data_list):
        exclude_keys.append(k)
```

Two supporting guards round out the change:
- `if k not in d: continue` in the dtype conversion loops, to skip keys missing
  from specific samples.
- `if k not in batch: continue` before the CSRData post-processing loop — if a
  key was excluded from the batch because it was absent from some samples,
  accessing `batch[k]` would otherwise raise a `KeyError` when converting the
  accumulated CSRData list.

---

## 4. NAG Level Bounds Check in SPT Forward Pass

**File:** `src/models/components/spt.py`

**Scenario:**
The SPT forward pass iterates through a fixed number of down-stages. When an
input produced a shallower hierarchy than the model was configured for (e.g.
because partitioning stopped early on a sparse cloud), the loop tried to access
a NAG level that didn't exist:
```
AssertionError: Level 2 is out of range. NAG has levels range(0, 2)
```

**Improvement:**
Break out of the down-stage loop when the next level would be out of range:
```python
if i_level > nag.end_i_level:
    break
```

Note: with #5 in place, partition-collapsing inputs are filtered upstream and
this branch is rarely if ever exercised in practice. It is retained as a
one-line defensive guard.

---

## 5. Sparse Subtile Skipping with `.skip` Sentinel

**File:** `src/datasets/base.py`

**Scenario:**
National-scale ALS surveys contain tiles that are mostly water or otherwise
nearly empty. After XY tiling, these produce subtiles with too few points to
yield a useful graph hierarchy. They previously caused crashes deep in the
preprocessing pipeline and consumed wall-clock time on every reprocess attempt
because nothing recorded that the tile had been intentionally skipped.

**Improvement:**
A `min_points_per_subtile` parameter on `BaseDataset.__init__`. During
preprocessing, if a subtile is below the threshold, a `.skip` sentinel file is
written next to where the `.h5` would otherwise live:
```python
if n_pts < self._min_points_per_subtile:
    skip_path = cloud_path.replace('.h5', '.skip')
    open(skip_path, 'w').close()
    return
```

Four supporting additions respect the sentinel throughout the dataset:
- **`_skip_path()`** — resolves the expected sentinel path for any
  `cloud_id`/stage.
- **`processed_file_names`** — returns the `.skip` path for skipped tiles so
  PyG's existence check passes.
- **`_valid_processed_paths`** — returns only `.h5` paths for non-skipped
  tiles; used in `__getitem__`, in-memory loading, and class-weight
  computation.
- **`cloud_ids`** — updated to exclude skipped tiles so dataset length and
  indexing remain consistent.

Configured via `min_points_per_subtile` in the datamodule config (set to 0 to
disable). This is the design centerpiece of the robustness changes: with it in
place, several downstream guards (#4, #7, #9) become defensive-only rather than
load-bearing.

---

## 6. RANSAC Fallback for Degenerate Ground Candidates

**File:** `src/utils/ground.py` (`single_plane_model`)

**Scenario:**
On some tiles (sparse terrain, water-dominated areas), the surviving
ground-candidate points share near-identical XY coordinates. sklearn's
`RANSACRegressor` needs at least two points with distinct XY to fit a
`LinearRegression`; when every random sub-sample is degenerate it raises:
```
ValueError: RANSAC could not find a valid consensus set.
```
First observed on tile `32-1-510-131-63.laz` after ~8 hours of preprocessing
(tile 396/588). The pre-existing guard `if len(pos) < 3` only protected
against too-few points, not against ≥ 3 XY-degenerate points.

Note: this scenario is independent of total point count, so #5 does not
preclude it — a 50k-point subtile over flat water can still have collinear
ground-class candidates.

**Improvement:**
Wrap the RANSAC fit in `try/except ValueError` with a flat-plane fallback at
the mean Z of the candidate ground points:
```python
try:
    ransac = RANSACRegressor(...).fit(xy, z)
    def predict_elevation(pos_query): ...
except ValueError:
    z_mean = float(z.mean())
    print(f"WARNING: RANSAC could not find a valid consensus set. "
          f"Falling back to a flat ground plane at z={z_mean:.3f}.")
    def predict_elevation(pos_query):
        return pos_query[:, 2] - z_mean
```

---

## 7. Empty Input Guards in GridSampling3D and QuantizePointCoordinates

**File:** `src/transforms/sampling.py`

**Scenario:**
`torch_cluster.grid_cluster` (used internally by both transforms) does not
support empty input tensors, so a 0-point cloud reaching either transform
would crash.

**Improvement:**
Early-return guards at the top of each transform's `_process` method. For
`GridSampling3D`, the input data is returned unchanged with `coords` and
`grid_size` set. For `QuantizePointCoordinates`, an empty NAG is constructed
and returned.

With #5 in place this guard sits downstream of the `.skip` filter and is
retained as defensive programming rather than as the primary line of defence.

---

## 8. Span Clamp and Empty-Input Guard in SampleXYTiling

**File:** `src/transforms/sampling.py`

**Scenario:**
Two distinct failure modes in `SampleXYTiling._process`:
1. When all points share the same XY coordinate (zero spatial span), dividing
   by the span produced NaN/inf, placing all points in tile (0, 0) and leaving
   all requested tiles empty.
2. When the requested tile `(x, y)` contained no points, the empty selection
   crashed downstream.

`SampleXYTiling` runs *before* the `.skip` check in [base.py:883-903](src/datasets/base.py#L883-L903),
so #5 does not preclude these issues — they have to be handled here.

**Improvement:**
- **Span clamp:** `torch.where(span > 0, span, ones_like(span))` to avoid
  division by zero.
- **Upper-boundary clip:** Use `1 - eps` as the clip max so points on the
  upper edge stay in the last tile.
- **Empty input guard:** Return immediately if the input cloud has 0 points.
- ~~**Empty-tile fallback:** If `idx.numel() == 0`, fall back to the tile
  with the most points (`counts.argmax()`), with a warning log.~~ *(removed —
  see below)*

**Removed sub-improvement:** *Empty-tile fallback*

An earlier version of this change also added a `counts.argmax()` fallback
that, when the requested `(x, y)` tile had no points, returned the
most-populated tile instead. This was later removed: with #5 in place, empty
subtiles are caught downstream and the most-populated-quadrant substitution
became unnecessary. The silent tile substitution also masked legitimate
configuration issues, so removing it makes preprocessing failures more
visible.

---

## 9. Empty Point Cloud Guards in Neighbor and Feature Transforms

**Files:** `src/transforms/neighbors.py`, `src/transforms/point.py`

**Scenario:**
When a 0-point cloud reached `KNN`, `Inliers`, `Outliers`, or `PointFeatures`,
each crashed because their internal routines assumed at least one point.

**Improvement:**
Early-return guards at the top of each `_process` method:
- `KNN`: initialises empty `neighbor_index`, `neighbor_distance` (and
  optionally `neighbors` as a `CSRData`) then returns.
- `Inliers` / `Outliers`: returns `data` unchanged.
- `PointFeatures`: returns `data` unchanged after logging a warning.

With #5 in place these guards sit downstream of the `.skip` filter and are
retained as defensive programming.

---

## 10. Single-Node Level Guard in Horizontal Graph Construction

**File:** `src/transforms/graph.py` (`_horizontal_graph_by_radius_for_single_level`)

**Scenario:**
When a NAG level contained only one superpoint node, the horizontal graph
builder raised:
```python
ValueError: Input NAG only has 1 node at level=X. Cannot compute radius-based horizontal graph.
```
This is independent of total point count: even a non-sparse subtile can have
a deeper hierarchy level that collapses to one node, depending on the
partitioning. Empirically this warning fires constantly on Viken and
Trondheim runs.

**Improvement:**
Replaced the `raise ValueError` with a logged warning, then set
`data.edge_index` to an empty `(2, 0)` tensor and `data.edge_attr = None`
before returning the NAG unchanged:
```python
log.warning(f"NAG only has {num_nodes} node(s) at level={i_level}. Skipping horizontal graph construction.")
data.edge_index = torch.zeros((2, 0), dtype=torch.long, device=data.pos.device)
data.edge_attr = None
nag._list[i_level] = data
return nag
```

The empty `edge_index` produced here is absorbed downstream by #2.
