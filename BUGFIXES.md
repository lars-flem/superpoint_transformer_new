# Model Bug Fixes and Changes

This document outlines all the bug fixes and improvements made to the SuperPoint Transformer model to handle edge cases and improve robustness.

## Summary

A total of **10 bug fixes** were implemented to address issues with:
- Empty graph structures
- Heterogeneous data batching
- Hierarchical model architecture mismatch
- Sparse subtile skipping during preprocessing
- Degenerate ground candidates causing RANSAC failure
- Empty point clouds propagating through the transform pipeline
- Single-node NAG levels crashing horizontal graph construction
- Empty/degenerate XY tiling producing empty subtiles

| Fix # | Error Type | File | Prevention Method |
|-------|-----------|------|-------------------|
| 1 | Empty tensor concatenation in `scatter_nearest_neighbor` | `src/utils/scatter.py` | Early guard check |
| 2 | Empty tensor concatenation in `subedges` | `src/utils/graph.py` | Early guard check |
| 3 | KeyError on missing fields in batching | `src/data/data.py` | Dynamic key exclusion |
| 4 | AssertionError on out-of-range NAG level | `src/models/components/spt.py` | Bounds check |
| 5 | Crash on sparse/empty subtile | `src/datasets/base.py` | `.skip` sentinel + dataset filtering |
| 6 | RANSAC ValueError on degenerate ground candidates | `src/utils/ground.py` | `try/except` with flat-plane fallback |
| 7 | Crash in `GridSampling3D` / `QuantizePointCoordinates` on empty input | `src/transforms/sampling.py` | Early return with empty tensors |
| 8 | Division-by-zero / empty tile in `SampleXYTiling` | `src/transforms/sampling.py` | Span clamp + fallback to most-populated tile |
| 9 | Crash in `KNN`, `Inliers`, `Outliers`, `PointFeatures` on 0-point cloud | `src/transforms/neighbors.py`, `point.py` | Early return unchanged |
| 10 | ValueError in horizontal graph construction for single-node level | `src/transforms/graph.py` | Warn + set empty edge_index |

---

## 1. Empty Edge Index Guard in scatter_nearest_neighbor

**File:** `src/utils/scatter.py` (lines 130-135)

**Issue:**
When processing graph levels with no edges (e.g., very small point clouds or single-node levels), the `scatter_nearest_neighbor` function would create an empty chunking list and crash when trying to concatenate tensors from an empty list:
```python
candidate = torch.cat([elt[0] for elt in out_list], dim=0)
# RuntimeError: torch.cat(): expected a non-empty list of Tensors
```

**Root Cause:**
The function chunked edges for memory efficiency, but when `edge_index.shape[1] == 0`, no chunks were created, leaving `out_list` empty. The concatenation then failed.

**Fix:**
Added an early guard before the chunking logic:
```python
if edge_index.shape[1] == 0:
    candidate = torch.empty((0, points.shape[1]), dtype=points.dtype, device=points.device)
    candidate_idx = torch.empty((2, 0), dtype=edge_index.dtype, device=edge_index.device)
    return candidate, candidate_idx
```

---

## 2. Empty Edge Index Guard in subedges Function

**File:** `src/utils/graph.py` (lines 163-167)

**Issue:**
Similar to fix #1, the `subedges` function attempts to compute superedge pairs between segments. When `to_trimmed()` produces an empty edge list (no edges remain after removing self-loops and duplicates), the function crashes:
```python
edge_index = torch.cat([elt[0] for elt in out_list], dim=1)
# RuntimeError: torch.cat(): expected a non-empty list of Tensors
```

**Fix:**
Added early return for empty edges after the trimming step:
```python
if edge_index.shape[1] == 0:
    ST_pairs = torch.empty((2, 0), dtype=torch.long, device=points.device)
    ST_uid = torch.empty((0,), dtype=torch.long, device=points.device)
    return edge_index, ST_pairs, ST_uid
```

---

## 3. Dynamic Key Exclusion in from_data_list

**File:** `src/data/data.py` (lines 169-180, 219-223)

**Issue:**
When batching data samples from different points in the hierarchy, some samples would have optional fields (like `'super_index'`) while others wouldn't. The batching code tried to access these fields uniformly, causing:
```python
KeyError: 'super_index'
```

**Root Cause:**
Some graph levels don't create superedges (e.g., single-node levels skip horizontal graph construction), so different samples have different optional fields. PyG's `collate()` expects all samples to have the same keys.

**Fix:**
Dynamically detect and exclude missing keys before calling PyG's collate function:
```python
if exclude_keys is None:
    exclude_keys = []
else:
    exclude_keys = list(exclude_keys)

for k in data_list[0].to_dict().keys():
    if k not in exclude_keys and not all(k in d for d in data_list):
        exclude_keys.append(k)
```

Also added two further guards:
- `if k not in d: continue` in the dtype conversion loops to skip keys missing from specific samples.
- `if k not in batch: continue` before the CSRData post-processing loop (line 1257) — if a key was excluded from the batch because it was absent from some samples, accessing `batch[k]` would raise a `KeyError` when trying to convert the accumulated CSRData list.

---

## 4. NAG Level Bounds Check in SPT Forward Pass

**File:** `src/models/components/spt.py` (lines 823-826)

**Issue:**
The model was designed for a fixed number of hierarchical levels, but when data had fewer levels than expected (e.g., due to small point clouds stopping hierarchy expansion early), the model would crash:
```
AssertionError: Level 2 is out of range. NAG has levels range(0, 2)
```

**Root Cause:**
The forward loop iterated through all expected down-stages, trying to access NAG levels that didn't exist for sparse inputs.

**Fix:**
Added a guard check before accessing the level:
```python
if i_level > nag.end_i_level:
    break
```

---

## 5. Sparse Subtile Skipping with `.skip` Sentinel

**File:** `src/datasets/base.py`

**Issue:**
When tiling large LiDAR tiles (e.g. over water bodies or empty terrain) into subtiles, some subtiles would have too few points to produce a valid graph hierarchy. This caused crashes deep in the preprocessing pipeline and required the entire tile to be reprocessed on each run.

**Fix:**
Added a `min_points_per_subtile` parameter to `BaseDataset.__init__`. During preprocessing, if a subtile has fewer points than the threshold, a `.skip` sentinel file is written in place of the `.h5`:
```python
if n_pts < self._min_points_per_subtile:
    skip_path = cloud_path.replace('.h5', '.skip')
    open(skip_path, 'w').close()
    return
```

Three supporting additions respect the sentinel throughout the dataset:
- **`_skip_path()`** — resolves the expected sentinel path for any cloud_id/stage.
- **`processed_file_names`** — returns the `.skip` path for skipped tiles so PyG's existence check passes.
- **`_valid_processed_paths`** — returns only `.h5` paths for non-skipped tiles; used in `__getitem__`, in-memory loading, and class weight computation.
- **`cloud_ids`** — updated to exclude skipped tiles so dataset length and indexing are consistent.

Controlled via `min_points_per_subtile` in the datamodule config (set to 0 to disable).

---

## 6. RANSAC Fallback for Degenerate Ground Candidates

**File:** `src/utils/ground.py` (`single_plane_model`, line ~140)

**Issue:**
On some tiles (e.g. sparse terrain or water tiles), all surviving ground-candidate points share near-identical XY coordinates. sklearn's `RANSACRegressor` needs at least 2 points with distinct XY to fit a `LinearRegression` model; when every random sub-sample is degenerate, it raises:
```
ValueError: RANSAC could not find a valid consensus set.
```
First observed on tile `32-1-510-131-63.laz` after ~8 hours of preprocessing (tile 396/588).

**Root Cause:**
The existing guard (`if len(pos) < 3`) only protected against too-few points, not against ≥ 3 XY-degenerate points.

**Fix:**
Wrapped the RANSAC fit in a `try/except ValueError` with a flat-plane fallback at the mean Z of the candidate ground points:
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

Note: a `UndefinedMetricWarning: R^2 score is not well-defined with less than two samples` from sklearn is a related but benign warning — not the same crash.

---

## 7. Empty Input Guards in GridSampling3D and QuantizePointCoordinates

**File:** `src/transforms/sampling.py`

**Issue:**
`torch_cluster.grid_cluster` (used internally by both transforms) does not support empty input tensors. When a 0-point cloud reached either transform, it crashed.

**Fix:**
Added early-return guards at the top of each transform's `_process` method. For `GridSampling3D`, the data is returned unchanged (with `coords` and `grid_size` set). For `QuantizePointCoordinates`, an empty NAG is constructed and returned.

---

## 8. Division-by-Zero and Empty-Tile Fallback in SampleXYTiling

**File:** `src/transforms/sampling.py`

**Issue:**
Two separate failure modes in `SampleXYTiling._process`:
1. When all points share the same XY coordinate (zero spatial span), dividing by the span produced NaN/inf, placing all points in tile (0, 0) and leaving all requested tiles empty.
2. When the requested tile `(x, y)` contained no points, the empty selection crashed downstream.

**Fix:**
- **Span clamp:** `torch.where(span > 0, span, ones_like(span))` to avoid division by zero.
- **Upper-boundary clip:** Use `1 - eps` as the clip max so points on the upper edge stay in the last tile.
- **Empty-tile fallback:** If `idx.numel() == 0`, fall back to the tile with the most points (`counts.argmax()`), with a warning log.
- **Empty input guard:** Return immediately if the input cloud has 0 points.

---

## 9. Empty Point Cloud Guards in Neighbor and Feature Transforms

**Files:** `src/transforms/neighbors.py`, `src/transforms/point.py`

**Issue:**
When an empty point cloud (0 points) reached `KNN`, `Inliers`, `Outliers`, or `PointFeatures`, each crashed because their internal routines assumed at least one point.

**Fix:**
Added early-return guards at the top of each `_process` method:
- `KNN`: initialises empty `neighbor_index`, `neighbor_distance` (and optionally `neighbors` as a CSRData) then returns.
- `Inliers` / `Outliers`: returns `data` unchanged.
- `PointFeatures`: returns `data` unchanged after logging a warning.

---

## 10. Single-Node Level Guard in Horizontal Graph Construction

**File:** `src/transforms/graph.py` (`_horizontal_graph_by_radius_for_single_level`)

**Issue:**
When a NAG level contained only one superpoint node, the horizontal graph builder raised:
```python
ValueError: Input NAG only has 1 node at level=X. Cannot compute radius-based horizontal graph.
```

**Fix:**
Replaced the `raise ValueError` with a logged warning, then set `data.edge_index` to an empty `(2, 0)` tensor and `data.edge_attr = None` before returning the NAG unchanged:
```python
log.warning(f"NAG only has {num_nodes} node(s) at level={i_level}. Skipping horizontal graph construction.")
data.edge_index = torch.zeros((2, 0), dtype=torch.long, device=data.pos.device)
data.edge_attr = None
nag._list[i_level] = data
return nag
```

The empty edge_index produced here is handled downstream by fix #2.
