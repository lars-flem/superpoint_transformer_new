# Model Bug Fixes and Changes

This document outlines all the bug fixes and improvements made to the SuperPoint Transformer model to handle edge cases and improve robustness.

## Summary

A total of **4 critical bug fixes** were implemented to address issues with:
- Empty graph structures
- Heterogeneous data batching
- Hierarchical model architecture mismatch

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
# Handle empty edge_index early
if edge_index.shape[1] == 0:
    candidate = torch.empty((0, points.shape[1]), dtype=points.dtype, device=points.device)
    candidate_idx = torch.empty((2, 0), dtype=edge_index.dtype, device=edge_index.device)
    return candidate, candidate_idx
```

**Impact:**
- ✅ Allows graceful handling of disconnected graph components
- ✅ Prevents crashes when processing sparse or minimal graphs
- ✅ Returns correctly-shaped empty tensors for downstream processing
- ⚡ Minor performance benefit: skips unnecessary computation for empty edge lists

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
Added early return for empty edges **after** the trimming step:
```python
# Handle empty edge_index early
if edge_index.shape[1] == 0:
    ST_pairs = torch.empty((2, 0), dtype=torch.long, device=points.device)
    ST_uid = torch.empty((0,), dtype=torch.long, device=points.device)
    return edge_index, ST_pairs, ST_uid
```

**Impact:**
- ✅ Handles cases where all edges are self-loops or duplicates after trimming
- ✅ Prevents cascading failures in superedge feature computation
- ✅ Returns consistent tensor shapes for batching operations
- 📊 Occurs when NAG levels have very few nodes or isolated components

---

## 3. Dynamic Key Exclusion in from_data_list

**File:** `src/data/data.py` (lines 169-180, 219-223)

**Issue:**
When batching data samples from different points in the hierarchy, some samples would have optional fields (like `'super_index'`) while others wouldn't. The batching code tried to access these fields uniformly, causing:
```python
KeyError: 'super_index'
```

**Root Causes:**
1. Some graph levels don't create superedges (e.g., single-node levels skip horizontal graph construction)
2. Different data samples have different optional fields set depending on their processing path
3. PyG's `collate()` function expects all samples to have the same keys

**Fix:**
Dynamically detect and exclude missing keys **before** calling PyG's collate function:
```python
# Dynamically exclude keys that don't exist in all data objects
if exclude_keys is None:
    exclude_keys = []
else:
    exclude_keys = list(exclude_keys)

for k in data_list[0].to_dict().keys():
    if k not in exclude_keys and not all(k in d for d in data_list):
        exclude_keys.append(k)
```

Also added checks in the two dtype conversion loops to skip keys that don't exist in specific samples:
```python
if k not in d:
    continue
```

**Impact:**
- ✅ Enables batching of heterogeneous data samples
- ✅ Supports variable graph hierarchies within a batch
- ✅ Gracefully skips optional fields instead of crashing
- 📊 Essential for datasets where graph structure varies across samples
- 🔧 Maintains type safety for fields that do exist

---

## 4. NAG Level Bounds Check in SPT Forward Pass

**File:** `src/models/components/spt.py` (lines 823-826)

**Issue:**
The SuperPoint Transformer model was designed for a fixed number of hierarchical levels, but when data had fewer levels than expected (e.g., due to small point clouds stopping hierarchy expansion early), the model would crash:
```
AssertionError: Level 2 is out of range. NAG has levels range(0, 2)
```

**Root Cause:**
The forward loop iterated through all expected down-stages:
```python
for i_stage, (stage, node_mlp, ...) in enumerate(self.down_stages):
    i_level = i_stage + 1 + self.nano
    # Tries to access nag[2] but NAG only has levels 0-1
```

The assertion happened after attempting to access the non-existent level.

**Fix:**
Added a guard check **before** accessing the level:
```python
# Skip this stage if the level doesn't exist in the NAG
if i_level > nag.end_i_level:
    break
```

**Impact:**
- ✅ Gracefully handles NAGs with fewer levels than model capacity
- ✅ Model adapts to variable-depth hierarchies
- ✅ All accumulated down outputs are still used for up-sampling
- ✅ Prevents crashes on small or sparse point clouds
- 📊 Critical for datasets with variable-sized scenes
- 🎯 Maintains full forward-backward compatibility

---

## Data Flow Impact Diagram

```
Raw LiDAR Data (Viken2022)
    ↓
Graph Construction (Empty edge guards) [Fixes #1, #2]
    ↓
Batch Creation (Key exclusion) [Fix #3]
    ↓
Model Forward Pass (Level bounds check) [Fix #4]
    ↓
Training/Validation Complete ✓
```

---

## Testing Recommendations

To verify these fixes work correctly, test the following scenarios:

1. **Small Point Clouds:** Test with point clouds that might result in single-node levels
2. **Sparse Graphs:** Test with scenes that have isolated points or disconnected components
3. **Variable Batch Composition:** Mix samples with different hierarchy depths in a single batch
4. **Edge Cases:** Test with empty tiles or very sparse data

---

## Error Prevention Summary

| Fix # | Error Type | Severity | Prevention Method |
|-------|-----------|----------|-------------------|
| 1 | Empty Tensor Concatenation | High | Early guard check |
| 2 | Empty Tensor Concatenation | High | Early guard check |
| 3 | KeyError on Missing Fields | High | Dynamic exclusion |
| 4 | AssertionError on Out-of-Range Level | High | Bounds checking |

---

## Notes

- All fixes maintain backward compatibility with existing code
- No changes to model architecture or training procedures
- Fixes are defensive programming patterns that handle edge cases
- Documentation updated in `/memories/repo/preprocessing-notes.md`
