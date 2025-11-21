# **Milestone 1 — Precompute Skeleton Vectors (Core Implementation)**

### **1.1 Add vector-building helper**

* [x] Create `_build_skeleton_vectors` in `napari_backend.py`
* [x] Accept `(BodypartData, Environment)` as inputs
* [x] Collect all needed bodypart coordinates from `bodypart_data.bodyparts`
* [x] Apply `_transform_coords_for_napari` exactly once per bodypart
* [x] Precompute per-frame `t` index (use float32 or int32)
* [x] Allocate final vectors array shaped `(n_frames * n_edges, 2, 3)`
* [x] Fill vectors in contiguous batches (avoid Python loops over frames)
* [x] Build `features={"edge_name": ...}`
* [x] Return `(vectors, features)`

### **1.2 Guard against empty/missing skeleton**

* [x] Early-return empty array+empty features if `bodypart_data.skeleton` is None or empty

### **1.3 Follow project constraints**

* [x] Maintain NumPy-first operations (per CLAUDE.md)
* [x] Respect environment coordinate → napari coordinate transform rules correctly
* [x] Use type hints + NumPy docstrings

---

# **Milestone 2 — Replace Shapes Skeleton With Vectors Layer**

### **2.1 Update `_render_bodypart_overlay`**

* [ ] Remove old Shapes-based `_create_skeleton_frame_data`
* [ ] Remove Shapes-based `_setup_skeleton_update_callback`
* [ ] Insert call to `_build_skeleton_vectors`
* [ ] Add a single `viewer.add_vectors(...)` layer using the returned data
* [ ] Apply `bodypart_data.skeleton_color` + `skeleton_width`
* [ ] Add features for future styling (`edge_name`)
* [ ] Append this vectors layer to the return list of layers

### **2.2 Align with existing bodypart Points layer**

* [ ] Ensure Points layer creation remains unchanged
* [ ] Ensure coordinate transforms are not duplicated
* [ ] Validate that time dimension is interpreted correctly by napari

### **2.3 Remove per-frame callbacks**

* [ ] Completely delete skeleton-related dims events
* [ ] Ensure nothing in the viewer tries to update skeleton during playback

---

# **Milestone 3 — Downsampling & Large Dataset Support**

### **3.1 Downsample skeleton frames**

* [ ] Add optional parameter `skeleton_step` (default 1)
* [ ] Apply subsampling to frame indices before precomputation
* [ ] Slice bodypart trajectories and time vector accordingly

### **3.2 Optional memory optimization**

* [ ] Allow float32 vectors storage
* [ ] Add warning when `(n_frames * n_edges)` exceeds threshold

### **3.3 Runtime skeleton visibility toggle**

* [ ] Add a boolean `show_skeleton` keyword to the napari backend
* [ ] Skip vector generation entirely when disabled

---

# **Milestone 4 — Cleanup and Removal of Old Code Path**

### **4.1 Remove dead code**

* [ ] Delete `_create_skeleton_frame_data`
* [ ] Delete `_setup_skeleton_update_callback`
* [ ] Remove any debug/perf events tied to skeleton callback

### **4.2 Update docstrings & comments**

* [ ] Update backend docs: skeleton → vectors, no per-frame rebuild
* [ ] Update overlay system docs to reflect new generation path
* [ ] Rewrite napari backend notes to comply with CLAUDE.md formatting

### **4.3 Remove Shapes skeleton references**

* [ ] Update tests referring to Shapes-based skeleton overlays
* [ ] Update developer guide and CLAUDE.md if needed

---

# **Milestone 5 — Testing & Validation**

### **5.1 Unit tests**

* [x] Test `_build_skeleton_vectors` directly with synthetic data
* [x] Verify vector shape: `(n_frames * n_edges, 2, 3)`
* [x] Confirm correct time stamping
* [x] Validate that coordinate transforms match existing logic

### **5.2 Napari rendering test**

* [ ] Smoke-test vector skeleton visualization manually
* [ ] Ensure vectors respect napari’s dims/time axis

### **5.3 Performance validation**

* [ ] Benchmark large bodypart datasets:

  * [ ] 10k frames
  * [ ] 100k frames (with downsampling)
* [ ] Confirm no UI freezing during playback
* [ ] Verify that switching layers on/off has no callback cost

### **5.4 Regression tests**

* [ ] Ensure PositionOverlay and HeadDirectionOverlay still function
* [ ] Confirm HTML/Video backends are unaffected

---

# **Milestone 6 — Integration & Release Prep**

### **6.1 Integrate with `animate_fields` pipeline**

* [ ] Ensure `_convert_overlays_to_data` remains unchanged
* [ ] Verify `OverlayData` remains pickle-stable
* [ ] Confirm napari remains fully functional for all overlays

### **6.2 Update documentation**

* [ ] Add examples using new skeleton vectors
* [ ] Add explanation of bake-time vs per-frame update cost
* [ ] Add advice for large datasets (downsampling, toggles)
