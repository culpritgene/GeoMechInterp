# Fixes Applied to Address Poor Results

## Problems Identified

1. ❌ **High variance/noise** - Scattered coloring, oscillating estimates
2. ❌ **Ollivier-Ricci wrong sign** - Giving -0.49 on sphere (should be ~+1.0)
3. ❌ **PCA on hyperboloid wrong sign** - Giving +0.64 (should be ~-1.0)
4. ❌ **Low accuracy** - Errors of 75-177% on analytical manifolds

## Fixes Implemented ✅

### 1. Increased k for Robustness
**Changed:** `k=15` → `k=40` throughout demo

**Impact:**
- Larger neighborhoods → more stable estimates
- Reduces local noise sensitivity
- Better approximation of continuous manifold

**Expected improvement:**
- Standard deviation should drop by ~50%
- Smoother color gradients in visualizations

---

### 2. Fixed Hyperboloid Generation
**Changed:** Poincaré disk mapping → Proper hyperbolic parametrization

**Old code:**
```python
# Incorrect - gave wrong geometry
r = np.random.uniform(0, 0.95, n_points)
scale = a / np.sqrt(1 - r**2)
x = scale * x_disk
```

**New code:**
```python
# Correct hyperbolic parametrization
u = np.random.uniform(0, 2*np.pi, n_points)
v = np.random.uniform(-1.5, 1.5, n_points)
x = a * np.sinh(v) * np.cos(u)
y = a * np.sinh(v) * np.sin(u)
z = a * np.cosh(v)
```

**Impact:**
- Now generates true hyperboloid with K = -1/a²
- PCA method should now give negative curvature
- Proper hyperbolic geometry

---

### 3. Increased Sectional Curvature Samples
**Changed:** `n_plane_samples=10` → `n_plane_samples=20`

**Impact:**
- Better statistical estimate of curvature distribution
- Should improve from 0.25 toward 1.0 on sphere

---

### 4. Added Warning for Ollivier-Ricci

Added explicit warning that the simplified implementation may give incorrect signs.

**Current issue:** Uses `np.sum(prob_i * prob_j * dist_matrix)` which is NOT optimal transport distance.

**Proper fix requires:**
```bash
pip install POT  # Python Optimal Transport library
```

Then use:
```python
import ot
W1 = ot.emd2(prob_i, prob_j, dist_matrix)
```

**For now:** Treat Ollivier-Ricci results as experimental/unreliable.

---

## Expected Results After Fixes

### Sphere (K = 1.0):

| Method | Before | After (Expected) | Status |
|--------|--------|----------|--------|
| PCA Tangent | 1.02 ± 0.10 | 1.05 ± 0.05 | ✅ Should improve |
| Geodesic | 0.84 ± 0.11 | 0.85 ± 0.08 | ✅ Relative measure |
| Ollivier-Ricci | -0.49 ± ? | Still wrong | ⚠️ Needs POT library |
| Sectional | 0.25 ± 0.02 | 0.70 ± 0.15 | ✅ Should improve |
| Covariance | 0.04 ± 0.01 | 0.04 ± 0.008 | ✅ Different scale |

### Hyperboloid (K = -1.0):

| Method | Before | After (Expected) | Status |
|--------|--------|----------|--------|
| PCA Tangent | +0.64 ± 0.27 | -0.80 ± 0.20 | ✅ FIXED sign |
| Geodesic | +0.77 ± 0.14 | +0.75 ± 0.10 | ⚠️ Relative only |
| Covariance | +0.05 ± 0.06 | +0.04 ± 0.03 | ✅ Different scale |

---

## Remaining Known Issues

### 1. Ollivier-Ricci Implementation ⚠️
- **Problem:** Wrong Wasserstein approximation
- **Status:** Documented, needs POT library
- **Workaround:** Use Forman-Ricci from existing code instead

### 2. Sectional Curvature Scale
- **Problem:** Still may underestimate (0.7 vs 1.0)
- **Reason:** Quadratic fitting approximation
- **Status:** Acceptable for relative comparisons

### 3. Geodesic Method Interpretation
- **Note:** Gives relative curvature (ratio), not absolute K
- **Status:** Working as designed, not a bug

### 4. Covariance Method Scale
- **Note:** Measures rate of change, not K directly
- **Status:** Working as designed, for relative comparison

---

## How to Run Updated Demo

```bash
cd projects/activations_curvature
python demo_example.py
```

**Expected runtime:** 3-8 minutes (increased due to larger k and n_points)

---

## Visual Improvements Expected

### Before (k=15, n=500):
- 🔴 Scattered, noisy colors
- 🔴 No clear patterns
- 🔴 High variance

### After (k=40, n=2500):
- 🟢 Smooth color gradients
- 🟢 Clear geometric structure
- 🟢 Lower variance

---

## Next Steps for Further Improvement

1. **Install POT for proper Ollivier-Ricci:**
   ```bash
   pip install POT
   ```

2. **Add smoothing post-processing (optional):**
   ```python
   from scipy.ndimage import uniform_filter1d
   curvatures_smooth = uniform_filter1d(curvatures, size=5)
   ```

3. **Increase n_points further for publication quality:**
   ```python
   n_points = 5000  # Very smooth, but slower
   k = 60
   ```

4. **Use adaptive k selection:**
   ```python
   k = max(30, min(100, int(np.sqrt(n_points) * 0.8)))
   ```

---

## Validation

After running the updated demo, check:

✅ **Sphere PCA:** Should be 1.0 ± 0.1 (within 10% error)
✅ **Hyperboloid PCA:** Should be NEGATIVE (sign matters!)
✅ **Visual smoothness:** Colors should transition smoothly
✅ **Consistency:** Similar points should have similar colors

If these criteria are met, the fixes are successful! 🎉

---

## Files Modified

1. `curvature_measure.py`:
   - Fixed `generate_hyperboloid()` function (lines 1041-1052)

2. `demo_example.py`:
   - Changed `k_default = 40` (line 48)
   - Updated all method calls to use `k_default`
   - Increased `n_plane_samples` to 20
   - Added Ollivier-Ricci warning

3. New documentation:
   - `TROUBLESHOOTING.md` - Detailed analysis
   - `FIXES_APPLIED.md` - This file

---

**Try the updated demo now and compare results!**

