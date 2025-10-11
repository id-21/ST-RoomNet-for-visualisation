# Normal Detection & Shading - Research Plan

**Research Date:** Oct 6, 2025
**Research Type:** Enhancement Investigation
**Researcher:** Human/Claude

---

## Problem Statement

**Issue 1: Normal Vector Accuracy**
- Current normal vectors from Option A (direct resize) may be inaccurate due to aspect ratio distortion
- Hypothesis: Option B (letterbox padding) preserves geometry better → more accurate normals

**Issue 2: No Shading Implementation**
- Wallpaper appears flat without lighting/depth perception
- Need physically-based shading for realistic visualization

---

## Experiment Plan

### Part 1: Normal Detection Comparison
**Test:** Compare normal vectors from Option A vs Option B preprocessing
- Option A: Direct resize 400×400 (current, distorted)
- Option B: Letterbox padding → 400×400 (aspect-preserved)

**Validation:**
- Visual inspection: Do walls align with expected orientation?
- Quantitative: Measure angular deviation from ground truth (manual annotation)

### Part 2: Shading Implementation
**Approach:** Lambertian shading with directional light
- Use normal vectors (n) from best option (A or B)
- Define light direction vector (L)
- Apply shading: `intensity = max(0, n · L)`
- Modulate wallpaper RGB by intensity

**Parameters to experiment:**
- Light position (overhead, corner, adjustable)
- Ambient occlusion factor
- Specular highlights (optional)