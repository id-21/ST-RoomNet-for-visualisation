I saw the output and it seems like only the pixels labelled as wall were multiplied with the data from the
 shadow model. Was this by design? If so, please refernece the lines in the various Markdown files and the 
lines of code that implement this. 

⏺ Yes, this was absolutely by design. The shadow overlay is intentionally applied only to pixels identified
  as walls (not ceiling or floor). Let me reference the specific lines that document and implement this:

  Planning Documents

  planning/shadow-darkening-visualization-plan.md:
  - Lines 185-196 - Main pipeline explicitly shows per-wall iteration:
  # Step 4: Apply shadow with multiple clamp values
  white_results = {}
  for clamp in CLAMP_VALUES:
      white_with_shadow = white_canvas.copy()
      for wall in layout.walls:
          if not wall.is_visible:
              continue
          # Get wall mask and apply shadow
          wall_mask = refined_masks[wall_label]
          shadowed = apply_shadow_to_wall(white_with_shadow, shadow_resized,
                                          wall_mask, ambient_clamp=clamp)
          white_with_shadow[wall_mask] = shadowed[wall_mask]

  - Line 195 - The critical line: white_with_shadow[wall_mask] = shadowed[wall_mask] - only wall pixels are
  replaced

  planning/shadow-overlay-experiment-plan.md:
  - Lines 102-111 - Step 6 explicitly documents per-wall shadow application:
  # Step 6: Apply Shadow to Each Wall
  for wall in layout.walls:
      # Get refined mask for this wall
      wall_mask = refined_masks[wall.wall_id]
      # Apply shadow to this wall region only
      shadowed_wall = apply_shadow_to_wall(...)
      # Composite shadowed walls back onto original image

  Code Implementation

  visualize_shadow_darkening.py:
  - Lines 533-566 - Main loop applies shadow per wall:
    - Line 536: for wall in layout.walls: - iterates only through detected walls
    - Line 554: wall_mask = refined_masks[wall_label] - gets boolean mask for this wall
    - Line 557-563: Calls apply_shadow_to_wall() with the specific wall mask
    - Line 566: white_with_shadow[wall_mask] = shadowed_wall[wall_mask] - Only replaces pixels where mask is
   True

  test_shadow_overlay.py (original experiment):
  - Lines 324-366 - Same pattern:
    - Line 324: for wall in layout.walls:
    - Line 353: wall_mask = refined_masks[wall_label]
    - Line 365: wallpaper_with_shadow[wall_mask] = shadowed_wall[wall_mask] - Per-wall masking

  apply_shadow_to_wall() function (visualize_shadow_darkening.py lines 129-192):
  - Line 129 - Function signature takes wall_mask parameter
  - Line 157 - Uses wall mask for normalization: wall_shadow_pixels = shadow_exr[wall_mask].mean(axis=1)
  - Lines 166-178 - Per-wall baseline normalization uses min/max from THAT wall's pixels only

  Why This Design?

  This is intentional because:

  1. ST-RoomNet's purpose: Detects walls (labels 1-3) vs ceiling (0) and floor (4)
  2. Wallpaper visualization context: The entire V3 pipeline is designed to apply wallpaper to WALLS only
  3. Per-wall normalization: Each wall gets its own shadow baseline (2nd/98th percentile) because different
  wall orientations receive different lighting
  4. Semantic correctness: Shadows from irradiance should only affect vertical wall surfaces, not horizontal
   surfaces (floor/ceiling)

  The white canvas shows white in non-wall regions (ceiling, floor) because those pixels are never part of
  any wall_mask and therefore never modified by the shadow application loop.
