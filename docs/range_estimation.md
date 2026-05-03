# Geometric Range Estimation

We are not using supervised range regression for the first training run.

Instead, local inference should estimate range with pinhole-camera geometry:

```text
distance_m = focal_length_px * real_object_size_m / observed_bbox_size_px
```

Required runtime inputs:

- Camera intrinsics, especially focal length in pixels.
- Detection bbox width/height in pixels.
- Drone physical-size prior.

The size prior can start as a conservative interval across common drone classes, then improve when we add a drone-type classifier:

```text
small quad:       ~0.35 m width
consumer quad:    ~0.60 m width
large quad:       ~1.20 m width
small fixed wing: ~1.50 m width
```

This should produce an interval or confidence band, not a fake precise number. The allocation layer can use the interval midpoint for ranking while showing uncertainty in the demo.
