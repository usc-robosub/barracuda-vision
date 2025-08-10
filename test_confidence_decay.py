#!/usr/bin/env python3
"""
Test script to demonstrate confidence-based track loss
"""

# Calculate how many frames it takes for confidence to drop below threshold
def frames_to_loss(survival_factor, min_confidence_threshold):
    """Calculate frames needed for confidence to drop below threshold"""
    import math
    return math.log(min_confidence_threshold) / math.log(survival_factor)

# Test different configurations
configs = [
    {"survival_factor": 0.97, "threshold": 0.1},  # Default config
    {"survival_factor": 0.95, "threshold": 0.1},  # Faster decay
    {"survival_factor": 0.99, "threshold": 0.1},  # Slower decay
    {"survival_factor": 0.97, "threshold": 0.05}, # Lower threshold
]

print("Confidence-based Track Loss Analysis")
print("====================================")
print(f"{'Survival Factor':<15} {'Threshold':<10} {'Frames to Loss':<15} {'Time @ 30fps (s)':<15}")
print("-" * 65)

for config in configs:
    sf = config["survival_factor"]
    thresh = config["threshold"]
    frames = frames_to_loss(sf, thresh)
    time_30fps = frames / 30.0
    
    print(f"{sf:<15.2f} {thresh:<10.2f} {frames:<15.1f} {time_30fps:<15.2f}")

print("\nExample decay progression (survival_factor=0.97, threshold=0.1):")
print("Frame | Confidence")
print("------|----------")
confidence = 1.0
survival_factor = 0.97
for frame in range(0, 81, 10):
    print(f"{frame:5d} | {confidence:8.3f}")
    if confidence < 0.1:
        print(f"      | TRACK LOST (below {0.1} threshold)")
        break
    # Apply 10 frames of decay
    for _ in range(10):
        confidence *= survival_factor
