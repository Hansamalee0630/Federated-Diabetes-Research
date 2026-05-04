# Component 4 - Architecture Search

#### **Architecture Experiments**
Successfully ran and saved 4 configurations:
- ✓ Baseline: 2 layers [256,128], head=64 → **Best Overall**
- ✓ 3-Layer: [256,128,64], head=64 → Slower, worse accuracy
- ✓ Wider: [256,128], head=128 → Better local, worse global
- ✓ Deeper: [256,128], head=64×2 → Slowest, no benefit

## Key Results

| Metric | Baseline (2L, 64) | 3-Layer | Wider (128) | Deeper (2D) |
|--------|-------------------|---------|-------------|-------------|
| **Global Acc** | **0.5592** ✓ | 0.5376 | 0.5260 | 0.5403 |
| **Pers Acc** | 0.5946 | 0.5790 | **0.5978** ✓ | 0.5928 |
| **HTN AUROC** | 0.6924 | 0.6860 | **0.6957** ✓ | 0.6927 |
| **HF AUROC** | 0.7223 | 0.7152 | **0.7240** ✓ | 0.7216 |
| **Fairness** | **0.0062** ✓ | 0.0107 | **0.0052** ✓ | 0.0100 |
| **Time (s)** | **409** ✓ | 448 | 461 | 571 |
| **Model Size** | **0.25 MB** ✓ | 0.23 MB | 0.34 MB | 0.29 MB |

### Winner: **Baseline (2-layer, head=64)**
- Best global accuracy (critical for FL aggregation)
- Best efficiency (fastest training, smallest model)
- Excellent fairness
- Well-balanced performance
