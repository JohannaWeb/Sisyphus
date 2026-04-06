# Training Monitoring Guide

## Quick Check (takes 5 seconds)

```bash
# See last 5 training steps
tail -5 training.20m.log | grep "step"
```

Output shows: `step NUMBER | train LOSS | val LOSS | lr RATE`

## Live View (updates in real-time)

```bash
# Watch training as it runs
tail -f training.20m.log

# Exit with Ctrl+C
```

## Parse Current Status

```bash
# Get just the progress
tail -1 training.20m.log | grep "step"

# Example: "step   600 | train 2.4200 | val 2.2934 | lr 0.000200"
```

## Check If Training Is Running

```bash
# See GPU usage (if NVIDIA)
nvidia-smi -l 1

# Or check process
ps aux | grep train.py | grep config.20m

# Or check memory
watch -n 1 'ps aux | grep train.py'
```

## Track Loss Over Time

```bash
# Show all steps with loss
grep "step" training.20m.log | awk '{print $2, $6, $8}'

# Example output:
# step 0 train 5.4766
# step 300 train 2.6154
# step 600 train 2.4200
```

## Check Checkpoint Files

```bash
# See what's been saved
ls -lh checkpoints/sisyphus.20m*

# sisyphus.20m.last.pt = most recent (updates every 1000 steps)
# sisyphus.20m.pt = best validation loss (updated when val improves)
```

## Create a Dashboard

```bash
# One-liner to show status every 30 seconds
watch -n 30 'echo "=== Training Status ===" && tail -1 training.20m.log && echo "" && ls -lh checkpoints/sisyphus.20m* 2>/dev/null | tail -2'
```

## Full Monitoring Command

```bash
# Comprehensive status check
cat << 'STATS'
=== SISYPHUS 20M TRAINING STATUS ===
STATS

echo "Latest steps:"
tail -3 training.20m.log | grep "step"

echo ""
echo "Checkpoint files:"
ls -lh checkpoints/sisyphus.20m*.pt 2>/dev/null || echo "No checkpoints yet"

echo ""
echo "Is training running?"
ps aux | grep "train.py --config config.20m" | grep -v grep || echo "Not found (may have finished)"

echo ""
echo "Training logfile size:"
wc -l training.20m.log | awk '{print $1 " lines"}'
```

## Expected Progress

- **Step 0-300**: Loss drops from 5.5 → 2.5 (fast initial learning)
- **Step 300-5000**: Loss drops from 2.5 → 1.5 (steady improvement)
- **Step 5000-15000**: Loss drops from 1.5 → 1.2 (slow refinement)

If loss isn't decreasing, something is wrong.

## What to Look For

| Sign | Meaning |
|------|---------|
| Loss decreasing | ✓ Training working |
| Loss flat/increasing | ✗ Problem (stop and debug) |
| No new log lines | ✗ Process died (check errors) |
| val loss < train loss | ✓ Normal (overfitting slightly) |
| Checkpoints updating | ✓ Running and saving |

## Troubleshooting

```bash
# Check for errors in the log
tail -50 training.20m.log | grep -i error

# Check if out of memory
tail -50 training.20m.log | grep -i "out of memory"

# Full last 100 lines
tail -100 training.20m.log
```

## When Training Finishes

You'll see:
```
Best checkpoint written to checkpoints/sisyphus.20m.pt
Last checkpoint written to checkpoints/sisyphus.20m.last.pt
```

Then run:
```bash
python3 src/generate.py --checkpoint checkpoints/sisyphus.20m.pt \
  --prompt "fn main() {" --temperature 0.7 --max-new-tokens 100
```
