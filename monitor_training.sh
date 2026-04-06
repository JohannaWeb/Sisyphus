#!/bin/bash
# Monitor 20M model training
echo "=== Sisyphus 20M Training Monitor ==="
echo "Config: config.20m.yaml"
echo "Steps: 15,000 (~1.6 days on 4060 Ti)"
echo ""

while true; do
    if [ -f training.20m.log ]; then
        echo "Last 5 updates:"
        tail -5 training.20m.log | grep "step" | tail -5
        echo ""
        
        # Count steps completed
        steps=$(tail -1 training.20m.log | grep -o "step [0-9]*" | grep -o "[0-9]*" || echo "?")
        if [ "$steps" != "?" ]; then
            echo "Progress: Step $steps / 15,000"
            pct=$((steps * 100 / 15000))
            echo "Complete: $pct%"
            
            # Estimate time remaining
            if [ $steps -gt 300 ]; then
                time_per_step=$(tail -300 training.20m.log | grep "step" | tail -2 | awk '{print $1}' | tail -1)
                echo ""
                echo "Check again in 1 hour for next update"
            fi
        fi
    else
        echo "Log file not found yet..."
    fi
    echo ""
    echo "To stop training: kill $(pgrep -f 'train.py --config config.20m')"
    echo "To resume later: python3 src/train.py --config config.20m.yaml --resume checkpoints/sisyphus.20m.last.pt"
    echo ""
    sleep 3600
done
