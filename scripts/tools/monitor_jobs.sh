#!/bin/bash
# Monitor MIB jobs: count done, report failures, resubmit to sphinx
WORK=/nlp/scr/aryaman/learning-to-attribute

# Count completed pkls
done=$(find $WORK/results -name '*.pkl' -newer $WORK/logs/fn-to-ioi-gpt2.out 2>/dev/null | wc -l)
queue=$(squeue -u aryaman -h 2>/dev/null | wc -l)
failed=$(grep -rl 'Error\|oom_kill' $WORK/logs/fn-*.out $WORK/logs/fe-*.out $WORK/logs/napig-*.out 2>/dev/null | wc -l)

echo "$(date '+%H:%M:%S') — queue=$queue done=$done failed=$failed"

# List new failures (not yet resubmitted)
for f in $WORK/logs/fn-*.out $WORK/logs/fe-*.out $WORK/logs/napig-*.out; do
    [ ! -f "$f" ] && continue
    name=$(basename "$f" .out)
    if strings "$f" 2>/dev/null | grep -q 'Error\|oom_kill'; then
        # Check if still in queue (already resubmitted)
        if ! squeue -u aryaman -h --name="$name" 2>/dev/null | grep -q .; then
            echo "  FAIL: $name"
        fi
    fi
done
