# Network Saturation Validation

## How to Run
The usage is similar to that in arch_sw/tiering/README.md, so refer to that document for guidance.

## Experiment Method
1. Select a target tiering configuration.
2. Experiment multiple times by changing the number of remote nodes in the configuration.
3. Multiple sets of logs are created if there are multiple checkpoints in one experiment.
4. In one checkpoint, several [serialization] and [sending] logs occur. Measure the transmission time of a checkpoint by averaging the [serialization]-[sending] pairs with the same number.
5. Record the time in `results/real_world/fig6_data.xlsx`.

## Log Example
```
[0] [2] [serialization] [3.1]
[1] [2] [serialization] [3.2]
[1] [2] [sending] [3.3]
[2] [2] [serialization] [3.4]
[0] [2] [sending] [3.6]
[2] [2] [sending] [3.7]
```
The first two numbers represent the [shard number(rank)] and [checkpoint number], respectively. Therefore, the transmission time for the 2nd checkpoint is calculated as follows:
$\frac{(3.6-3.1)+(3.3-3.2)+(3.7-3.4)}{3}$
