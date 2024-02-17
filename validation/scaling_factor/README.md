# Scaling Factor Validation

## How to Use
The preset settings and usage are similar to those in `arch_sw/single_anchor`.
However, when executing `run_experiments.sh`, it conducts multiple experiments.

## Experimental Method
1. Run the desired instances up to the maximum number (or as many as desired).
2. Modify `NODES` in `run_experiments.sh` appropriately. (Enter the number of nodes you want to experiment with in order. In the paper, the experiments were conducted in reverse order.)
3. Complete the preset settings and conduct the experiment. If necessary, the shell file can be modified as needed.
4. Parse the log files created in the home directory of the master node and record them in `results/real_world/fig5_data.xlsx`.

## Log Example
```
[2024-01-01-08-49-11] training_time_resnet18_32_nodes : 11.64543009
[2024-01-01-08-50-47] training_time_resnet18_16_nodes : 16.33841324
[2024-01-01-08-56-06] training_time_resnet18_1_nodes : 61.15543938
```
The log records the time, the name of the log and the number of nodes, and the total time spent on training, in that order.