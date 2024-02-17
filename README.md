# DeepVM : Integrating Spot and On-Demand VMs for Cost-Efficient Deep Learning Clusters in the Cloud
## Introduction
This repository hosts the open-source implementation of DeepVM, a solution introduced in a paper submitted to CCGrid. DeepVM is designed for cost-efficient deep learning clusters in cloud environments, offering an innovative approach to integrating Spot and On-Demand VMs.

## What is DeepVM?
DeepVM is a novel solution for achieving efficient and cost-effective use of resources in distributed deep learning. It operates through a four-stage process:
1. **User Pricing Input**: Collecting the user's maximum price per hour willingness.
2. **Instance-level Analysis**: Analyzing all available instances using the 'FLOPP' (Floating-point Operations Per Price) metric to measure each instance’s performance relative to its cost.
3. **Architecture-level Analysis**: Exploring the most cost-effective instance combinations within predefined architectures using linear programming, considering each architecture’s constraints and potential overheads in parallel processing.
4. **Final Decision**: Identifying the best combinations for all architectures and proposing the most cost-effective cluster configuration under given conditions​​.

## Components
### DeepVM
DeepVM serves as the planner or solution provider in this setup. It includes both real and simulated instances to offer a comprehensive solution for various scenarios. Its capability to rapidly make decisions and handle computational tasks efficiently is a key highlight.
### DeepCheck (asynchronous multilevel checkpointing software)
For validating DeepVM's Tiering architecture, we developed DeepCheck, a software supporting asynchronous multilevel checkpointing. Built using PyTorch and the MPI library, DeepCheck showcases the capabilities of DeepVM under various conditions. It is essential for ensuring the reliability and effectiveness of the Tiering architecture in DeepVM​.
### Other Components
Comprised of example codes for experiment reproduction, tools for generating figures used in the paper, and instance datasets.

## Dataset
### Virtual Instance Data
The virtual instance data is in the following `json` format.
```json
{
    "instances": [
        {
            "name": "A",
            "ondemand_price": 0.75,
            "spot_price": 0.225,
            "network_bandwidth": 10.0,
            "flops": 100.0,
            "a": 0.14,
            "b": 14.5,
            "c": 13.5,
            "type": "G",
            "vCPU": 4
        },
    ],
    "available_vcpus": {
        "G": {
            "ondemand": 32,
            "spot": 128
        },
    }
}
```
`instances` include information about each instance, encompassing name, price, hardware specifications, and performance.
`available_vcpus` records the maximum number of each type of instance that can be used.

### Real Instance Data
The real instance data is divided into parts embedded in the code and those stored in `json`.
The part embedded in the code includes the parameters of the regression function and the $n_{sat}$ values based on network bandwidth.
```python3
a_values = {
    ...
}
b_values = {
    ...
}
c_values= {
    ...
}
table = {
    ...
}
```
The part stored in `json` format is similar to the virtual instance data.
```json
{
    "instances": [
        {
            "name": "g3s.xlarge",
            "type": "G",
            "ondemand_price": 0.75,
            "spot_price": 0.225,
            "vCPU": 4,
            "memory": 30.5,
            "network_bandwidth": 10,
            "flops": 100
        },
    ],
    "available_vcpus": {
        "G": {
            "ondemand": 32,
            "spot": 128
        },
    }
}
```

## Important Note
- The implementation respects the double-anonymity policy of the CCGrid submission process.
- The provided software and algorithms are for research and educational purposes. Users should exercise caution and understand the limitations and requirements of each component.

## Repository Structure
The artifact consists of four main directories:

- `solution/`: Includes LP modeling and simulation programs, as well as the instance datasets used and dataset generation programs.

- `arch_sw/`: Contains example source codes for performing distributed learning specific to each architecture, to replicate the effectiveness experiments of DeepVM in the paper.

- `validation/`: Includes the training source code and usage instructions to replicate the overhead modeling experiments presented in the paper.

- `results/`: Contains all the source code required to generate the graphs presented in the paper using the result files.