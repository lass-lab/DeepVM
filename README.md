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
### DeepCheck
For validating DeepVM's Tiering architecture, we developed DeepCheck, a software supporting asynchronous multilevel checkpointing. Built using PyTorch and the MPI library, DeepCheck showcases the capabilities of DeepVM under various conditions. It is essential for ensuring the reliability and effectiveness of the Tiering architecture in DeepVM​.
<!-- ### DeepCheck-BASE
DeepCheck-BASE offers similar functionality to DeepCheck, supporting asynchronous checkpointing. However, it differs in that it is not optimized for the specific needs of DeepVM, providing a baseline for comparison and further development. -->

## Important Note
- The implementation respects the double-anonymity policy of the CCGrid submission process.
- The provided software and algorithms are for research and educational purposes. Users should exercise caution and understand the limitations and requirements of each component.

## Repository Structure
`deepvm/`: Contains the core DeepVM solution and associated algorithms.

`deepcheck/`: Includes the DeepCheck software for optimized asynchronous checkpointing.

<!-- `deepcheck-base/`: Houses the baseline DeepCheck software. -->