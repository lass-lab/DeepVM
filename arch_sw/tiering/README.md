# How to Use Tiering

> 7 files must be prepared.
> initialization.sh, compile.sh, DeepCheck.py, run_experiments.sh, hosts, mpi_module.cpp, tiering_example.py

0. Use sudo vi /etc/hosts to add the following content to the hosts file. (It's convenient to write it down in a notepad beforehand and then paste it.)

```
172.31.0.1 worker1
172.31.0.2 worker2
172.31.0.3 worker3
172.31.0.4 worker4
172.31.0.5 remote1
```

Enter the private IP addresses of each node as the IP addresses.
In this case, worker1 must always be the master of training.

1. Add execution permissions to all sh files. (chmod +rx *.sh)
2. Run initialization.sh. This script will automatically:
   - Download and unzip the dataset on the master (python3 ./download.py)
   - Send the dataset and training code to workers and remote (scp)
   - Perform remote dataset unzipping on the workers (execute python3 ./download.py on each node)
   - Compile the checkpointing module on workers and remote (compile.sh)
3. Manually edit the hosts file as necessary.
4. Please modify the parameters of run_experiments.sh. At this point, set the number of epochs to 1.
5. Execute run_experiments.sh to train for 1 epoch. (You need to run it once for subsequent experiment results to be accurate.)
6. If training goes well, modify the EPOCH value to your desired number of epochs and then perform the experiment.
7. You can also modify the model (resnet50) to a different model in tiering_example.py. If you do modify it, you need to resend the modified tiering_example.py to the workers and remote.
