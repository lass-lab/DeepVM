# How to Use Single-Anchor

> You should have 5 files and a temp directory ready.
> download.py, initialization.sh, run_experiments.sh, template, single_anchor_example.py, temp/ (directory)

0. Add the following content to the hosts file via sudo vi /etc/hosts. (It's convenient to write it down in a notepad and paste it.)

```
172.31.0.1 worker1
172.31.0.2 worker2
172.31.0.3 worker3
172.31.0.4 worker4
```

Enter the private ip addresses of each node for the ip addresses.
In this case, worker1 must always be the master.

1. Add execution permission to all sh files. (chmod +x *.sh)
2. Run initialization.sh. Then,
   - On the master, download and unzip the dataset (python3 ./download.py)
   - Send the dataset and training code to the workers (scp)
   - Remotely unzip the dataset on the worker. (Run python3 ./download.py on each node)
3. Modify the parameters in run_experiments.sh. (NNODES, MASTER_PORT, EPOCH). Modify the number of epochs to 1.
4. Run run_experiments.sh to perform training for 1 epoch. (You need to run it once for subsequent experimental results to come out correctly.)
5. If the training goes well, modify the EPOCH value to the desired number of epochs and then perform the experiment.
6. In single_anchor_example.py, you can also modify the model (resnet50) to another model. If you modify it, you need to resend the modified single_anchor_example.py to the slave.
