# DeepVM (Solution)

## Dataset

`data.json`: Real instance data for `run.py`
`simul_data.json`: Virtual instance data for `simulation.py`

## Usage of run.py

Execute with the following command:

```
python ./run.py --pw [pricing willingness] --buffer_size [buffer size] --ckp_file_size [checkpoint file size]
```

The meaning of each option is as follows:

* `--pw`: Indicates pricing willingness, with a default value of 3. This value represents the maximum amount per hour that the user is willing to pay. (Unit: USD)
* `--buffer_size`: Sets the buffer size of the remote node. The default value is 0.5. (Unit: GB)
* `--ckp_file_size`: Enter the checkpoint file size of the target deep learning model. The default value is 2. (Unit: 100MB)

For example, if you want to set the pw to 4, the buffer size to 0.7GB, and the checkpoint file size to 300MB, you would enter:

```
python run.py --pw 4 --buffer_size 0.7 --ckp_file_size 3
```

## Usage of simulation.py

Execute with the following command:

```
python ./simulation.py --pw_start [start] --pw_stop [stop] --pw_step [step] --buffer_size [buffer size] --ckp_file_size [checkpoint file size]
```

The meaning of each option is as follows:

* `--pw_start`: Sets the start value of the pricing willingness (pw) range. The default value is 0. (Unit: USD)
* `--pw_stop`: Sets the end value of the pricing willingness (pw) range. The default value is 10.1, and this value is not included. (Unit: USD)
* `--pw_step`: Sets the interval between pricing willingness (pw) values. The default value is 0.1. (Unit: USD)
* `--buffer_size`: Sets the buffer size of the remote node. The default value is 0.5. (Unit: GB)
* `--ckp_file_size`: Enter the checkpoint file size of the target deep learning model. The default value is 2. (Unit: 100MB)

For example, if you want to experiment with increasing the pw value from 0.1 by 0.3 up to 5, while setting the buffer size to 0.7GB and the checkpoint file size to 300MB, you would enter:

```
python ./simulation.py --pw_start 0.1 --pw_stop 5.3 --pw_step 0.3 --buffer_size 0.7 --ckp_file_size 3
```
