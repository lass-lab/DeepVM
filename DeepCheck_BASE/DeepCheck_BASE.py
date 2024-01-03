"""
DeepCheck Package

This package provides the DeepCheck framework for distributed deep learning. It includes classes and functions necessary for utilizing DeepCheck in your learning script.

To use DeepCheck, import the package into your learning script and utilize the package through the DeepCheck_init() function.

For more detailed information and example code, please refer to the README and example code available on the DeepCheck GitHub repository:
GitHub Repository: https://github.com/DeepteamUser1/DeepVM.git

Last Modified: 2024-01-03
Author: TEAM_DeepVM
"""

from ctypes import *
from typing import Union, Any, Callable
import pickle
import os
import threading
import queue
import time
from datetime import datetime
import io
import sys
import torch
from torch.distributed import init_process_group, destroy_process_group

__version__ = '0.1.5'

class MPI:
    """
    MPI class provides initialization and functions for MPI communication.

    Purpose:
    The purpose of this class is to provide the necessary setup for MPI communication and
    functions for sending and receiving data via MPI.

    Usage:
    An instance of this class facilitates communication utilizing the provided MPI functionalities.

    Functions:
    - send: Sends data via MPI communication.
    - recv: Receives data via MPI communication.
    - make_sharding_rank: Collects and organizes rank numbers for sharding.
    - set_environ: Sets environment variables for distributed training and multilevel checkpointing.

    Variables:
    - rank: The rank number of the processor.
    - size: The total number of processors.
    - Get_processor_name: The name of the processor.
    """

    # Importing and Initializing the MPI module
    __mpi_module = CDLL('./mpi_module.so')
    __create_mpi_communication = __mpi_module.create_mpi_communication
    __create_mpi_communication.restype = c_void_p
    __delete_mpi_communication = __mpi_module.delete_mpi_communication
    __delete_mpi_communication.argtypes = [c_void_p]

    def __init__(self):
        """
        Initializes the MPI class.

        Purpose:
        - Loads the shared object (.so) file containing the MPI module.
        - Defines the necessary functions and variables for MPI communication.
        """

        # Creating MPI communication
        self.__mpi_communication = self.__create_mpi_communication()

        # Retrieving rank and size
        self.__get_rank = self.__mpi_module.getRank
        self.__get_rank.argtypes = [c_void_p]
        self.__get_rank.restype = c_int
        self.rank = self.__get_rank (self.__mpi_communication)

        self.__get_size = self.__mpi_module.getSize
        self.__get_size.argtypes = [c_void_p]
        self.__get_size.restype = c_int
        self.size = self.__get_size (self.__mpi_communication)

        # Retrieving processor name
        self.__get_processor_name = self.__mpi_module.getProcessorName
        self.__get_processor_name.argtypes = [c_void_p, POINTER(c_char_p), POINTER(c_int)]
        self.__get_processor_name.restype = None
        self.__processor_name = None

        # Initializing send and receive functions
        self.__send = self.__mpi_module.send
        self.__send.argtypes = [c_void_p, c_int, c_int, c_void_p, c_int]
        self.__send.restype = None

        self.__recv = self.__mpi_module.recv
        self.__recv.argtypes = [c_void_p, c_int, c_int, POINTER(c_void_p), POINTER(c_int), c_int, c_int]
        self.__recv.restype = None

        # Initializing other necessary functions and variables
        self.__all_gather_int = self.__mpi_module.allGatherInt
        self.__all_gather_int.argtypes = [c_void_p, c_void_p, c_int, c_void_p, c_int]
        self.__all_gather_int.restype = None

        self.__free_buffer = self.__mpi_module.free_buffer
        self.__free_buffer.argtypes = [c_void_p]
        self.__free_buffer.restype = None

        self.sharding_rank_list = None

    def __del__(self):
        self.__delete_mpi_communication(self.__mpi_communication)

    def Get_processor_name(self):
        """
        Returns the name of the processor.

        Returns:
            The name of the processor.

        """
        if self.__processor_name is None:
            data_buffer = c_char_p()
            data_buffer_size = c_int()
            self.__get_processor_name(self.__mpi_communication, byref(data_buffer), byref(data_buffer_size))
            self.__processor_name = data_buffer.value.decode() 
        return self.__processor_name

    def _send(self, data:Any, dest:int, tag:int) -> None:
        serialized_data = pickle.dumps(data)
        data_length = len(serialized_data)
        self.__send(self.__mpi_communication, c_int(int(dest)), c_int(int(tag)), serialized_data, data_length)

    def _byte_send(self, data:bytes, dest:int, tag:int) -> None:
        data_length = len(data)
        self.__send(self.__mpi_communication, c_int(int(dest)), c_int(int(tag)), data, data_length)

    def send(self, data:Any, dest:int, tag:int) -> None:
        """
        Sends data via MPI communication.

        Args:
            data: The data to be sent.
            dest: The destination rank number.
            tag: The tag for message identification.

        Returns:
            None
        """
        if isinstance(data, bytes):
            self._byte_send(data, dest, tag)
        else:
            self._send(data, dest, tag)

    def _recv(self, source:int, tag:int, shard_rank: int, save_count: int) -> bytes:
        data_buffer = c_void_p()
        data_buffer_size = c_int()
        self.__recv(self.__mpi_communication, c_int(int(source)), c_int(int(tag)), byref(data_buffer), byref(data_buffer_size), c_int(int(shard_rank)), c_int(int(save_count)))
        recvdata = string_at(data_buffer, data_buffer_size.value)
        self.__free_buffer(data_buffer)
        return recvdata
    
    def recv(self, source:int, tag:int, shard_rank: int, save_count: int, deserialize=True) -> Union[bytes, Any]:
        """
        Receives data via MPI communication.

        Args:
            source: The source rank number.
            tag: The tag for message identification.
            deserialize: Whether to deserialize the received data. Default is True.

        Returns:
            The received data as bytes or deserialized object.
        """
        if deserialize:
            return pickle.loads(self._recv(source, tag, shard_rank, save_count))
        return self._recv(source, tag, shard_rank, save_count)
    
    def make_sharding_rank(self) -> int:
        """
        Collects and organizes rank numbers for sharding.

        Returns:
            The number of ranks involved in sharding.
        """
        array_type = c_int * self.size
        localRank = array_type ()
        my_local_rank = c_int(int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]))
        self.__all_gather_int(self.__mpi_communication, byref(my_local_rank), 1, localRank, 1)
        localRankList = [i for i in localRank]
        sharding_rank_list = []
        for i in range(self.size-1):
            if localRankList[i] == 0:
                sharding_rank_list.append(i)

        self.sharding_rank_list = sharding_rank_list
        return len(sharding_rank_list)

    def set_environ(self, training_master_address:str, training_master_port:str, shard_size: int, train_count: int) -> None:
        """
        Sets environment variables for distributed training and multilevel checkpointing.

        Args:
            training_master_address: The address of the master.
            training_master_port: The port number of the master.

        Returns:
            None
        """
        os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
        os.environ["RANK"] = f"{self.rank}"
        os.environ["WORLD_SIZE"] = f"{train_count}"
        os.environ["MASTER_ADDR"] = training_master_address
        os.environ["MASTER_PORT"] = training_master_port
        
        if(int(os.environ["LOCAL_RANK"]) == 0):
            shard_index = f"{self.sharding_rank_list.index(self.rank)}"
            if int(shard_index) < shard_size:
                os.environ["SHARD_RANK"] = shard_index
            else:
                os.environ["SHARD_RANK"] = "-1"
        else:
            os.environ["SHARD_RANK"] = "-1"

    def print(self, msg):
        """
        Prints a message with the rank number.

        Args:
            msg: The message to be printed.

        Returns:
            None
        """
        print(f"rank {self.rank}: {msg}")

class TRAIN:
    def __init__(
        self,
        MPI: MPI,
        save_count: int,
        shard_size: int,
        shard_rank: int,
        destination_list: list,
    ):
        self.__MPI = MPI
        self.__save_count = save_count
        self.__shard_size = shard_size
        self.__shard_rank = shard_rank
        self.__destinaion_list = destination_list
    
    def __method(self, obj):
        # Dump the model data to a byte buffer
        byteBuffer = io.BytesIO()
        torch.save(obj, byteBuffer)

        # Get the dumped data as bytes
        dump = byteBuffer.getvalue()
        del byteBuffer

        def compute_shard(dump):
            """
            Compute the range of data to be used by the current shard.

            Args:
                dump (bytes): The dumped model data.

            Returns:
                left (int): The start index of the shard's data.
                right (int): The end index of the shard's data.
            """
            key_len = len(dump)
            compute_node_len = self.__shard_size
            q = int(key_len / compute_node_len)
            r = int(key_len % compute_node_len)
            rank = self.__shard_rank

            if rank+1 <= r:
                left = (rank)*(q+1)
                right = (rank+1)*(q+1)-1
            else:
                left = (rank)*q+r
                right = (rank+1)*q+r-1
            return left, right
            
        # Get the data range for the current shard
        left, right = compute_shard(dump)
        if left<=right:
            dump = dump[left:right+1]
        else:
            dump = bytes()

        # print(f"from {self.__MPI.rank} to {self.__destinaion_list[self.__MPI.rank]}")
        self.__MPI.send(data=dump, dest=self.__destinaion_list[self.__MPI.rank], tag=0)
        del dump

    def save(self, obj):
        """
        Request to save the model data.

        Args:
            obj (Any): The model data to be saved.
        """
        if self.__shard_rank >= 0:
            self.__method(obj)

class REMOTE:
    """
    REMOTE class represents a remote (peer) node used for multilevel checkpointing.

    The remote node receives data from training nodes and stores it in its own memory before flushing to files.
    This abstraction is designed to prevent data loss when storing data on training nodes, especially with Amazon Spot VMs that may terminate abruptly.

    The remote node consists of four components:
    a. Buffer: Stores multiple data samples using a circular queue-like structure to prevent bottlenecks during flushing when the write speed is slower than the data receiving speed.
    b. Receivers: Multiple receivers simultaneously receive sharded data from training nodes based on shard ranks and store it in the buffer.
    c. Flusher: Joins the sharded data in the buffer and writes it to files. It works in parallel with the receivers.
    d. Master: Manages the buffer, dirty bits, and coordination between the receivers and flusher.

    The REMOTE class instantiates the Receiver, Flusher, and Master as sub-classes and provides simple functions for users without exposing the internal structure.
    """
    class __ReceiverClass:
        """
        ReceiverClass represents the Receiver module that receives data from training nodes and stores it in the buffer.

        Args:
            MPI (MPI): The MPI object for communication.
            rank (int): The rank of the receiver.
            save_count (int): The number of times to perform the receive operation.
            remote_buffer (list): The remote buffer to store the received data.
            get_remote_buffer_current_index_func (Callable[[], int]): The function to get the current index of the remote buffer.
        """

        def __init__(
            self,
            MPI: MPI,
            rank: int,
            first_rank: int,
            save_count: int,
            remote_buffer: list,
            get_remote_buffer_current_index_func: Callable[[], int]
        ):
            self.__MPI = MPI
            self.__rank = rank
            self.__first_rank = first_rank
            self.__save_count = save_count
            self.__remote_buffer = remote_buffer
            self.__get_remote_buffer_current_index = get_remote_buffer_current_index_func
            self.__start_event = threading.Event()
            self.__end_event = threading.Event()
            self.__thread = threading.Thread(target=self.__method, args=())

        def __method(self):
            for i in range(self.__save_count):
                self.__start_event.wait()
                self.__start_event.clear()
                remote_buffer_index = self.__get_remote_buffer_current_index()
                # print(f"receive: from {self.__MPI.rank}, listening {self.__MPI.sharding_rank_list[self.__rank]}")
                received_data = self.__MPI.recv(source=self.__MPI.sharding_rank_list[self.__rank], tag=0, shard_rank=self.__rank, save_count=i+1, deserialize=False)
                self.__remote_buffer[remote_buffer_index][self.__rank-self.__first_rank] = received_data
                self.__end_event.set()

        def start(self):
            """
            Start the Receiver thread.
            """
            self.__thread.start()

        def request(self):
            """
            Send a request to the Receiver to start receiving data.
            """
            self.__start_event.set()
        
        def wait(self):
            """
            Wait until the Receiver has finished receiving data.
            """
            self.__end_event.wait()
            self.__end_event.clear()
        
    class __FlusherClass:
        """
        FlusherClass represents the Flusher module that joins the data in the buffer and writes it to files.

        Args:
            save_count (int): The number of times to perform the flush operation.
            remote_buffer (list): The remote buffer that contains the data to be flushed.
            clear_dirty_bit_func (Callable[[int], None]): The function to clear the dirty bit for a specific buffer index.
            get_file_name_func (Callable[[], str]): The function to get the file name for writing the data.
        """

        def __init__(
            self,
            save_count: int,
            remote_buffer: list,
            clear_dirty_bit_func: Callable[[int],None],
            get_file_name_func: Callable[[],str]
        ):
            self.__save_count = save_count
            self.__remote_buffer = remote_buffer
            self.__queue = queue.Queue()
            self.__thread = threading.Thread(target=self.__method, args=())
            self.__clear_dirty_bit = clear_dirty_bit_func
            self.__get_file_name = get_file_name_func
        
        def __method(self):
            for _ in range(self.__save_count):
                remote_buffer_index = self.__queue.get()
                entire_recv_data = b''.join(self.__remote_buffer[remote_buffer_index])
                # print("join complete")
                self.__clear_dirty_bit(remote_buffer_index)
                
                file_path_name = self.__get_file_name()
                # print("write start")
                with open(file_path_name, 'wb') as f:
                    f.write(entire_recv_data)
                    f.flush()
                    os.fsync(f.fileno())
                    f.close()
                # print("write complete")
                del entire_recv_data

        def start(self):
            """
            Start the Flusher thread.
            """
            self.__thread.start()                

        def enqueue(self, index: int):
            """
            Enqueue a buffer index to be processed by the Flusher.

            Args:
                index (int): The buffer index to be processed.
            """
            self.__queue.put(index)

    def __init__(
        self,
        MPI:MPI,
        save_count: int,
        remote_buffer_size: int,
        source_list: list,
        shard_size: int,
        model_name: str,
        file_name_include_datetime: bool,
        file_save_in_dictionary: bool,
    ):
        """
        Initialize the REMOTE node.

        Args:
            MPI (MPI): The MPI object for communication.
            save_count (int): The number of times to perform the save operation.
            remote_buffer_size (int): The size of the remote buffer.
            shard_size (int): The number of shards.
            model_name (str): The name of the model.
            file_name_include_datetime (bool): Whether to include the datetime in the file name.
            file_save_in_dictionary (bool): Whether to save the files in a separate directory.
        """
        self.__MPI = MPI
        self.__save_count = save_count
        self.__remote_buffer_size = remote_buffer_size
        self.__source_list = source_list
        self.__SHARD_SIZE = shard_size
        self.__model_name = model_name
        self.__file_name_include_datetime = file_name_include_datetime
        # self.__file_name_include_version = file_name_include_version
        self.__file_save_in_dictionary = file_save_in_dictionary

        self.__is_running = False
        self.__dirty_bits = [0]*self.__remote_buffer_size
        self.__dirty_bits_lock = threading.Lock()
        self.__remote_buffer = [[0 for _ in range(self.__SHARD_SIZE)] for _ in range(self.__remote_buffer_size)]
        self.__remote_buffer_index = 0

        self.__Receivers = [self.__ReceiverClass(self.__MPI, rank, self.__source_list[0], self.__save_count, self.__remote_buffer, self.__get_remote_buffer_current_index) for rank in self.__source_list]
        self.__Flusher = self.__FlusherClass(self.__save_count, self.__remote_buffer, self.__clear_dirty_bit, self.__get_file_name_and_path)

    def __get_remote_buffer_current_index(self):
        """
        Get the current index of the remote buffer.

        Returns:
            int: The current index of the remote buffer.
        """
        return self.__remote_buffer_index
    
    def __inc_remote_buffer_current_index(self):
        """
        Increment the current index of the remote buffer.
        """
        self.__remote_buffer_index = (self.__remote_buffer_index + 1) % self.__remote_buffer_size
    
    def __wait_dirty_bit(self, index: int):
        """
        Wait until the dirty bit for a specific buffer index is cleared.

        Args:
            index (int): The buffer index to wait for.
        """
        while True:
            if self.__dirty_bits[index] == 0:
                self.__set_dirty_bit(index)
                break

    def __set_dirty_bit(self, index: int):
        """
        Set the dirty bit for a specific buffer index.

        Args:
            index (int): The buffer index to set the dirty bit for.
        """
        with self.__dirty_bits_lock:
            self.__dirty_bits[index] = 1

    def __clear_dirty_bit(self, index: int):
        """
        Clear the dirty bit for a specific buffer index.

        Args:
            index (int): The buffer index to clear the dirty bit for.
        """
        with self.__dirty_bits_lock:
            self.__dirty_bits[index] = 0

    def __get_file_name_and_path(self) -> str:
        """
        Get the file name and path for writing the data.

        Returns:
            str: The file name and path.
        """
        fileName = "./" + str(self.__model_name)
        # if self.__file_save_in_dictionary:
        #     fileName = "./" + self.__model_name+"/" + fileName
        # if self.__file_name_include_datetime:
        #     fileName = fileName + "_" + datetime.now().strftime("%Y-%m-%d-%H%M%S") 

        fileName = fileName + ".pt.tar"
        return fileName

    def start(self):
        """
        Start the REMOTE node.
        """
        assert not self.__is_running, "This remote node already started."
        self.__is_running = True

        if self.__file_save_in_dictionary and not os.path.exists("./"+self.__model_name):
            os.makedirs("./"+self.__model_name)

        self.__Flusher.start()
        for Receiver in self.__Receivers:
            Receiver.start()

        for _ in range(self.__save_count):
            target_buffer_index = self.__get_remote_buffer_current_index()
            self.__wait_dirty_bit(target_buffer_index)

            for Receiver in self.__Receivers:
                Receiver.request()
            
            for Receiver in self.__Receivers:
                Receiver.wait()
                
            self.__Flusher.enqueue(target_buffer_index)

            self.__inc_remote_buffer_current_index()

def __init_remote_node(
    communicator: MPI,
    save_count: int,
    remote_buffer_size: int,
    source_list: list,
    model_name: str,
    file_name_include_datatime: bool,
    file_save_in_dictionary: bool,
):
    """
    Initialize the REMOTE node.

    Args:
        communicator (MPI): The MPI object for communication.
        save_count (int): The number of times to perform the save operation.
        remote_buffer_size (int): The size of the remote buffer.
        shard_size (int): The number of shards.
        model_name (str): The name of the model.
        file_name_include_datetime (bool): Whether to include the datetime in the file name.
        file_save_in_dictionary (bool): Whether to save the files in a separate directory.

    Returns:
        REMOTE: The initialized REMOTE node.
    """
    shard_size = len(source_list)
    remote_node = REMOTE(communicator, save_count, remote_buffer_size, source_list, shard_size, model_name, file_name_include_datatime, file_save_in_dictionary)
    print("remote node ready")
    return remote_node

def __init_train_node(
    communicator: MPI,
    train_count: int,
    destination_list: list,
    training_master_address: str,
    training_master_port: str,
    save_count: int,
    shard_size: int,
):
    """
    Initialize the TRAIN node.

    Args:
        communicator (MPI): The MPI object for communication.
        training_master_address (str): The address of the master node.
        training_master_port (str): The port of the master node.
        save_count (int): The number of times to perform the save operation.
        shard_size (int): The number of shards.

    Returns:
        TRAIN: The initialized TRAIN node.
    """

    communicator.set_environ(training_master_address, training_master_port, shard_size, train_count)
    print("set environ complete")
    init_process_group (backend="nccl", rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
    print("init process group complete")
    torch.cuda.set_device (int(os.environ["LOCAL_RANK"]))
    train_node = TRAIN(communicator, save_count, shard_size, int(os.environ["SHARD_RANK"]), destination_list)
    print("train node ready.")
    return train_node

def calculate_save_count(
    start_epoch: int,
    total_epochs: int,
    save_period: int
) -> int:
    """
    Calculate the number of times to perform the save operation.

    Args:
        start_epoch (int): The starting epoch.
        total_epochs (int): The total number of epochs.
        save_period (int): The period of save operations.

    Returns:
        int: The number of times to perform the save operation.
    """

    save_points = list(range(1, total_epochs+1, save_period))
    save_counts = len([point for point in save_points if point >= start_epoch])
    return save_counts

def make_list(
    train_count: int,
    remote_count: int,
    shard_size: int,
    sharding_rank_list: list
) -> tuple[list, list]:
    
    q = shard_size // remote_count
    r = shard_size % remote_count
    
    even_divided = [q + 1] * r + [q] * (remote_count - r)

    destination_list = [-1] * train_count
    
    start_idx = 0
    for count_idx, count in enumerate(even_divided):
        for _ in range(count):
            destination_list[sharding_rank_list[start_idx]] = count_idx+train_count
            start_idx += 1
    
    start_idx = 0
    source_list = []
    for count in even_divided:
        end_idx = start_idx + count
        source_list.append(sharding_rank_list[start_idx:end_idx])
        start_idx = end_idx
    
    return destination_list, source_list

def init_DeepCheck(args, **overrides) -> MPI:
    """
    Initialize the DeepCheck environment.

    Args:
        args (Namespace): The arguments parsed from the command line.
        train_node_auto_start (bool): Whether to automatically start the train node.
        **overrides: Additional keyword arguments to override the arguments from `args`.

    Returns:
        Tuple[MPI, Optional[TRAIN], Optional[REMOTE]]: The MPI communicator, TRAIN node (if created), and REMOTE node (if created).
    """
    print("*** This is DeepCheck_BASE. ***")

    args_dict = vars(args)
    valid_keys = set(args_dict.keys())
    for key in overrides:
        if key not in valid_keys:
            raise ValueError(f"Unexpected key: {key}")
    
    args_dict.update(overrides)
    train_count = args_dict['train_count']
    remote_count = args_dict['remote_count']
    training_master_address = args_dict['training_master_addr']
    training_master_port = args_dict['training_master_port']
    total_epochs = args_dict['total_epochs']
    save_period = args_dict['save_period']
    starting_epoch = args_dict['starting_epoch']
    remote_buffer_size = args_dict['remote_buffer_size']
    shard_size = args_dict['shard_size']
    file_name_include_datetime = args_dict['file_name_include_datetime']
    file_save_in_dictionary = args_dict['file_save_in_dictionary']
    model_name = args_dict['model_name']
    snapshot_path = args_dict['snapshot_path']

    if snapshot_path:
        assert os.path.exists (snapshot_path), f"can't open file \'{snapshot_path}\': No such file or directory"
        snapshot = torch.load(snapshot_path, map_location='cpu')
        starting_epoch = snapshot['epoch']+1
        del snapshot

    communicator = MPI()
    
    assert train_count+remote_count == communicator.size, f"count error."
    
    assert train_count >= remote_count and shard_size >= remote_count, f"too many remote nodes."
    
    max_shard_size = communicator.make_sharding_rank()
    
    destination_list, source_list = make_list(train_count, remote_count, shard_size, communicator.sharding_rank_list)

    assert shard_size <= max_shard_size, f"The shard size is too large. (input: {shard_size}, max: {max_shard_size})"

    save_count = calculate_save_count(starting_epoch, total_epochs, save_period)

    train_node = None
    remote_node = None

    if communicator.rank >= communicator.size-remote_count:
        remote_node = __init_remote_node(communicator, save_count, remote_buffer_size, source_list[communicator.rank-train_count], model_name, file_name_include_datetime, file_save_in_dictionary)
        remote_node.start()
        sys.exit()
    else:
        train_node = __init_train_node(communicator, train_count, destination_list, training_master_address, training_master_port, save_count, shard_size)
    
    return communicator, train_node, remote_node

def destroy_DeepCheck():
    """
    Destroy the DeepCheck environment.
    """
    destroy_process_group ()

if __name__ == "__main__":
    print("This Python code is a package file.")
    print("[Error] It cannot be executed directly; please import it into your learning script.")
