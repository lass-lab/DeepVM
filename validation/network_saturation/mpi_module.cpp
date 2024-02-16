#include <mpi.h>
#include <cstdlib>
#include <stdio.h>
#include <sys/time.h>

class MPICommunication {
public:
    MPICommunication() {
        MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
		if(provided<MPI_THREAD_MULTIPLE) {
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		}
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
	    MPI_Get_processor_name(processor_name, &processor_name_size);
    }

    ~MPICommunication() {
        MPI_Finalize();
    }

    void _send(int dest, int tag, void* data, int count, MPI_Datatype datatype) {
        MPI_Send(data, count, datatype, dest, tag, MPI_COMM_WORLD);
    }

    void _recv(int source, int tag, void* data, int count, MPI_Datatype datatype, MPI_Status* status) {
        MPI_Recv(data, count, datatype, source, tag, MPI_COMM_WORLD, status);
    }

    void _allGather(void* send_data, int send_count, MPI_Datatype send_datatype, void* recv_data, int recv_count, MPI_Datatype recv_datatype) {
        MPI_Allgather(send_data, send_count, send_datatype, recv_data, recv_count, recv_datatype, MPI_COMM_WORLD);
    }

    int _getRank() {
	    return rank;
    }
    int _getSize() {
	    return size;
    }
    char* _getProcessorName(int* size) {
        *size = processor_name_size;
        return processor_name;
    }

   

private:
    int rank;
    int size;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int processor_name_size;
	int provided;
};

extern "C" {
    struct timeval tv;
    MPICommunication* create_mpi_communication() {
        return new MPICommunication();
    }

    void delete_mpi_communication(MPICommunication* mpi_comm) {
        delete mpi_comm;
    }

    void send(MPICommunication* mpi_comm, int dest, int tag, void* data, int count) {
        mpi_comm->_send(dest, tag, data, count, MPI_BYTE);
    }

    void recv(MPICommunication* mpi_comm, int source, int tag, void** data, int* count, int shard_rank, int save_count) {
        MPI_Status status;
        MPI_Probe(source, tag, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_BYTE, count);
        *data = malloc(*count);
        mpi_comm->_recv(source, tag, *data, *count, MPI_BYTE, &status);
        gettimeofday(&tv, NULL);
        printf("[%d] [%d] [sending] [%lf]\n", shard_rank, save_count , (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0);
    }
    void allGatherInt(MPICommunication* mpi_comm, void* send_data, int send_count, void* recv_data, int recv_count) {
        mpi_comm->_allGather(send_data, send_count, MPI_INT, recv_data, recv_count, MPI_INT);
    }

    int getRank(MPICommunication* mpi_comm) {
	    return mpi_comm->_getRank();
    }
    int getSize(MPICommunication* mpi_comm) {
	    return mpi_comm->_getSize();
    }
    void getProcessorName(MPICommunication* mpi_comm, void** buf, int* size) {
	    *buf = mpi_comm->_getProcessorName(size);
    }
    void free_buffer(void* data) {
	    free(data);
    }
}

