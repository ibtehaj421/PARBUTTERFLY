// your_program.c
#include <stdio.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    #pragma omp parallel
    {
        printf("Hello from thread %d in process %d\n", omp_get_thread_num(), world_rank);
    }

    MPI_Finalize();
    return 0;
}
