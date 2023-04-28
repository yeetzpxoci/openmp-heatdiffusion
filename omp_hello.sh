#!/bin/sh

#SBATCH --account=csmpistud
#SBATCH --cpus-per-task=32
#SBATCH --partition=csmpi_fpga_short
#SBATCH --time=00:05:00
#SBATCH --output=hello_omp.out

# Compile on the machine, not the head node
make bin/hello_world_omp

for P in 1 2 4 8 16 32; do
    OMP_NUM_THREADS="$P" bin/hello_world_omp > results/hello_omp_"$P".txt
done
