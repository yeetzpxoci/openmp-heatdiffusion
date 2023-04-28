#!/bin/sh

#SBATCH --account=csmpistud
#SBATCH --cpus-per-task=32
#SBATCH --partition=csmpi_fpga_short
#SBATCH --time=00:05:00
#SBATCH --output=stencil_omp.out

# Compile on the machine, not the head node
make bin/2dstencil_omp

for P in 1 2 3 4 5 6; do
    OMP_NUM_THREADS="$P" bin/2dstencil_omp 10000 100 > results/stencil_omp_"$P".txt
done
