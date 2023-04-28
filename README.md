# Installation

It is recommended to use WSL on windows in order to program locally,
as MPI on windows is a pain, and the cluster you will need for experiments
uses Ubuntu. Install either mpich or openmpi dev packages for your distribution.
E.g.

```
sudo apt install libopenmpi-dev
```

in Ubuntu. On the cluster everything is already installed.

# Compilation

Put your sequential programs in src/seq, your OpenMP programs in src/omp,
your MPI programs in src/mpi.
Compile with make and find your programs in bin/ with suffixes \_seq, \_omp, \_mpi

# Experiments on the cluster

To run experiments on the cluster, you will need to login to slurm22, e.g.

```
ssh tkoopman@lilo7.science.ru.nl
ssh slurm22
```

and request resources from there. This is done through SLURM, look up the
documentation. omp\_hello.sh is a simple example script. Submit these
using sbatch, so

```
sbatch omp\_hello.sh
```

With MPI, you have to be careful how you allocate resources, to make sure
you do not schedule multiple ranks on one physical core (CPUs i and i + 8 are
the same physical core). The script mpi\_hello.sh generates SLURM scripts from
mpi\_hello.sh.template and submits them.

scp is a useful command to get files to and from the cluster.
