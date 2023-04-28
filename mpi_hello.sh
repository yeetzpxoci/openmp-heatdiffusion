#!/bin/sh

NODES=1
while [ $NODES -le 8 ]
do
    NTASKS=$(( 8 * NODES ))
    
    sed "s/NODES/$NODES/g" mpi_hello.sh.template > mpi_hello_"$NTASKS".sh
    sed -i "s/NTASKS/$NTASKS/g" mpi_hello_"$NTASKS".sh
    sbatch mpi_hello_"$NTASKS".sh

    rm mpi_hello_"$NTASKS".sh

    NODES=$(( 2 * NODES ))
done
