#!/bin/bash

# clear

for i in {7..84..7}
do
if (("$i" <= "70"))
then
    l=$i
else
    l=$((70 + ($i - 70)*5))
fi
k=$(($l*$l))

echo "$k rays:"
for j in {1..16}
do
mpirun -n 4 ./discretize_geom ../tests/unitbox.h5m $k 0
# ./discretize_geom ../tests/unitbox.h5m $k 1
done
done
