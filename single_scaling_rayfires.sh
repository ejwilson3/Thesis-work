#!/bin/bash

clear

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
for j in {1..32}
do
./bld/discretize_geom ../tests/1LineVoxel.h5m $k 1

if (("$j" == 8 || "$j" == 16 || "$j" == 24 || "$j" == 32))
then
echo ""
fi

done
done
