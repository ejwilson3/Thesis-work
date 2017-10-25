#!/bin/bash

k=225
for m in {1..34}
do

echo ""
echo "$m voxels:"
for j in {1..32}
do
./discretize_geom ~/jous/"$m"LineVoxel.h5m $k 1 $m

if (("$j" == 8 || "$j" == 16 || "$j" == 24 || "$j" == 32))
then
echo ""
fi

done
done
