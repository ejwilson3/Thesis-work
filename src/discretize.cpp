#include "discretize.h"
#include <stdexcept>
#include <stdlib.h>
#include <math.h>
#include <iostream>

namespace pyne{
//WATCH THIS
std::vector<disc_result> discretize_geom(std::vector<std::vector<double> > mesh,
    std::map<EntityHandle, int> vol_handles_ids,
    int num_rays,
    bool grid = false) {

  /*
  for (std::map<EntityHandle, int>::iterator it = vol_handles_ids.begin();
       it != vol_handles_ids.end(); ++it) {
    std::cout << "Handle address: " << &it->first << std::endl;
    std::cout << "Vol id: " << it->second << std::endl;
  }
  */
  // This will store the information of the individual row and how the rays
  // are to be fired.
  struct mesh_row row;
  row.num_rays = num_rays;
  row.grid = grid;

  //WATCH THIS
  std::vector<disc_result> result;

  // These for loops visit each individual row.
  for (int d1 = 0; d1 < 3; d1++) {

    // Set up the different direction indices.
    int d2 = (d1 + 1)%3;

    row.d3 = 3 - d1 - d2;
    row.d3divs = mesh.at(row.d3);
    for (int i = 0; i < mesh.at(d1).size() - 1; i++) {
      row.d1div1 = mesh.at(d1).at(i);
      row.d1div2 = mesh.at(d1).at(i+1);

      for (int j = 0; j < mesh.at(d2).size() - 1; j++) {
        row.d2div1 = mesh.at(d2).at(j);
        row.d2div2 = mesh.at(d2).at(j+1);
        fireRays(row, vol_handles_ids);
      }
    }
  }

  int total_rays = num_rays*3;
  struct disc_result ray_results;
  for (int i = 0; i < row.totals.size(); i++) {
    for (std::map<EntityHandle, double*>::iterator it = row.totals.at(i).begin();
         it != row.totals.at(i).end(); ++it) {
      ray_results.idx = i;
      ray_results.cell = it->first;
      ray_results.vol_frac = it->second[0]/total_rays;
      ray_results.rel_error = sqrt((it->second[1])/pow((it->second[0]),2.0)
                                   - 1.0/total_rays);
      result.push_back(ray_results);
    }
  }
  return result;
}

void fireRays(mesh_row &row, std::map<EntityHandle, int> vol_handles_ids) {

  std::vector<double> width;
  for (int i = 0; i < row.d3divs.size() - 1; i++) {
    width.push_back(row.d3divs.at(i+1) - row.d3divs.at(i));
  }

  startPoints(row, 0);
  vec3 pt;
  pt[row.d3] = row.d3divs.at(0);
  pt[(row.d3+1)%3] = row.start_point_d1;
  pt[(row.d3+2)%3] = row.start_point_d2;
  int result;
  vec3 dir = {0};
  dir[row.d3] = 1;
  // Find the volume that has the starting point for the first ray.
  EntityHandle eh = find_volume(vol_handles_ids, pt, dir);

  for (int i = 1; i < row.num_rays; i++) {
    startPoints(row, i);
    pt[(row.d3+1)%3] = row.start_point_d1;
    pt[(row.d3+2)%3] = row.start_point_d2;
    // If the next point starts in the same volume as the last one, calling this
    // here can save calling find_volume, which gets expensive.
    dag_pt_in_vol(eh, pt, &result, dir, NULL);
    if (!result)
      eh = find_volume(vol_handles_ids, pt, dir);

    double value;

    int num_intersections;
    EntityHandle *surfs, *volumes;
    double *distances;
    void* buf;
    dag_ray_follow(eh, pt, dir, 0.0, &num_intersections,
                   &surfs, &distances, &volumes, buf);

    int count = 0;
    row.totals.resize(width.size());
    int j = 0;
    bool complete = false;
    double curr_width = width.at(count);
    // Loops over the ray's intersections with the different volumes.
    while(!complete) {
      // while the distance to the next intersection is greater than the current
      // width, we calculate the value and we move to the next width, shortening
      // the distance.
      while (distances[j] >= curr_width) {
        value = curr_width/width.at(count);
        std::map<EntityHandle, double*>::iterator it =
            row.totals.at(count).find(volumes[j]);
        // If the current cell isn't in the totals yet, add it.
        if (it == row.totals.at(count).end()){
          double zeros[2] = {0.0,0.0};
          row.totals.at(count).insert(it, std::make_pair(volumes[j],zeros));
        }
        row.totals.at(count).at(volumes[j])[0] += value;
        row.totals.at(count).at(volumes[j])[1] += value*value;
        distances[j] -= curr_width;

        // If there are more volume elements, move to the next one.
        if (width.size() - 1 > count) {
          count++;
          curr_width = width.at(count);
        }
        // If you get here you've finished the mesh for this ray.
        else {
          complete = true;
          break;
        }
      }
      // If the distance to the next intersection is smaller than the distance
      // to the next element, move up to the intersection and move on to the
      // next 'distance'.
      if(distances[j] < curr_width) {
        value = distances[j]/curr_width;
        std::map<EntityHandle, double*>::iterator it =
            row.totals.at(count).find(volumes[j]);
        // If the current cell isn't in the totals yet, add it.
        if (it == row.totals.at(count).end()){
          double zeros[2] = {0.0,0.0};
          row.totals.at(count).insert(it, std::make_pair(volumes[j],zeros));
        }
        row.totals.at(count).at(volumes[j])[0] += value;
        row.totals.at(count).at(volumes[j])[1] += value*value;
        curr_width -= distances[j];
      }
      j++;
    }
    dag_dealloc_ray_buffer(buf);
  }
}

void startPoints(mesh_row &row, int iter) {
  double dist_d1 = row.d1div2 - row.d1div1;
  double dist_d2 = row.d2div2 - row.d2div1;
  if (row.grid) {
    int points = sqrt(row.num_rays);
    //Iterate from 1 to points, repeat points times.
    int iter_d1 = iter%points + 1;
    //Iterate slowly from 1 to points once.
    int iter_d2 = iter/points + 1;

    row.start_point_d1 = row.d1div1 + dist_d1/(points + 1)*iter_d1;
    row.start_point_d2 = row.d2div1 + dist_d2/(points + 1)*iter_d2;
  }
  else {
    // Is there a better way to do random numbers between 0 and 1?
    row.start_point_d1 = row.d1div1 + dist_d1*(double)rand()/((double)RAND_MAX);
    row.start_point_d2 = row.d2div1 + dist_d2*(double)rand()/((double)RAND_MAX);
  }
}

EntityHandle find_volume(std::map<EntityHandle, int> vol_handles_ids,
                         vec3 pt, vec3 dir) {
  // Figure out how to iterate over vols.
  // Really, figure out how to get vols in here.
  int result;
  std::map<EntityHandle, int>::iterator it;

  for ( it = vol_handles_ids.begin(); it != vol_handles_ids.end(); it++ ) {
    void* ptr;
    dag_pt_in_vol(it->first, pt, &result, dir, ptr);
    if (result)
      return it->first;
  }
  throw std::runtime_error("It appears that this point is not in any volume.");
}

} //namespace pyne
