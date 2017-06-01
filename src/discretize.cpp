#include "discretize.h"
#include <stdexcept>
#include <stdlib.h>
#include <math.h>
#include <iostream>

// The maxiumum volume fraction to be considered valid
float VOL_FRAC_TOLERANCE = 1e-10;

namespace pyne{

std::vector<std::vector<double> > discretize_geom(
    std::vector<std::vector<double> > mesh,
    std::map<EntityHandle, int> vol_handles_ids,
    int num_rays,
    bool grid = false) {

  // This will store the information of the individual row and how the rays
  // are to be fired.
  struct mesh_row row;
  row.num_rays = num_rays;
  row.grid = grid;

  // Initialize this here to avoid creating it multiple times during loops.
  std::vector<double> zeros(2,0.0);
  // This will be the end result.
  std::vector<std::vector<double> > result;
  // This stores the intermediate totals.
  std::vector<std::map<int, std::vector<double> > > row_totals;
  // Define the size now to use [] assignment.
  row_totals.resize((mesh[0].size()-1)*(mesh[1].size()-1)*(mesh[2].size()-1));

  // These for loops visit each individual row.
  for (int d1 = 0; d1 < 3; d1++) {

    // Set up the different direction indices.
    int d2 = (d1 + 1)%3;

    row.d3 = 3 - d1 - d2;
    row.d3divs = mesh[row.d3];

    // Iterate over each starting volume element in the d1/d2 directions.
    for (int i = 0; i < mesh[d1].size() - 1; i++) {
      row.d1div1 = mesh[d1][i];
      row.d1div2 = mesh[d1][i+1];

      for (int j = 0; j < mesh[d2].size() - 1; j++) {
        row.d2div1 = mesh[d2][j];
        row.d2div2 = mesh[d2][j+1];
        int sizes[] = {mesh[0].size() - 1, mesh[1].size() - 1, mesh[2].size() - 1};
        std::vector<int> idx = get_idx(sizes, i, j, row.d3);

        // The rays are fired and totals collected here.
        std::vector<std::map<int, std::vector<double> > > ray_totals =
            fireRays(row, vol_handles_ids);

        // Combine the totals collected from each direction by volume element.
        for (int k = 0; k < ray_totals.size(); k++) {
          for (std::map<int, std::vector<double> >::iterator ray_it =
              ray_totals[k].begin(); ray_it != ray_totals[k].end();
              ++ray_it) {
            if (ray_it->second[0] < VOL_FRAC_TOLERANCE)
              continue;
            std::map<int, std::vector<double> >::iterator row_it =
                row_totals[idx[k]].find(ray_it->first);
            if (row_it == row_totals[idx[k]].end()){
              row_totals[idx[k]].insert(row_it,
                                        std::make_pair(ray_it->first, zeros));
            }
            row_totals[idx[k]][ray_it->first][0] += ray_it->second[0];
            row_totals[idx[k]][ray_it->first][1] += ray_it->second[1];
          }
        }
      }
    }
  }


  // Evaluate the results and prepare them for return.
  int total_rays = num_rays*3;
  std::vector<double> ray_results(4,0.0);
  for (int i = 0; i < row_totals.size(); i++) {
    for (std::map<int, std::vector<double> >::iterator it =
         row_totals[i].begin(); it != row_totals[i].end(); ++it) {
      ray_results[0] = i;
      ray_results[1] = it->first;
      ray_results[2] = it->second[0]/total_rays;
      ray_results[3] = sqrt((it->second[1])/pow((it->second[0]),2.0)
                                    - 1.0/total_rays);
      result.push_back(ray_results);
    }
  }
  return result;
}

std::vector<std::map<int, std::vector<double> > > fireRays(
    mesh_row &row,
    std::map<EntityHandle, int> vol_handles_ids) {

  std::vector<std::map<int, std::vector<double> > > row_totals;
  std::vector<double> width;
  for (int i = 0; i < row.d3divs.size() - 1; i++) {
    width.push_back(row.d3divs[i+1] - row.d3divs[i]);
  }
  row_totals.resize(width.size());

  // These variables are needed for dag_pt_in_vol and dag_ray_follow.
  vec3 pt;
  pt[row.d3] = row.d3divs[0];
  int result = 0;
  vec3 dir = {0};
  dir[row.d3] = 1;
  EntityHandle eh;
  int num_intersections;
  EntityHandle *surfs, *volumes;
  double *distances;
  void* buf;

  for (int i = 0; i < row.num_rays; i++) {
    startPoints(row, i);
    pt[(row.d3+1)%3] = row.start_point_d1;
    pt[(row.d3+2)%3] = row.start_point_d2;
    // If the next point starts in the same volume as the last one, calling this
    // here can save calling find_volume, which gets expensive.
    if (i != 0)
      dag_pt_in_vol(eh, pt, &result, dir, NULL);
    if (!result)
      eh = find_volume(vol_handles_ids, pt, dir);


    dag_ray_follow(eh, pt, dir, 0.0, &num_intersections,
                   &surfs, &distances, &volumes, buf);

    std::vector<double> zeros(2,0.0);
    double value;
    // Keep track of at which volume element we're currently looking.
    int count = 0;
    // Keep track of the "current" intersection in the geometry.
    int intersection = 0;
    bool complete = false;
    double curr_width = width[count];
    int vol_id = vol_handles_ids[eh];
    // Loops over the ray's intersections with the different volumes.
    while(!complete) {
      // while the distance to the next intersection is greater than the current
      // width, we calculate the value and we move to the next width, shortening
      // the distance.
      while (distances[intersection] >= curr_width) {
        value = curr_width/width[count];
        std::map<int, std::vector<double> >::iterator it =
            row_totals[count].find(vol_id);
        // If the current cell isn't in the totals yet, add it.
        if (it == row_totals[count].end()){
          row_totals[count].insert(it, std::make_pair(vol_id, zeros));
        }
        row_totals[count][vol_id][0] += value;
        row_totals[count][vol_id][1] += value*value;
        distances[intersection] -= curr_width;

        // If there are more volume elements, move to the next one.
        if (width.size() - 1 > count) {
          count++;
          curr_width = width[count];
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
      if (distances[intersection] < curr_width &&
          distances[intersection] > VOL_FRAC_TOLERANCE &&
          !complete) {
        value = distances[intersection]/curr_width;
        std::map<int, std::vector<double> >::iterator it =
            row_totals[count].find(vol_id);

        // If the current cell isn't in the totals yet, add it.
        if (it == row_totals[count].end()){
          row_totals[count].insert(it, std::make_pair(vol_id, zeros));
        }
        row_totals[count][vol_id][0] += value;
        row_totals[count][vol_id][1] += value*value;
        curr_width -= distances[intersection];
      }
      vol_id = vol_handles_ids[volumes[intersection]];
      intersection++;
    }
    dag_dealloc_ray_buffer(buf);
  }
  return row_totals;
}

void startPoints(mesh_row &row, int iter) {
  double dist_d1 = row.d1div2 - row.d1div1;
  double dist_d2 = row.d2div2 - row.d2div1;
  if (row.grid) {
    int points = sqrt(row.num_rays);
    //Iterate slowly from 1 to points once.
    int iter_d1 = iter/points + 1;
    //Iterate from 1 to points, repeat points times.
    int iter_d2 = iter%points + 1;

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
  int result;
  std::map<EntityHandle, int>::iterator it;

  // Check each volume in the list. This is why it can take so long.
  for (it = vol_handles_ids.begin(); it != vol_handles_ids.end(); it++) {
    void* ptr;
    dag_pt_in_vol(it->first, pt, &result, dir, ptr);
    if (result)
      return it->first;
  }
  throw std::runtime_error("It appears that this point is not in any volume.");
}

std::vector<int> get_idx(int sizes[], int d1, int d2, int d3) {
  std::vector<int> idx;
  // The ids iterate over x, then y, then z. Which one of those is defined by
  // which variable depends on d3.
  if (d3 == 0){
    for (int i = 0; i < sizes[d3]; i++) {
      idx.push_back(i + sizes[0]*d1 + sizes[0]*sizes[1]*d2);
    }
  }
  else if (d3 == 1){
    for (int i = 0; i < sizes[d3]; i++) {
      idx.push_back(d2 + sizes[0]*i + sizes[0]*sizes[1]*d1);
    }
  }
  else if (d3 == 2){
    for (int i = 0; i < sizes[d3]; i++) {
      idx.push_back(d1 + sizes[0]*d2 + sizes[0]*sizes[1]*i);
    }
  }
  return idx;
}

} //namespace pyne
