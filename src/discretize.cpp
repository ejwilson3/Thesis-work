#include "discretize.h"
#include <stdexcept>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include "dagmc_bridge.h"

namespace pyne{
//WATCH THIS
std::vector<sums> discretize_geom(std::vector<std::vector<double> > mesh,
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
  std::vector<sums> result;

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
        //WATCH THIS
        result.push_back(row.result);
      }
    }
  }
  
  /*
  std::cout << "C++ sums: ";
  for (int k = 0; k < result.size(); k++) {
    if (k != 0)
      std::cout << ", ";
    std::cout << result.at(k).sum << "-" << result.at(k).sqr_sum;
  }
  std::cout << "!" << std::endl;
  */

  return result;
}

void fireRays(mesh_row &row, std::map<EntityHandle, int> vol_handles_ids) {
  //std::cout << "6.1" << std::endl;
  //What is the vol?
  //WATCH THIS
  //vol vol;

  std::vector<double> width;
  for (int i = 0; i < row.d3divs.size() - 1; i++) {
    //std::cout << "6.2" << std::endl;
    width.push_back(row.d3divs.at(i+1) - row.d3divs.at(i));
  }

  for (int i = 0; i < row.num_rays; i++) {
    //std::cout << "6.3" << std::endl;
    int x, y;
    double a, b, c, d;
    startPoints(row, i);
    //TODO
    //if (!point_in_volume(vol, point, direction) {
    //  vol = find_volume(?);
    int vol_id = find_volume(row, vol_handles_ids);
    //}
    double value;

    //??????????????????????
    //WATCH THIS
    //Placeholder; distance should come from some dagmc thing.
    double distance = (rand() % 300)/100;
    //std::cout << "6.4" << std::endl;

    for (int j = 0; j < width.size(); j++) {
      do {
        if (distance >= width.at(j)) {
          value = 1;
          distance -= width.at(j);
        }
        else {
          value = distance/width.at(j);
        }
        //why did I add this?
        //double[2] sample;
        //WATCH THIS
        row.result.sum += value;
        row.result.sqr_sum += value*value;
      //figure out how to iterate to the next distance...
      } while (false);
      //std::cout << "6.5" << std::endl;
    }
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
    row.start_point_d1 = row.d1div1 + dist_d1*rand();
    row.start_point_d2 = row.d2div1 + dist_d2*rand();
  }
}

int find_volume(mesh_row &row, std::map<EntityHandle, int> vol_handles_ids) {
  // Figure out how to iterate over vols.
  // Really, figure out how to get vols in here.
  vec3 pt;
  pt[row.d3] = 0;
  pt[(row.d3+1)%3] = row.start_point_d1;
  pt[(row.d3+2)%3] = row.start_point_d2;
  int result;
  vec3 dir = {0};
  dir[row.d3] = 1;
//  dag_pt_in_vol(vol, pt, result, dir, history)
}

} //namespace pyne
