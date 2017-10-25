#include "discretize.h"
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <stdexcept>
#include <stdlib.h>
#include <moab/CartVect.hpp>
#include <moab/Range.hpp>

#define CHECKERR(err) \
    if((err) != moab::MB_SUCCESS){ \
      std::cout << "Error on line " << __LINE__ << "of discretize.cpp" \
                << std::endl; \
      return err;}

#define CHECKERR_HERE(err) \
    if((err) != moab::MB_SUCCESS) \
      std::cout << "Error on line " << __LINE__ << "of discretize.cpp" \
                << std::endl;

using moab::CartVect;
using moab::Core;
using moab::GeomTopoTool;
using moab::ErrorCode;
using moab::EntityHandle;
using moab::GeomQueryTool;

float VOL_FRAC_TOLERANCE = 1e-10;
moab::Interface* MBI;
GeomTopoTool* GTT;
GeomQueryTool* GQT;

std::vector<std::vector<double> > discretize_geom(
    std::vector<std::vector<double> > mesh,
    const char* filename,
    int num_rays,
    bool grid) {

  // vol_handles holds a list of all of the EntityHandles in the given file.
  ErrorCode rval;
  std::vector<EntityHandle> vol_handles;
  rval = load_geometry(filename, &vol_handles);
  CHECKERR_HERE(rval);
  // This will hold the information about this individual row.
  struct mesh_row row;
  row.num_rays = num_rays;
  row.grid = grid;
  if (grid && (pow((int)sqrt(num_rays), 2) != num_rays))
    throw std::runtime_error("For rays fired in a grid, num_rays must be "
                             "a perfect square.");

  // Initialize these here to avoid creating it multiple times during loops.
  // This is just for adding new elements to row_totals to use +=.
  std::vector<double> zeros(2,0.0);
  // This is needed for the function get_idx, in order to determine the
  // identities of the elements in the current row. It is also used often to
  // help with separating the work between cores, as it contains the number of
  // volume elements in each direction.
  int sizes[] = {mesh[0].size() - 1, mesh[1].size() - 1,
                  mesh[2].size() - 1};

  // This will be the end result.
  std::vector<std::vector<double> > result;
  // This stores the intermediate totals.
  std::vector<std::map<int, std::vector<double> > > row_totals;
  // Define the size now to use [] assignment.
  row_totals.resize(sizes[0]*sizes[1]*sizes[2]);
  int total_iter = (sizes[0]*sizes[1] + sizes[0]*sizes[2] +
                    sizes[1]*sizes[2])*row.num_rays;
  int rank, num_proc, remainder, loc_iter, init, fin;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

  // These define the number of iterations done by each processor. The cores of
  // rank less than remainder will fire one ray more than those of rank equal
  // or greater than it.
  remainder = total_iter%(num_proc);
  loc_iter = total_iter/(num_proc) + (rank < remainder);
  if (rank < remainder) {
    init = rank*loc_iter;
    fin = init + loc_iter;
  } else {
    init = rank*loc_iter + remainder;
    fin = init + loc_iter;
  }

  // The work is divided into three directions. These determine the first and
  // last directions down which rays will be fired by each processor, the z
  // directions, if you will.
  int d1_i = (init > (sizes[0]*sizes[1])*row.num_rays) +
             (init > (sizes[0]*sizes[1] + sizes[1]*sizes[2])*row.num_rays);
  int d1_f = (fin > (sizes[0]*sizes[1])*row.num_rays) +
             (fin > (sizes[0]*sizes[1] + sizes[1]*sizes[2])*row.num_rays) + 1;
  // These will determine where to start and end on the other two directions,
  // the "x's" and "y's".
  int init2, fin2, init3, fin3;

  // These for loops visit each individual row.
  for (int d1 = d1_i; d1 < d1_f; d1++) {
    // Set up the different direction indices.
    int d2 = (d1 + 1)%3;
    row.d3 = 3 - d1 - d2;
    row.d3divs = mesh[row.d3];
    // These keep track of where we start and end based off of this current
    // direction rather than overall.
    int init_step, fin_step;
    if (d1 == d1_i) {
      init_step = init - ((d1>0)*sizes[0]*sizes[1]*row.num_rays +
                          (d1==2)*sizes[1]*sizes[2]*row.num_rays);
      init2 = floor(init_step/row.num_rays/sizes[d2]);
    } else {
      // If you've moved on to an additional d1, the others will need to start
      // at zero. Likewise for determining init3.
      init_step = 0;
      init2 = 0;
    }
    if (d1 == (d1_f - 1)) {
      fin_step = fin - ((d1>0)*sizes[0]*sizes[1]*row.num_rays +
                        (d1==2)*sizes[1]*sizes[2]*row.num_rays);
      fin2 = ceil((float)fin_step/row.num_rays/sizes[d2]);
    } else {
      // Likewise, if you're not on the last d1, you'll need to end at the last
      // possible. Likewise for determining fin3.
      fin_step = sizes[d1]*sizes[d2]*row.num_rays;
      fin2 = sizes[d1];
    }

    // Iterate over each starting volume element in the d1/d2 directions.
    for (int i = init2; i < fin2; i++) {
      row.d1div1 = mesh[d1][i];
      row.d1div2 = mesh[d1][i+1];
      if (i == init2 && d1 == d1_i) {
        init3 = floor((init_step - i*sizes[d2]*row.num_rays)/row.num_rays);
      } else {
        init3 = 0;
      }
      if (i == (fin2 - 1) && d1 == (d1_f - 1)) {
        fin3 = ceil(((float)fin_step - i*sizes[d2]*row.num_rays)/row.num_rays);
      } else {
        fin3 = sizes[d2];
      }

      for (int j = init3; j < fin3; j++) {
        row.d2div1 = mesh[d2][j];
        row.d2div2 = mesh[d2][j+1];
        std::vector<int> idx = get_idx(sizes, i, j, row.d3);
        if (j == init3 && i == init2 && d1 == d1_i) {
          row.init = init%row.num_rays;
        } else {
          row.init = 0;
        }
        if (j == (fin3 - 1) && i == (fin2 -1) && d1 == (d1_f - 1)) {
          row.fin = ((fin - 1)%row.num_rays) + 1;
        } else {
          row.fin = row.num_rays;
        }

        // The rays are fired and totals collected here.
        std::vector<std::map<int, std::vector<double> > > ray_totals =
            fireRays(row, vol_handles);

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

  // The number of unique values on this processor, for passing information.
  int elem_count = 0;
  for (int i = 0; i < row_totals.size(); i++) {
    elem_count += row_totals[i].size();
  }
  // Begin combining results here. max_count is needed in case some processes
  // end up with a greater number of results than others.
  int max_count;
  MPI_Allreduce(&elem_count, &max_count, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  int total_count;
  // total_count will be used to allocate space, but it's only needed on rank 0.
  if (rank != 0) {
    total_count = 1;
  } else {
    total_count = max_count*num_proc;
  }
  // These will hold the combined results after the MPI_Gather function calls.
  int all_partitions[total_count];
  EntityHandle all_ehs[total_count];
  double all_sums[total_count], all_sqr_sums[total_count];
  // These will hold the results of each ray, to be combined.
  int mesh_partition[max_count];
  EntityHandle ehs[max_count];
  double sums[max_count], sqr_sums[max_count];
  // If there are fewer results than max_count, fill the rest of mesh_partiton
  // with -1 to avoid trying to use garbage as data.
  for (int i = elem_count; i < max_count; i++) {
    mesh_partition[i] = -1;
  }

  // Easy way to keep track of the current element in an array while using
  // a combination of ints and iterators.
  int cur_count = 0;
  // Fill the arrays with the results for use with MPI_Gather.
  for (int i = 0; i < row_totals.size(); i++) {
    for (std::map<int, std::vector<double> >::iterator it =
        row_totals[i].begin(); it != row_totals[i].end(); ++it) {
      mesh_partition[cur_count] = i;
      ehs[cur_count] = it->first;
      sums[cur_count] = it->second[0];
      sqr_sums[cur_count] = it->second[1];
      cur_count++;
    }
  }
  MPI_Gather(&mesh_partition, max_count, MPI_INT, &all_partitions,
             max_count, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gather(&ehs, max_count, MPI_LONG, &all_ehs,
             max_count, MPI_LONG, 0, MPI_COMM_WORLD);
  MPI_Gather(&sums, max_count, MPI_DOUBLE, &all_sums,
             max_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(&sqr_sums, max_count, MPI_DOUBLE, &all_sqr_sums,
             max_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // The largest partition number, for correctly sizing the combined totals.
  int max_partition;
  MPI_Reduce(&mesh_partition[elem_count - 1], &max_partition, 1,
             MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  // Now put all of the results back and combine them more cleanly.
  if (rank == 0){
    std::vector<std::map<int, std::vector<double> > > totals;
    totals.resize(max_partition + 1);
    for (int i = 0; i < total_count; i++) {
      // If it's -1, it came from one of the processors with less than max_count
      // elements in its array, and needs to be skipped.
      if (all_partitions[i] < 0) {
        continue;
      }
      // If the current cell isn't in the totals yet, add it.
      std::map<int, std::vector<double> >::iterator it =
          totals[all_partitions[i]].find(all_ehs[i]);
      if (it == totals[all_partitions[i]].end()){
        std::vector<double> both_sums;
        both_sums.push_back(all_sums[i]);
        both_sums.push_back(all_sqr_sums[i]);
        totals[all_partitions[i]].insert(it, std::make_pair(all_ehs[i],
                                         both_sums));
      } else {
        totals[all_partitions[i]][all_ehs[i]][0] += all_sums[i];
        totals[all_partitions[i]][all_ehs[i]][1] += all_sqr_sums[i];
      }
    }

  // Evaluate the results and prepare them for return.
    int total_rays = num_rays*3;
    std::vector<double> ray_results(4,0.0);
    for (int i = 0; i < totals.size(); i++) {
      for (std::map<int, std::vector<double> >::iterator it =
          totals[i].begin(); it != totals[i].end(); ++it) {
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
}

std::vector<std::map<int, std::vector<double> > > fireRays(
    mesh_row &row, std::vector<EntityHandle> vol_handles) {

  std::vector<std::map<int, std::vector<double> > > row_totals;
  // This is the difference between each pair of divisions.
  std::vector<double> width;
  ErrorCode rval;
  for (int i = 0; i < row.d3divs.size() - 1; i++) {
    width.push_back(row.d3divs[i+1] - row.d3divs[i]);
  }
  row_totals.resize(width.size());

  // These variables are needed for point_in_volume and dag_ray_follow.
  vec3 pt;
  pt[row.d3] = row.d3divs[0];
  int result = 0;
  vec3 dir = {0};
  dir[row.d3] = 1;
  EntityHandle eh;
  int num_intersections;
  EntityHandle *surfs, *volumes;
  double *distances;

  for (int i = row.init; i < row.fin; i++) {

    startPoints(row, i);
    pt[(row.d3+1)%3] = row.start_point_d1;
    pt[(row.d3+2)%3] = row.start_point_d2;
    // If the next point starts in the same volume as the last one, calling
    // this here can save calling find_volume, which gets expensive.
    if (i != row.init)
      GQT->point_in_volume(eh, pt, result, dir);
    if (!result){
      eh = find_volume(vol_handles, pt, dir);
    }

    // This stores information from dag_ray_follow.
    ray_buffers* buf = new ray_buffers;
    rval = dag_ray_follow(eh, pt, dir, 0.0, &num_intersections,
                   &surfs, &distances, &volumes, buf);
    CHECKERR_HERE(rval);

    std::vector<double> zeros(2,0.0);
    // This will hold the numbers to add to the totals.
    double value;
    // Keep track of at which volume element we're currently looking.
    int count = 0;
    // Keep track of the "current" intersection in the geometry.
    int intersection = 0;
    double curr_width = width[count];
    bool complete = false;
    // Loops over the ray's intersections with the different volumes.
    while(!complete) {
      // while the distance to the next intersection is greater than the
      // current width, we calculate the value and we move to the next width,
      // shortening the distance.
      while (distances[intersection] >= curr_width) {
        // If the current cell isn't in the totals yet, add it.
        std::map<int, std::vector<double> >::iterator it =
            row_totals[count].find(eh);
        if (it == row_totals[count].end()){
          row_totals[count].insert(it, std::make_pair(eh, zeros));
        }

        value = curr_width/width[count];
        row_totals[count][eh][0] += value;
        row_totals[count][eh][1] += value*value;
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

        // If the current cell isn't in the totals yet, add it.
        std::map<int, std::vector<double> >::iterator it =
            row_totals[count].find(eh);
        if (it == row_totals[count].end()){
          row_totals[count].insert(it, std::make_pair(eh, zeros));
        }

        value = distances[intersection]/width[count];
        row_totals[count][eh][0] += value;
        row_totals[count][eh][1] += value*value;
        curr_width -= distances[intersection];
      }
      if (intersection < num_intersections) {
        eh = volumes[intersection];
        intersection++;
      } else {
        complete = true;
        break;
      }
    }
    delete buf;
  }

  return row_totals;
}

void startPoints(mesh_row &row, int iter) {
  // These are the dimensions of the current row.
  double dist_d1 = row.d1div2 - row.d1div1;
  double dist_d2 = row.d2div2 - row.d2div1;
  if (row.grid) {
    int points = sqrt(row.num_rays);
    // Iterate slowly from 1 to points once.
    int iter_d1 = iter/points + 1;
    // Iterate from 1 to points, repeat points times.
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

EntityHandle find_volume(std::vector<EntityHandle> vol_handles,
                         vec3 pt, vec3 dir) {
  int result = 0;
  ErrorCode rval;

  // Check against each volume in the list. This is why it can take so long.
  for (int i = 0; i < vol_handles.size(); i++) {
    void* ptr;
    rval = GQT->point_in_volume(vol_handles[i], pt, result, dir,
        static_cast<const GeomQueryTool::RayHistory*>(ptr));
    CHECKERR_HERE(rval);
    if (result)
      return vol_handles[i];
  }
  std::cerr << "(" << pt[0] << ", " << pt[1] <<", "<< pt[2] << ")" << std::endl;
  throw std::runtime_error("It appears that this point is not in any volume.");
}

std::vector<int> get_idx(int sizes[], int d1, int d2, int d3) {
  std::vector<int> idx;
  // The ids iterate over the three directions in such a way that it matches
  // with that done by PyNE. Which direction is defined by which coordinate
  // depends on d3.
  if (d3 == 0){
    for (int i = 0; i < sizes[d3]; i++) {
      idx.push_back(d2 + sizes[2]*d1 + sizes[1]*sizes[2]*i);
    }
  }
  else if (d3 == 1){
    for (int i = 0; i < sizes[d3]; i++) {
      idx.push_back(d1 + sizes[2]*i + sizes[1]*sizes[2]*d2);
    }
  }
  else if (d3 == 2){
    for (int i = 0; i < sizes[d3]; i++) {
      idx.push_back(i + sizes[2]*d2 + sizes[1]*sizes[2]*d1);
    }
  }
  return idx;
}

// Dagmc bridge functions


ErrorCode dag_ray_follow(EntityHandle firstvol, vec3 ray_start, vec3 ray_dir,
                         double distance_limit, int* num_intersections,
                         EntityHandle** surfs, double** distances,
                         EntityHandle** volumes, ray_buffers* buf){

  ErrorCode err;

  EntityHandle vol = firstvol;
  double dlimit = distance_limit;
  CartVect ray_point(ray_start);
  EntityHandle next_surf;
  double next_surf_dist;

  CartVect uvw(ray_dir);

  // iterate over the ray until no more intersections are available
  while(vol) {
    err = GQT->ray_fire(vol, ray_point.array(), ray_dir, next_surf,
                        next_surf_dist, &(buf->history), dlimit);
    CHECKERR(err);

    if(next_surf) {
      ray_point += uvw * next_surf_dist;
      buf->surfs.push_back(next_surf);
      buf->dists.push_back(next_surf_dist);
      err = GQT->gttool()->next_vol(next_surf, vol, vol);
      CHECKERR(err);
      buf->vols.push_back(vol);
      if(dlimit != 0){
        dlimit -= next_surf_dist;
      }
    }
    else vol = 0;
  }

  // assign to the output variables
  *num_intersections = buf->surfs.size();
  *surfs = &(buf->surfs[0]);
  *distances = &(buf->dists[0]);
  *volumes = &(buf->vols[0]);

  return err;
}

ErrorCode load_geometry(const char* filename,
                        std::vector<EntityHandle>* vol_handles){
  std::vector<int> volList;

  // Initialize the tools taken from MOAB
  MBI = new Core();
  ErrorCode err = MBI->load_file(filename);
  CHECKERR(err);
  GTT = new GeomTopoTool(MBI);
  GQT = new GeomQueryTool(GTT);
  err = GQT->initialize();
  CHECKERR(err);

  int num_vols = GQT->gttool()->num_ents_of_dim(3);

  moab::Range vols;
  err = GQT->gttool()->get_gsets_by_dimension(3, vols);
  CHECKERR(err);
  vol_handles->resize(num_vols);
  int i = 0;
  for (moab::Range::iterator it = vols.begin(); it != vols.end(); ++it) {
    vol_handles->at(i++) = *it;
  }

  return err;
}
