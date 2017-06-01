#ifndef PYNE_DISCRETIZE_GEOM_H
#define PYNE_DISCRETIZE_GEOM_H

#include <vector>
#include <map>
#include <moab/Types.hpp>
#include "dagmc_bridge.h"

using moab::EntityHandle;

namespace pyne{


//Keep together those things which need to be passed to many different functions
struct mesh_row {
  // The number of rays fired each time
  int num_rays;
  // True if evenly spaced ray fires, false if random. If true, num_rays must be
  // a perfect square.
  bool grid;
  // The coordinates of the start of each ray fired.
  double start_point_d1, start_point_d2;
  // The different divisions that define the current row
  double d1div1, d1div2, d2div1, d2div2;
  // The divisions along the current row.
  std::vector<double> d3divs;
  // The index for the direction the row is facing (along which the rays will be
  // fired.
  int d3;
};

/*
 * This function discretizes the mesh attached to a geometry.
 *
 * Parameters:
 *    mesh:             a vector of three vectors, each containing the
 * coordinates of the divisions in one direction of the actual mesh.
 *    vol_handles_ids:  a map of the EntityHandles and ids of the various
 * volumes of the geometry..
 *    num_rays: The number of rays to be fired.
 *    grid:     Whether the rays are to be fired evenly spaced apart or at
 * random. True if evenly spaced.
 *
 * Returns:
 *    A vector of the results for each row, consisting of the sum of the values
 * given by the fired rays as well as the sum of the values squared.
*/
std::vector<std::vector<double> > discretize_geom(
    std::vector<std::vector<double> > mesh,
    std::map<EntityHandle, int> vol_handles_ids,
    int num_rays,
    bool grid);

/*
 * This function fires rays down a single row of the mesh.
 *
 * Parameters:
 *    row:  The various variables which define the current row, as well as how
 * the rays are to be fired.
 *    vol_handles_ids:  a map of the EntityHandles and ids of the various
 * volumes of the geometry..
 *
 * Output:
 *    row:  The "result" member of row will have been changed to reflect the
 * information obtained by firing the rays.
*/
std::vector<std::map<int, std::vector<double> > > fireRays(
    mesh_row &row,
    std::map<EntityHandle, int> vol_handles_ids);

/*
 * This function determines the starting coordinates for the next ray. It is
 * called from within fireRays.
 *
 * Parameters:
 *    row:  The various variables which define the current row, as well as how
 * the rays are to be fired.
 *    iter: The number of times this function has been called; only needed if
 * the rays are being fired in a grid; otherwise starting points are chosen
 * randomly.
 *
 * Output:
 *    row:  the "start_point_d1" and "start_point_d2" members of row will have
 * been changed to reflect the new start points.
*/
void startPoints(mesh_row &row, int iter);

/*
 * This function determines the current volume in which a point is.
 * Called from within fireRays.
 *
 * Parameters:
 *    vol_handles_ids: The map of entity handles and ids cooresponding to the
 * volumes.
 *    pt:              The point for which the volume is being found.
 *    dir:             The direction in which we're firing rays.
 *
 * Returns:
 *    eh:              The entity handle for the desired volume.
*/
EntityHandle find_volume(std::map<EntityHandle, int> vol_handles_ids, vec3 pt,
                         vec3 dir);

/*
 * This function determines the ids of the volume elements in the current row.
 *
 * Parameters:
 *    sizes:  The number of volume elements in each direction.
 *    d1:     The current volume element count in direction d1.
 *    d2:     The current volume element count in direction d2.
 *    d3:     This value, between 0 and 2, determines in which direction the
 * ray is being fired.
 *
 * Returns:
 *    A vector with the ids of each volume element in the current row.
*/
std::vector<int> get_idx(int sizes[], int d1,
                         int d2, int d3);

} //namespace pyne

#endif // PYNE_DISCRETIZE_GEOM_H
