#ifndef PYNE_DISCRETIZE_GEOM_H
#define PYNE_DISCRETIZE_GEOM_H

#include <vector>

namespace pyne{

struct disc_result {
  int id;
  int cell;
  double vol_frac;
  double rel_error;
};

std::vector<disc_result> discretize_geom(std::vector<std::vector<double> > mesh, int num_rays, bool grid);

class MeshRow {
  int num_rays;
  bool grid;
  //WATCH THIS
  std::vector<disc_result> sums;
  double start_point_x;
  double start_point_y;
  double d1div1, d1div2, d2div1, d2div2;
  std::vector<double> d3divs;

public:
  MeshRow(int num_rays, bool grid);
//  ~MeshRow(void);
  
  void setDimension1(double div1, double div2);
  void setDimension2(double div1, double div2);
  void fireRays(int d3, std::vector<double> divs);
  //WATCH THIS
  std::vector<disc_result> getSums(void);

private:
  void startPoints(int iter);
};

} //namespace pyne

#endif // PYNE_DISCRETIZE_GEOM_H
