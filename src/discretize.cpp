#include "discretize.h"
#include <stdexcept>
#include <stdlib.h>
#include <iostream>

namespace pyne{
//WATCH THIS
std::vector<disc_result> discretize_geom(std::vector<std::vector<double> > mesh,
                                         int num_rays, bool grid = false) {
  /*
  for (int i = 0; i < mesh.size(); i++) {
    std::cout << "<";
    for (int j = 0; j < mesh.at(i).size(); j++) {
      std::cout << mesh.at(i).at(j) << ", ";
    }
    std::cout << ">" << std::endl;
  }
  */

  /*
  if (!mesh.structured) {
    throw std::runtime_error("Unstructured mesh not supported at this time.");
  }
  else{
  */
    //WATCH THIS
    std::cout << num_rays << " and " << grid << std::endl;
    //MeshRow row(num_rays, grid);
    
    int d3;
//    std::vector<double> divs[3];
    //WATCH THIS
    std::vector<disc_result> sums;

    /*
    for (int i = 0; i < 3; i++) {
      divs[i] = mesh.structured_get_divisions(i);
      //Placeholder
      std::vector<double> foo;
      for (int j = 0; j < 10; i++) {
        foo.push_back(j*i/3);
      }
      divs[i] = foo;
    }
    */


    for (int d1 = 0; d1 < 3; d1++) {
      for (int i = 0; i < mesh.at(d1).size(); i++) {
        int d2;
        if (d1 = 2)
          d2 = 0;
        else
          d2 = d1 + 1;

        d3 = 3 - d1 - d2;
        //row.setDimension1(mesh.at(d1).at(i), mesh.at(d1).at(i+1));

        for (int j = 0; j < mesh.at(d2).size(); j++) {
          //row.setDimension2(mesh.at(d2).at(j), mesh.at(d2).at(j+1));
          //row.fireRays(d3, mesh.at(d3));
          //WATCH THIS
//          std::vector<disc_result> sums2 = row.getSums();
//          sums.insert(sums.end(), sums2.begin(), sums2.end());
        }
      }
    }
  //}
  return sums;
}


} //namespace pyne
