#include "src/discretize.h"
#include <iostream>
#include <mpi.h>
#include <time.h>

int main(int argc, char* argv[]){
  clock_t t1, t2;
  int rank, num_proc;
  int num_rays = 10;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  bool grid = false;
  // This is needed only if you want to tell it how many voxels to take.
  // if (argc > 4) {
  int loop = 0;
  if (argc > 5) {
    throw std::runtime_error("discretize_geom called with too many argments.");
  } else if (argc < 2) {
    throw std::runtime_error("No geometry provided.");
  } else if (argc > 2) {
    if (isdigit(*argv[2]))
      num_rays = atoi(argv[2]);
    // if (argc == 4){
    if (argc >= 4){
      grid = (!strcmp(argv[3], "true") || !strcmp(argv[3], "1"));
    }
    // This is only needed if you're testing multiple iterations.
    if (argc == 5){
      if (isdigit(*argv[4]))
        loop = atoi(argv[4]);
    }
  }
  char* filename = argv[1];

  std::vector<std::vector<double> > mesh;
  mesh.resize(3);
  for (int i = 0; i < 3; i++) {
    mesh[i].push_back(-1);
    mesh[i].push_back(1);
  }
  
  // If checking each voxel in order...
  for (int n = 1; n < loop; n++) {
    int voxel_bound = n*2 + 1;
    mesh[0].push_back(voxel_bound);
  }
  if (rank == 0) {
    t1 = clock();
  }
  std::vector<std::vector<double> > results = discretize_geom(mesh, filename,
                                                              num_rays, grid);
  if (rank == 0) {
    
    t2 = clock();
    float diff((float)t2 - (float)t1);
    float seconds = diff/CLOCKS_PER_SEC;
    /*
    for (int i = 0; i < results.size(); i++) {
      std::cout << "[ ";
      for (int j = 0; j < results[i].size(); j++) {
        std::cout << results[i][j] << " ";
      }
      std::cout << "]" << std::endl;
    }
    */
    std::cout << seconds << " ";
  }
  MPI_Finalize();
}
