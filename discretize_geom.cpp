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
  if (argc > 4) {
    throw std::runtime_error("discretize_geom called with too many argments.");
  } else if (argc < 2) {
    throw std::runtime_error("No geometry provided.");
  } else if (argc > 2) {
    if (isdigit(*argv[2]))
      num_rays = atoi(argv[2]);
    if (argc == 4){
      grid = (!strcmp(argv[3], "true") || !strcmp(argv[3], "1"));
    }
  }
  if (rank == 0) {
    t1 = clock();
  }
  char* filename = argv[1];

  std::vector<std::vector<double> > mesh;
  mesh.resize(3);
  for (int i = 0; i < 3; i++) {
    mesh[i].push_back(-4);
    mesh[i].push_back(-2);
    mesh[i].push_back(2);
    mesh[i].push_back(4);
  }
  std::vector<std::vector<double> > results = discretize_geom(mesh, filename,
                                                              num_rays, grid);
  if (rank == 0) {
    t2 = clock();
    float diff((float)t2-(float)t1);
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
    std::cout << seconds << std::endl;
  }
  MPI_Finalize();
}
