# Parallel SSSP Update Algorithm for Dynamic Graphs

This project implements a parallel algorithm for efficiently updating Single-Source Shortest Paths (SSSP) in large-scale dynamic graphs. It uses a hybrid MPI+OpenMP approach for combined distributed and shared-memory parallelism.

## Overview

The algorithm is based on the paper "A Parallel Algorithm Template for Updating Single-Source Shortest Paths in Large-Scale Dynamic Networks." The key insight is to only update the affected subgraphs when edge changes occur, rather than recomputing the entire SSSP.

### Features

- Distributed graph representation using MPI
- Shared-memory parallelism using OpenMP
- Efficient handling of edge insertions and deletions
- Support for large-scale dynamic networks
- Optimized communication of ghost vertices

## Building the Project

### Prerequisites

- C++17 compatible compiler
- CMake 3.10+
- OpenMPI 4.0+
- METIS 5.1+ for graph partitioning

### Compilation

```bash
mkdir -p build
cd build
cmake ..
make
```

## Running the Algorithm

The algorithm can be run with the following command:

```bash
mpirun -np <num_processes> ./sssp_update <num_vertices> <edge_density> <change_ratio> <num_threads>
```

where:
- `num_processes`: Number of MPI processes
- `num_vertices`: Number of vertices in the graph
- `edge_density`: Probability of edge creation (0.0-1.0)
- `change_ratio`: Ratio of edges to change (0.0-1.0)
- `num_threads`: Number of OpenMP threads per process

Example:
```bash
mpirun -np 2 ./sssp_update 100 0.1 0.01 2
```

## Algorithm Design

The algorithm follows these key steps:

1. **Graph Partitioning**: The graph is partitioned across MPI processes using METIS.
2. **Ghost Vertex Management**: Each process maintains "ghost" copies of vertices owned by other processes.
3. **Edge Change Processing**: When edges change (insertions/deletions), affected vertices are marked.
4. **Affected Subgraph Identification**: Only vertices whose shortest paths could be affected are processed.
5. **Parallel Update**: The affected subgraph is updated in parallel using OpenMP.
6. **Ghost Synchronization**: Updated shortest path information is communicated between processes.

## Key Optimizations

During the development, we implemented several optimizations to improve performance and stability:

1. **Vertex Mapping System**: 
   - Efficient translation between global and local vertex IDs
   - Proper initialization of the mapping data structures
   - Careful handling of ghost vertices

2. **MPI Communication**:
   - Simplified blocking communication pattern using `MPI_Sendrecv`
   - Eliminated race conditions and deadlocks that were causing segmentation faults
   - Efficient packing/unpacking of messages

3. **OpenMP Parallelism**:
   - Parallel processing of edge changes
   - Reduction operations to track global state changes
   - Dynamic work scheduling to handle load imbalance

4. **Memory Management**:
   - Proper initialization of data structures
   - Clear memory ownership semantics between processes
   - Careful handling of dynamic memory allocation

## Performance Analysis

We conducted a comprehensive performance analysis of the algorithm, examining:

1. **Strong Scaling**: Increasing processes/threads for fixed problem size
   - Good efficiency (0.83) when scaling from 1 to 2 MPI processes
   - Additional speedup from OpenMP parallelism (up to 2.0x with 4 threads)

2. **Problem Size Scaling**: Behavior with increasing graph size
   - Near-linear scaling for small to medium-sized graphs
   - Increased execution time with larger graphs due to algorithmic complexity

3. **Impact of Graph Properties**:
   - Denser graphs require more processing time (2-5ms for density 0.05-0.20)
   - Higher change ratios lead to larger affected subgraphs (3-7ms for ratios 0.01-0.10)

For detailed performance data and visualizations, see [performance_report.md](performance_report.md).

## Limitations and Future Work

Current limitations and opportunities for future work include:

1. Load balancing for highly irregular graphs
2. Optimized non-blocking communication patterns
3. Better detection of affected vertices to minimize redundant computations
4. Hierarchical parallelization for larger scale systems
5. Dynamic repartitioning for evolving graphs

## References

1. "A Parallel Algorithm Template for Updating Single-Source Shortest Paths in Large-Scale Dynamic Networks"
2. MPI: The Complete Reference
3. OpenMP Application Programming Interface

## License

This project is licensed under the MIT License - see the LICENSE file for details. 