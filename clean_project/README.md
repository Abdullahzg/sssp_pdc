# Parallel SSSP Update Algorithm for Dynamic Graphs

This project implements a parallel algorithm for efficiently updating Single-Source Shortest Paths (SSSP) in large-scale dynamic graphs. It uses a hybrid MPI+OpenMP approach for combined distributed and shared-memory parallelism, optimized for handling edge insertions and deletions without recomputing the entire shortest path tree.

## Project Description

### Problem Statement

Single-Source Shortest Path (SSSP) is a fundamental graph problem that finds the shortest paths from a single source vertex to all other vertices in a graph. In dynamic graphs, where edges can be inserted or deleted over time, recomputing the entire SSSP after each change is inefficient. This project implements an efficient parallel algorithm that only updates the affected subgraphs when edge changes occur.

### Implementation Approach

The algorithm is based on the paper "A Parallel Algorithm Template for Updating Single-Source Shortest Paths in Large-Scale Dynamic Networks." Our implementation includes:

1. **Graph Distribution**: The graph is partitioned across MPI processes using the METIS library, minimizing edge cuts to reduce inter-process communication.

2. **Two-Phase Algorithm**:
   - **Phase 1**: Identify affected vertices based on edge changes
   - **Phase 2**: Iteratively update the shortest paths only for affected vertices and their neighbors

3. **Hybrid Parallelism**:
   - **MPI**: For distributed memory parallelism across multiple compute nodes
   - **OpenMP**: For shared memory parallelism within each compute node

### Features

- Distributed graph representation using MPI
- Shared-memory parallelism using OpenMP
- Efficient handling of edge insertions and deletions
- Support for large-scale dynamic networks
- Optimized communication of ghost vertices
- Comprehensive performance analysis tools

## Building the Project

### Prerequisites

- C++17 compatible compiler (GCC 7.0+ or equivalent)
- CMake 3.10+
- OpenMPI 4.0+ or another MPI implementation
- METIS 5.1+ for graph partitioning
- Python 3.6+ with pandas, matplotlib, and numpy (for performance analysis)

### Obtaining METIS

If METIS is not installed on your system:

```bash
# Download METIS
wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
tar -xzf metis-5.1.0.tar.gz
cd metis-5.1.0

# Build METIS
make config shared=1
make
sudo make install
```

### Compilation

```bash
# Create build directory
mkdir -p build
cd build

# Configure and build
cmake ..
make -j4

# Return to project root
cd ..
```

## Running the Algorithm

### Basic Usage

The algorithm can be run with the following command:

```bash
mpirun -np <num_processes> ./build/sssp_update <num_vertices> <edge_density> <change_ratio> <num_threads>
```

where:
- `num_processes`: Number of MPI processes
- `num_vertices`: Number of vertices in the graph
- `edge_density`: Probability of edge creation (0.0-1.0)
- `change_ratio`: Ratio of edges to change (0.0-1.0)
- `num_threads`: Number of OpenMP threads per process

### Examples

1. Small test with 2 processes and 2 threads per process:
```bash
mpirun -np 2 ./build/sssp_update 100 0.1 0.01 2
```

2. Medium-sized test with more threads:
```bash
mpirun -np 4 ./build/sssp_update 500 0.1 0.05 4
```

3. Large-scale test:
```bash
mpirun -np 8 ./build/sssp_update 1000 0.05 0.01 4
```

### Setting OpenMP Threads

You can explicitly set the number of OpenMP threads using the environment variable:

```bash
export OMP_NUM_THREADS=4
mpirun -np 2 ./build/sssp_update 100 0.1 0.01 4
```

## Running Performance Tests

We provide scripts to run a comprehensive performance analysis of the algorithm:

### 1. Basic Performance Tests

```bash
# Make the script executable if needed
chmod +x run_performance_tests.sh

# Run the standard set of performance tests
./run_performance_tests.sh
```

This script will test the algorithm with:
- Different numbers of MPI processes
- Different numbers of OpenMP threads
- Different graph sizes
- Different edge densities
- Different edge change ratios

Results are saved to `performance_results.csv`.

### 2. Analyzing Performance Results

```bash
# Make sure you have the required Python packages
pip install pandas matplotlib numpy

# Run the analysis script
python analyze_performance.py performance_results.csv
```

This will generate:
- Performance graphs (PNG files) for various metrics
- A detailed analysis of scaling efficiency
- Performance comparisons across different configurations

### 3. MPI Profiling (Advanced)

For detailed MPI performance analysis:

```bash
# Make the script executable if needed
chmod +x run_mpip_analysis.sh

# Run the profiling script
./run_mpip_analysis.sh
```

This requires the mpiP library and generates detailed reports on MPI communication patterns and potential bottlenecks.

## Algorithm Design Details

The algorithm follows these key steps:

1. **Graph Partitioning**: The graph is partitioned across MPI processes using METIS.
   - Each process owns a subset of vertices and their outgoing edges
   - Processes maintain "ghost" copies of non-local vertices that are adjacent to local vertices

2. **Ghost Vertex Management**: 
   - Ghost vertices are identified during graph construction
   - Mapping systems translate between global and local vertex IDs
   - Ghost vertex information is synchronized when needed

3. **Edge Change Processing**: 
   - Edge changes (insertions/deletions) are processed in parallel
   - For deletions: If an edge was in the SSSP tree, mark affected vertices
   - For insertions: Check if the new edge provides a shorter path

4. **Affected Subgraph Identification**: 
   - Identify vertices whose shortest paths could be affected by edge changes
   - Use boolean flags (Affected[] and Affected_Del[]) to track these vertices

5. **Parallel Update**: 
   - Iteratively update shortest paths in the affected subgraph
   - Use OpenMP for thread-level parallelism within each MPI process

6. **Ghost Synchronization**: 
   - Synchronize updated distance information between processes
   - Use MPI communication to exchange ghost vertex updates

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

## Troubleshooting

### Common Issues

1. **Compilation Errors:**
   - Ensure all dependencies are installed
   - Check that METIS is correctly linked in CMakeLists.txt

2. **MPI Runtime Errors:**
   - If you see "not enough slots available", reduce the number of MPI processes
   - For deadlocks, try running with fewer processes or a smaller graph

3. **Performance Analysis Issues:**
   - Ensure Python with required packages is installed
   - Check that performance_results.csv has been generated

### Debugging

For detailed debugging output, you can modify the code to add more print statements:
- In `graph.cpp`: Add prints to check vertex mappings
- In `sssp_new.cpp`: Add prints to track affected vertices
- In `graph_partitioner.cpp`: Add prints to verify partitioning

## References

1. "A Parallel Algorithm Template for Updating Single-Source Shortest Paths in Large-Scale Dynamic Networks"
2. Karypis, G., & Kumar, V. (1998). A fast and high quality multilevel scheme for partitioning irregular graphs. SIAM Journal on scientific Computing, 20(1), 359-392.
3. MPI: The Complete Reference
4. OpenMP Application Programming Interface Version 5.0

## License

This project is available under the MIT License.

## Contributors

This implementation was developed as part of a Parallel and Distributed Computing course project. 