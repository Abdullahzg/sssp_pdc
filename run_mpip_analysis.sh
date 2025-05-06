#!/bin/bash

# This script runs a series of tests with mpiP profiling enabled
# to analyze MPI communication patterns in the SSSP update algorithm

# Check if mpiP is installed
if [ ! -f "mpiP-3.5/lib/libmpiP.a" ]; then
    echo "Building mpiP from source..."
    cd mpiP-3.5
    ./configure
    make
    cd ..
fi

# Compile with mpiP
echo "Recompiling with mpiP support..."
cd build
cmake -DENABLE_MPIP=ON ..
make clean
make
cd ..

# Run tests with mpiP profiling
echo "Running performance tests with mpiP profiling..."

# Create directory for profiling results
mkdir -p mpip_results

# Test scenarios with different process counts
for p in 1 2; do
    # Set a consistent test case
    vertices=100
    threads=2
    density=0.1
    change_ratio=0.01
    
    echo "Running test with: $p processes, $threads threads, $vertices vertices"
    
    # Set environment variables for mpiP
    export MPIP="-f -o -t 1.0 -s"
    
    # Run the test with mpiP
    cd build
    mpirun -np $p ./sssp_update $vertices $density $change_ratio $threads
    cd ..
    
    # Move profiling results to the results directory
    mv build/mpiP-*.*.* mpip_results/ 2>/dev/null || echo "No profiling results found for test with $p processes"
done

echo "MPI profiling completed. Check the mpip_results/ directory for detailed performance data." 