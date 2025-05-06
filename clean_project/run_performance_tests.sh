#!/bin/bash

# Output file for results
OUTPUT_FILE="performance_results.csv"

# Clear previous results
echo "vertices,processes,threads,edge_density,change_ratio,time_ms,affected_vertices" > $OUTPUT_FILE

# Function to run a test and append results
run_test() {
    local vertices=$1
    local processes=$2
    local threads=$3
    local density=$4
    local change_ratio=$5
    
    echo "Running test with vertices=$vertices, processes=$processes, threads=$threads, density=$density, change_ratio=$change_ratio"
    
    # Set OpenMP threads
    export OMP_NUM_THREADS=$threads
    
    # Run the test
    cd build
    mpirun -np $processes ./sssp_update $vertices $density $change_ratio $threads
    cd ..
    
    # Let's sleep a bit to ensure files are written
    sleep 1
}

# Test with increasing process counts (fixed threads=2, medium-sized graph)
for p in 1 2; do
    run_test 100 $p 2 0.1 0.01
done

# Test with increasing thread counts (fixed processes=2, medium-sized graph)
for t in 1 2 4; do
    run_test 100 2 $t 0.1 0.01
done

# Test with increasing graph sizes (fixed processes=2, threads=2)
for v in 50 100 200; do
    run_test $v 2 2 0.1 0.01
done

# Test with different edge densities (fixed processes=2, threads=2, medium-sized graph)
for d in 0.05 0.1 0.2; do
    run_test 100 2 2 $d 0.01
done

# Test with different change ratios (fixed processes=2, threads=2, medium-sized graph)
for c in 0.01 0.05 0.1; do
    run_test 100 2 2 0.1 $c
done

echo "Performance tests completed. Results saved to $OUTPUT_FILE" 