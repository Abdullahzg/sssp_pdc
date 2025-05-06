#pragma once

#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <filesystem>

class Benchmark {
public:
    Benchmark();
    ~Benchmark() = default;

    // Start timing
    void start();

    // Stop timing
    void stop();

    // Get elapsed time
    double get_elapsed_time() const;

    // Write results to file
    void write_results(const std::string& filename,
                      int num_vertices,
                      int num_edges,
                      int num_processes,
                      int num_threads,
                      const std::string& operation);

    // Print results to console
    void print_results(const std::string& operation) const;

    // Gather and print statistics from all processes
    void gather_statistics();

private:
    double start_time_;
    double end_time_;
    double elapsed_time_;
}; 