#include "benchmark.hpp"
#include "graph.hpp"
#include "sssp.hpp"
#include "graph_partitioner.hpp"
#include <fstream>
#include <random>
#include <chrono>
#include <iomanip>
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <sstream>
#include <ctime>

Benchmark::Benchmark()
    : start_time_(0.0),
      end_time_(0.0),
      elapsed_time_(0.0) {}

void Benchmark::run_experiments() {
    // Experiment parameters
    std::vector<int> num_vertices = {1000, 5000, 10000, 50000};
    std::vector<int> num_partitions = {1, 2, 4, 8, 16};
    std::vector<int> num_threads = {1, 2, 4, 8};
    std::vector<double> edge_densities = {0.1, 0.2, 0.5};
    std::vector<double> change_ratios = {0.01, 0.05, 0.1};

    // Run experiments for each configuration
    for (int vertices : num_vertices) {
        for (int partitions : num_partitions) {
            for (int threads : num_threads) {
                for (double density : edge_densities) {
                    for (double change_ratio : change_ratios) {
                        run_single_experiment(vertices, partitions, threads, density, change_ratio);
                    }
                }
            }
        }
    }

    // Generate summary report
    generate_summary_report();
}

void Benchmark::run_single_experiment(int num_vertices, int num_partitions,
                                    int num_threads, double edge_density,
                                    double change_ratio) {
    // Set OpenMP threads
    omp_set_num_threads(num_threads);

    // Create random graph
    Graph graph = create_random_graph(num_vertices, edge_density);

    // Partition graph
    GraphPartitioner partitioner(graph);
    partitioner.partition(num_partitions);

    // Create SSSP instance
    SSSP sssp(graph);

    // Create random edge changes
    std::vector<EdgeChange> changes = create_random_edge_changes(graph, change_ratio);

    // Start mpiP profiling
    mpiP_start();

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    
    // Initialize SSSP
    sssp.initialize(0);  // Use vertex 0 as source

    // Process edge changes
    sssp.update_sssp(changes);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Stop mpiP profiling and save results
    mpiP_stop();
    std::string mpiP_output = "mpiP_" + 
        std::to_string(num_vertices) + "_" +
        std::to_string(num_partitions) + "_" +
        std::to_string(num_threads) + ".mpiP";
    mpiP_write(mpiP_output.c_str());

    // Collect statistics
    ExperimentStats stats;
    stats.num_vertices = num_vertices;
    stats.num_partitions = num_partitions;
    stats.num_threads = num_threads;
    stats.edge_density = edge_density;
    stats.change_ratio = change_ratio;
    stats.execution_time = duration.count();
    stats.num_affected_vertices = sssp.get_num_affected_vertices();

    // Save results
    save_experiment_results(stats);
}

Graph Benchmark::create_random_graph(int num_vertices, double edge_density) {
    Graph graph(num_vertices);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> weight_dist(1.0, 100.0);
    std::uniform_real_distribution<> edge_dist(0.0, 1.0);

    // Add edges with given density
    for (int u = 0; u < num_vertices; u++) {
        for (int v = u + 1; v < num_vertices; v++) {
            if (edge_dist(gen) < edge_density) {
                double weight = weight_dist(gen);
                graph.add_edge(u, v, weight);
            }
        }
    }

    return graph;
}

std::vector<EdgeChange> Benchmark::create_random_edge_changes(
    const Graph& graph, double change_ratio) {
    std::vector<EdgeChange> changes;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> change_dist(0.0, 1.0);
    std::uniform_real_distribution<> weight_dist(1.0, 100.0);

    // Get all edges
    std::vector<std::pair<int32_t, int32_t>> edges;
    for (int u = 0; u < graph.get_num_vertices(); u++) {
        for (const auto& edge : graph.get_edges(u)) {
            if (u < edge.dest) {  // Add each edge only once
                edges.push_back({u, edge.dest});
            }
        }
    }

    // Randomly select edges to change
    int num_changes = static_cast<int>(edges.size() * change_ratio);
    std::shuffle(edges.begin(), edges.end(), gen);

    for (int i = 0; i < num_changes; i++) {
        EdgeChange change;
        change.u = edges[i].first;
        change.v = edges[i].second;
        change.weight = weight_dist(gen);
        change.is_deletion = change_dist(gen) < 0.5;  // 50% deletions, 50% insertions
        changes.push_back(change);
    }

    return changes;
}

void Benchmark::save_experiment_results(const ExperimentStats& stats) {
    std::string filename = "experiment_results.csv";
    std::ofstream file(filename, std::ios::app);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    // Write header if file is empty
    if (file.tellp() == 0) {
        file << "num_vertices,num_partitions,num_threads,edge_density,"
             << "change_ratio,execution_time,num_affected_vertices\n";
    }

    // Write results
    file << stats.num_vertices << ","
         << stats.num_partitions << ","
         << stats.num_threads << ","
         << stats.edge_density << ","
         << stats.change_ratio << ","
         << stats.execution_time << ","
         << stats.num_affected_vertices << "\n";
}

void Benchmark::generate_summary_report() {
    std::string filename = "summary_report.txt";
    std::ofstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    // Read experiment results
    std::string results_file = "experiment_results.csv";
    std::ifstream results(results_file);

    if (!results.is_open()) {
        throw std::runtime_error("Could not open file: " + results_file);
    }

    // Skip header
    std::string line;
    std::getline(results, line);

    // Process results
    std::map<std::tuple<int, int, int>, std::vector<double>> execution_times;
    std::map<std::tuple<int, int, int>, std::vector<int>> affected_vertices;

    while (std::getline(results, line)) {
        std::stringstream ss(line);
        std::string value;
        
        // Parse CSV line
        std::getline(ss, value, ','); int num_vertices = std::stoi(value);
        std::getline(ss, value, ','); int num_partitions = std::stoi(value);
        std::getline(ss, value, ','); int num_threads = std::stoi(value);
        std::getline(ss, value, ','); double edge_density = std::stod(value);
        std::getline(ss, value, ','); double change_ratio = std::stod(value);
        std::getline(ss, value, ','); double execution_time = std::stod(value);
        std::getline(ss, value, ','); int num_affected = std::stoi(value);

        // Store results
        auto key = std::make_tuple(num_vertices, num_partitions, num_threads);
        execution_times[key].push_back(execution_time);
        affected_vertices[key].push_back(num_affected);
    }

    // Generate summary
    file << "Performance Analysis Summary\n"
         << "==========================\n\n";

    // Analyze execution times
    file << "Execution Time Analysis\n"
         << "----------------------\n";
    
    for (const auto& [key, times] : execution_times) {
        auto [vertices, partitions, threads] = key;
        
        double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double min_time = *std::min_element(times.begin(), times.end());
        double max_time = *std::max_element(times.begin(), times.end());

        file << "Configuration: " << vertices << " vertices, "
             << partitions << " partitions, " << threads << " threads\n"
             << "Average time: " << avg_time << " ms\n"
             << "Min time: " << min_time << " ms\n"
             << "Max time: " << max_time << " ms\n\n";
    }

    // Analyze affected vertices
    file << "Affected Vertices Analysis\n"
         << "------------------------\n";
    
    for (const auto& [key, counts] : affected_vertices) {
        auto [vertices, partitions, threads] = key;
        
        double avg_count = std::accumulate(counts.begin(), counts.end(), 0.0) / counts.size();
        int min_count = *std::min_element(counts.begin(), counts.end());
        int max_count = *std::max_element(counts.begin(), counts.end());

        file << "Configuration: " << vertices << " vertices, "
             << partitions << " partitions, " << threads << " threads\n"
             << "Average affected vertices: " << avg_count << "\n"
             << "Min affected vertices: " << min_count << "\n"
             << "Max affected vertices: " << max_count << "\n\n";
    }

    // Calculate speedups
    file << "\nSpeedup Analysis\n"
         << "---------------\n";

    // Get sequential baseline (1 process, 1 thread)
    double baseline_time = 0.0;
    for (const auto& [key, times] : execution_times) {
        auto [vertices, partitions, threads] = key;
        if (partitions == 1 && threads == 1) {
            baseline_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
            break;
        }
    }

    // Calculate MPI speedups (varying processes, fixed threads)
    file << "\nMPI Speedup (Threads = 1)\n"
         << "------------------------\n";
    
    for (int threads = 1; threads <= 8; threads *= 2) {
        file << "\nThreads = " << threads << "\n";
        for (int partitions = 1; partitions <= 16; partitions *= 2) {
            auto key = std::make_tuple(10000, partitions, threads);  // Use 10k vertices as example
            if (execution_times.count(key) > 0) {
                double avg_time = std::accumulate(execution_times[key].begin(),
                                                execution_times[key].end(), 0.0) / 
                                                execution_times[key].size();
                double speedup = baseline_time / avg_time;
                file << "Processes = " << partitions << ": Speedup = " 
                     << std::fixed << std::setprecision(2) << speedup << "x\n";
            }
        }
    }

    // Calculate OpenMP speedups (varying threads, fixed processes)
    file << "\nOpenMP Speedup (Processes = 1)\n"
         << "----------------------------\n";
    
    for (int partitions = 1; partitions <= 16; partitions *= 2) {
        file << "\nProcesses = " << partitions << "\n";
        for (int threads = 1; threads <= 8; threads *= 2) {
            auto key = std::make_tuple(10000, partitions, threads);  // Use 10k vertices as example
            if (execution_times.count(key) > 0) {
                double avg_time = std::accumulate(execution_times[key].begin(),
                                                execution_times[key].end(), 0.0) / 
                                                execution_times[key].size();
                double speedup = baseline_time / avg_time;
                file << "Threads = " << threads << ": Speedup = " 
                     << std::fixed << std::setprecision(2) << speedup << "x\n";
            }
        }
    }

    // Calculate combined MPI+OpenMP speedups
    file << "\nCombined MPI+OpenMP Speedup\n"
         << "-------------------------\n";
    
    for (int partitions = 1; partitions <= 16; partitions *= 2) {
        for (int threads = 1; threads <= 8; threads *= 2) {
            auto key = std::make_tuple(10000, partitions, threads);  // Use 10k vertices as example
            if (execution_times.count(key) > 0) {
                double avg_time = std::accumulate(execution_times[key].begin(),
                                                execution_times[key].end(), 0.0) / 
                                                execution_times[key].size();
                double speedup = baseline_time / avg_time;
                file << "Processes = " << partitions << ", Threads = " << threads 
                     << ": Speedup = " << std::fixed << std::setprecision(2) 
                     << speedup << "x\n";
            }
        }
    }

    // Add MPI profiling analysis
    file << "\nMPI Performance Analysis\n"
         << "----------------------\n";
    
    // Process mpiP files
    for (const auto& [key, times] : execution_times) {
        auto [vertices, partitions, threads] = key;
        
        std::string mpiP_file = "mpiP_" + 
            std::to_string(vertices) + "_" +
            std::to_string(partitions) + "_" +
            std::to_string(threads) + ".mpiP";

        if (std::filesystem::exists(mpiP_file)) {
            file << "\nConfiguration: " << vertices << " vertices, "
                 << partitions << " partitions, " << threads << " threads\n";
            
            // Read and analyze mpiP file
            std::ifstream mpiP(mpiP_file);
            if (mpiP.is_open()) {
                std::string line;
                while (std::getline(mpiP, line)) {
                    // Extract relevant metrics
                    if (line.find("MPI_Allreduce") != std::string::npos ||
                        line.find("MPI_Isend") != std::string::npos ||
                        line.find("MPI_Irecv") != std::string::npos) {
                        file << line << "\n";
                    }
                }
            }
        }
    }
}

void Benchmark::start() {
    start_time_ = MPI_Wtime();
}

void Benchmark::stop() {
    end_time_ = MPI_Wtime();
    elapsed_time_ = end_time_ - start_time_;
}

double Benchmark::get_elapsed_time() const {
    return elapsed_time_;
}

void Benchmark::write_results(const std::string& filename,
                            int num_vertices,
                            int num_edges,
                            int num_processes,
                            int num_threads,
                            const std::string& operation) {
    std::ofstream outfile(filename, std::ios::app);
    if (!outfile) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // Get current timestamp
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream timestamp;
    timestamp << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");

    // Write results in CSV format
    outfile << timestamp.str() << ","
            << num_vertices << ","
            << num_edges << ","
            << num_processes << ","
            << num_threads << ","
            << operation << ","
            << std::fixed << std::setprecision(6) << elapsed_time_
            << std::endl;

    outfile.close();
}

void Benchmark::print_results(const std::string& operation) const {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::cout << "Operation: " << operation << std::endl
                  << "Elapsed time: " << std::fixed << std::setprecision(6)
                  << elapsed_time_ << " seconds" << std::endl;
    }
}

void Benchmark::gather_statistics() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Gather timing data from all processes
    std::vector<double> all_times(size);
    MPI_Gather(&elapsed_time_, 1, MPI_DOUBLE,
               all_times.data(), 1, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Calculate statistics
        double min_time = *std::min_element(all_times.begin(), all_times.end());
        double max_time = *std::max_element(all_times.begin(), all_times.end());
        double avg_time = std::accumulate(all_times.begin(), all_times.end(), 0.0) / size;

        // Print statistics
        std::cout << "Performance Statistics:" << std::endl
                  << "  Minimum time: " << std::fixed << std::setprecision(6) << min_time << " seconds" << std::endl
                  << "  Maximum time: " << max_time << " seconds" << std::endl
                  << "  Average time: " << avg_time << " seconds" << std::endl
                  << "  Load imbalance: " << (max_time - min_time) / avg_time * 100.0 << "%" << std::endl;
    }
} 