#include "graph.hpp"
#include "sssp.hpp"
#include "common.hpp"
#include "graph_partitioner.hpp"
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <random>
#include <vector>
#include <limits>
#include <fstream>

// Create a test graph with specified number of vertices and edge density
Graph create_test_graph(int num_vertices, double edge_density) {
    std::cout << "Creating test graph with " << num_vertices << " vertices..." << std::endl;
    Graph graph(num_vertices);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> weight_dist(1.0, 10.0);
    std::uniform_real_distribution<> edge_dist(0.0, 1.0);

    // Add edges with probability edge_density
    for (int i = 0; i < num_vertices; ++i) {
        for (int j = i + 1; j < num_vertices; ++j) {
            if (edge_dist(gen) < edge_density) {
                double weight = weight_dist(gen);
                graph.add_edge(i, j, weight);
            }
        }
    }
    std::cout << "Test graph created successfully." << std::endl;
    return graph;
}

// Create random edge changes
std::vector<EdgeChange> create_random_edge_changes(const Graph& graph, double change_ratio) {
    std::cout << "Creating random edge changes..." << std::endl;
    std::vector<EdgeChange> changes;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> weight_dist(1.0, 10.0);
    std::uniform_real_distribution<> change_dist(0.0, 1.0);
    std::uniform_int_distribution<> vertex_dist(0, graph.get_num_vertices() - 1);

    // Calculate number of changes based on ratio
    int num_changes = static_cast<int>(graph.get_num_vertices() * change_ratio);
    
    for (int i = 0; i < num_changes; ++i) {
        EdgeChange change;
        change.u = vertex_dist(gen);
        do {
            change.v = vertex_dist(gen);
        } while (change.v == change.u);
        
        change.weight = weight_dist(gen);
        change.is_deletion = change_dist(gen) < 0.5;  // 50% chance of deletion
        changes.push_back(change);
    }
    std::cout << "Created " << changes.size() << " edge changes." << std::endl;
    return changes;
}

// Print SSSP distances
void print_sssp_distances(const std::vector<double>& distances) {
    std::cout << "SSSP Distances from source:" << std::endl;
    for (size_t v = 0; v < distances.size(); ++v) {
        if (distances[v] == std::numeric_limits<double>::infinity())
            std::cout << "Vertex " << v << ": INF" << std::endl;
        else
            std::cout << "Vertex " << v << ": " << distances[v] << std::endl;
    }
}

// Save performance results to file
void save_performance_results(const std::string& filename,
                            int num_vertices,
                            int num_partitions,
                            int num_threads,
                            double edge_density,
                            double change_ratio,
                            double execution_time,
                            int num_affected) {
    std::ofstream file(filename, std::ios::app);
    if (file.is_open()) {
        file << num_vertices << ","
             << num_partitions << ","
             << num_threads << ","
             << edge_density << ","
             << change_ratio << ","
             << execution_time << ","
             << num_affected << "\n";
        file.close();
    }
}

int main(int argc, char** argv) {
    std::cout << "Initializing MPI..." << std::endl;
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse command line arguments
    int num_vertices = 1000;  // Default values
    double edge_density = 0.1;
    double change_ratio = 0.01;
    int num_threads = omp_get_max_threads();

    if (argc > 1) num_vertices = std::stoi(argv[1]);
    if (argc > 2) edge_density = std::stod(argv[2]);
    if (argc > 3) change_ratio = std::stod(argv[3]);
    if (argc > 4) num_threads = std::stoi(argv[4]);

    // Set OpenMP threads
    omp_set_num_threads(num_threads);

    if (rank == 0) {
        std::cout << "Running SSSP Update with:" << std::endl
                  << "  Vertices: " << num_vertices << std::endl
                  << "  Edge Density: " << edge_density << std::endl
                  << "  Change Ratio: " << change_ratio << std::endl
                  << "  MPI Processes: " << size << std::endl
                  << "  OpenMP Threads: " << num_threads << std::endl;
    }

    try {
        std::cout << "Rank " << rank << ": Creating test graph..." << std::endl;
        // Create and partition graph
        Graph graph = create_test_graph(num_vertices, edge_density);
        
        std::cout << "Rank " << rank << ": Creating graph partitioner..." << std::endl;
        GraphPartitioner partitioner(graph);
        
        std::cout << "Rank " << rank << ": Partitioning graph..." << std::endl;
        partitioner.partition(size);  // Partition into size parts

        std::cout << "Rank " << rank << ": Creating SSSP instance..." << std::endl;
        // Create SSSP instance
        SSSP sssp(graph);

        std::cout << "Rank " << rank << ": Creating random edge changes..." << std::endl;
        // Create random edge changes
        std::vector<EdgeChange> changes = create_random_edge_changes(graph, change_ratio);

        // Export edge changes to JSON
        if (rank == 0) {
            sssp.export_changes_to_json(changes, "edge_changes.json");
        }

        std::cout << "Rank " << rank << ": Initializing SSSP..." << std::endl;
        // Initialize SSSP
        sssp.initialize(0);  // Use vertex 0 as source

        // Export initial graph and SSSP tree to JSON
        sssp.export_to_json("graph_before.json");

        // Start timing
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();

        std::cout << "Rank " << rank << ": Processing edge changes..." << std::endl;
        // Process edge changes
        sssp.update_sssp(changes);

        // End timing
        MPI_Barrier(MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Export updated graph and SSSP tree to JSON
        sssp.export_to_json("graph_after.json");

        // Gather and print results on rank 0
        if (rank == 0) {
            std::cout << "\nExecution time: " << duration.count() << " ms" << std::endl;
            std::cout << "Number of affected vertices: " << sssp.get_num_affected_vertices() << std::endl;

            // Save performance results
            save_performance_results("performance_results.csv",
                                   num_vertices,
                                   size,
                                   num_threads,
                                   edge_density,
                                   change_ratio,
                                   duration.count(),
                                   sssp.get_num_affected_vertices());
        }
    } catch (const std::exception& e) {
        std::cerr << "Rank " << rank << ": Error: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::cout << "Rank " << rank << ": Finalizing MPI..." << std::endl;
    MPI_Finalize();
    return 0;
} 