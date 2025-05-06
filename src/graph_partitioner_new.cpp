#include "graph_partitioner.hpp"
#include <metis.h>
#include <mpi.h>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <iostream>

GraphPartitioner::GraphPartitioner(Graph& graph)
    : graph_(graph) {
}

void GraphPartitioner::partition(int num_partitions) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get graph information
    int32_t num_vertices = graph_.get_num_vertices();
    std::vector<int32_t> xadj;  // Adjacency list offsets
    std::vector<int32_t> adjncy;  // Adjacency list
    std::vector<int32_t> adjwgt;  // Edge weights

    // Convert graph to METIS format
    convert_to_metis_format(xadj, adjncy, adjwgt);

    // METIS options
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_NUMBERING] = 0;  // 0-based indexing

    // Partition parameters
    idx_t nvtxs = num_vertices;
    idx_t ncon = 1;  // Number of balancing constraints
    idx_t nparts = num_partitions;
    idx_t objval;  // Edge cut
    std::vector<idx_t> part(num_vertices);

    // Convert to idx_t for METIS
    std::vector<idx_t> idx_xadj(xadj.begin(), xadj.end());
    std::vector<idx_t> idx_adjncy(adjncy.begin(), adjncy.end());
    std::vector<idx_t> idx_adjwgt(adjwgt.begin(), adjwgt.end());

    // Perform partitioning
    int result = METIS_PartGraphKway(&nvtxs, &ncon, idx_xadj.data(), idx_adjncy.data(),
                                   nullptr, nullptr, idx_adjwgt.data(), &nparts,
                                   nullptr, nullptr, options, &objval, part.data());

    if (result != METIS_OK) {
        throw std::runtime_error("METIS partitioning failed");
    }

    // Convert partition assignments to int32_t
    std::vector<int32_t> partition_assignments(num_vertices);
    std::copy(part.begin(), part.end(), partition_assignments.begin());

    // Distribute graph according to partition assignments
    graph_.distribute_graph(partition_assignments);

    if (rank == 0) {
        std::cout << "Graph partitioned into " << num_partitions << " parts" << std::endl;
        std::cout << "Edge cut: " << objval << std::endl;
    }
}

void GraphPartitioner::convert_to_metis_format(std::vector<int32_t>& xadj,
                                             std::vector<int32_t>& adjncy,
                                             std::vector<int32_t>& adjwgt) {
    int32_t num_vertices = graph_.get_num_vertices();

    // Initialize xadj
    xadj.resize(num_vertices + 1);
    xadj[0] = 0;

    // Count edges and build xadj
    for (int32_t v = 0; v < num_vertices; ++v) {
        const auto& neighbors = graph_.get_neighbors(v);
        xadj[v + 1] = xadj[v] + neighbors.size();
    }

    // Initialize adjncy and adjwgt
    int32_t num_edges = xadj[num_vertices];
    adjncy.resize(num_edges);
    adjwgt.resize(num_edges);

    // Fill adjncy and adjwgt
    int32_t edge_idx = 0;
    for (int32_t v = 0; v < num_vertices; ++v) {
        const auto& neighbors = graph_.get_neighbors(v);
        for (const auto& [neighbor, weight] : neighbors) {
            adjncy[edge_idx] = neighbor;
            adjwgt[edge_idx] = static_cast<int32_t>(weight * 100);  // Convert to integer weights
            ++edge_idx;
        }
    }
}

void GraphPartitioner::setup_vertex_mappings(const std::vector<int32_t>& partition_assignments) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Clear existing mappings
    global_to_local_.clear();
    local_to_global_.clear();

    // Count local vertices
    local_size_ = 0;
    for (int32_t v = 0; v < graph_.get_num_vertices(); v++) {
        if (partition_assignments[v] == rank) {
            local_to_global_[local_size_] = v;
            global_to_local_[v] = local_size_;
            local_size_++;
        }
    }
}

void GraphPartitioner::identify_ghost_vertices(const std::vector<int32_t>& partition_assignments) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Clear existing ghost vertices
    ghost_vertices_.clear();
    num_ghost_vertices_ = 0;

    // For each local vertex, check its neighbors
    for (int32_t v = 0; v < graph_.get_num_vertices(); v++) {
        if (partition_assignments[v] == rank) {
            for (const auto& [neighbor, _] : graph_.get_neighbors(v)) {
                if (partition_assignments[neighbor] != rank) {
                    ghost_vertices_.insert(neighbor);
                }
            }
        }
    }

    // Sort and remove duplicates
    std::vector<int32_t> sorted_ghosts(ghost_vertices_.begin(), ghost_vertices_.end());
    std::sort(sorted_ghosts.begin(), sorted_ghosts.end());
    ghost_vertices_.clear();
    for (int32_t ghost : sorted_ghosts) {
        if (ghost_vertices_.insert(ghost).second) {
            num_ghost_vertices_++;
        }
    }
} 