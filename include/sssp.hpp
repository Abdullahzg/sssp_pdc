#pragma once

#include "graph.hpp"
#include "common.hpp"
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <mpi.h>
#include <string>

class SSSP {
public:
    explicit SSSP(const Graph& graph);
    ~SSSP() = default;

    // Initialize SSSP with source vertex
    void initialize(int32_t source_vertex, MPI_Comm comm = MPI_COMM_WORLD);

    // Update SSSP with edge changes
    void update_sssp(const std::vector<EdgeChange>& edge_changes,
                    MPI_Comm comm = MPI_COMM_WORLD);

    // Get distance to vertex
    double get_distance(int32_t vertex) const;

    // Get parent of vertex
    int32_t get_parent(int32_t vertex) const;

    // Get number of affected vertices
    int32_t get_num_affected_vertices() const;

    // Check if vertex is ghost
    bool is_ghost_vertex(int32_t global_vertex) const;

    // Check if vertex is local
    bool is_local_vertex(int32_t global_vertex) const;

    // Get vertex owner
    int32_t get_vertex_owner(int32_t global_vertex) const;

    // Convert global vertex ID to local
    int32_t global_to_local(int32_t global_vertex) const;

    // Convert local vertex ID to global
    int32_t local_to_global(int32_t local_vertex) const;

    // Export graph and SSSP tree to JSON
    void export_to_json(const std::string& filename) const;
    
    // Export only edge changes to JSON
    void export_changes_to_json(const std::vector<EdgeChange>& changes, const std::string& filename) const;

private:
    const Graph& graph_;
    std::vector<double> distances_;      // Shortest distances from source
    std::vector<int32_t> parents_;       // Parent pointers in SSSP tree
    std::vector<bool> affected_;         // Vertices affected by updates
    std::vector<bool> affected_del_;     // Vertices affected by deletions
    std::vector<MessageBuffer> ghost_buffers_;  // Buffers for ghost vertex updates
    int32_t source_;                     // Source vertex
    bool has_source_;                    // Whether source is set

    // Process edge changes and identify affected vertices
    void process_edge_changes(const std::vector<EdgeChange>& changes);

    // Update affected vertices iteratively
    void update_affected_vertices(MPI_Comm comm);

    // Propagate deletion effects
    void propagate_deletion_effects(bool& changes);

    // Propagate updates
    void propagate_updates(bool& changes);

    // Synchronize ghost distances
    void synchronize_ghost_distances(MPI_Comm comm);

    // Gather results from all processes
    void gather_results(MPI_Comm comm);

    // Broadcast source vertex to all processes
    void broadcast_source(int32_t source, MPI_Comm comm);

    // Handle edge deletion
    void handle_edge_deletion(int32_t u, int32_t v, double weight,
                            std::unordered_set<int32_t>& affected);

    // Handle edge insertion
    void handle_edge_insertion(int32_t u, int32_t v, double weight,
                             std::unordered_set<int32_t>& affected);
}; 