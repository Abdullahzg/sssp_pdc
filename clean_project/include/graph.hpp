#pragma once

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <mpi.h>
#include <cstdint>
#include <utility>

// Edge structure
struct Edge {
    int32_t dest;
    double weight;
};

class Graph {
public:
    explicit Graph(int32_t num_vertices);
    ~Graph() = default;

    // Add edge to graph
    void add_edge(int32_t src, int32_t dest, double weight);

    // Remove edge from graph
    void remove_edge(int32_t src, int32_t dest);

    // Get number of vertices
    int32_t get_num_vertices() const { return num_vertices_; }

    // Get number of ghost vertices
    int32_t get_num_ghost_vertices() const { return num_ghost_vertices_; }

    // Get neighbors of vertex
    const std::vector<std::pair<int32_t, double>>& get_neighbors(int32_t vertex) const;

    // Get ghost vertices
    const std::unordered_set<int32_t>& get_ghost_vertices() const { return ghost_vertices_; }

    // Get vertex owner
    int32_t get_vertex_owner(int32_t vertex) const;

    // Convert global vertex to local
    int32_t global_to_local(int32_t global_vertex) const;

    // Convert local vertex to global
    int32_t local_to_global(int32_t local_vertex) const;

    // Check if vertex is ghost
    bool is_ghost_vertex(int32_t vertex) const;

    // Distribute graph across processes
    void distribute_graph(const std::vector<int32_t>& partition_assignments);

    // Gather graph from all processes
    void gather_graph(MPI_Comm comm = MPI_COMM_WORLD);

    // Update ghost vertices
    void update_ghost_vertices(MPI_Comm comm = MPI_COMM_WORLD);

    // Setup vertex mappings
    void setup_vertex_mappings(const std::vector<int32_t>& partition_assignments);

    // Identify ghost vertices
    void identify_ghost_vertices(const std::vector<int32_t>& partition_assignments);

private:
    int32_t num_vertices_;
    std::vector<std::vector<std::pair<int32_t, double>>> adjacency_list_;
    std::unordered_set<int32_t> ghost_vertices_;
    int32_t num_ghost_vertices_;
    int32_t local_size_;

    // Vertex mapping arrays
    std::vector<int32_t> vertex_owner_;    // Global vertex -> owner rank
    std::vector<int32_t> global_to_local_; // Global vertex -> local index
    std::vector<int32_t> local_to_global_; // Local index -> global vertex
}; 