#include "graph.hpp"
#include <algorithm>
#include <stdexcept>
#include <iostream>

Graph::Graph(int32_t num_vertices)
    : num_vertices_(num_vertices),
      adjacency_list_(num_vertices),
      num_ghost_vertices_(0),
      local_size_(0) {
    // Initialize vertex mappings
    vertex_owner_.resize(num_vertices, -1);
    global_to_local_.resize(num_vertices, -1);
    local_to_global_.resize(num_vertices, -1);
}

void Graph::add_edge(int32_t src, int32_t dest, double weight) {
    if (src < 0 || src >= num_vertices_ || dest < 0 || dest >= num_vertices_) {
        throw std::invalid_argument("Invalid vertex index");
    }

    // Add edge in both directions (undirected graph)
    adjacency_list_[src].emplace_back(dest, weight);
    adjacency_list_[dest].emplace_back(src, weight);
}

void Graph::remove_edge(int32_t src, int32_t dest) {
    if (src < 0 || src >= num_vertices_ || dest < 0 || dest >= num_vertices_) {
        throw std::invalid_argument("Invalid vertex index");
    }

    // Remove edge in both directions
    auto& src_edges = adjacency_list_[src];
    src_edges.erase(
        std::remove_if(src_edges.begin(), src_edges.end(),
                      [dest](const auto& edge) { return edge.first == dest; }),
        src_edges.end());

    auto& dest_edges = adjacency_list_[dest];
    dest_edges.erase(
        std::remove_if(dest_edges.begin(), dest_edges.end(),
                      [src](const auto& edge) { return edge.first == src; }),
        dest_edges.end());
}

const std::vector<std::pair<int32_t, double>>& Graph::get_neighbors(int32_t vertex) const {
    if (vertex < 0 || vertex >= num_vertices_) {
        throw std::invalid_argument("Invalid vertex index");
    }
    return adjacency_list_[vertex];
}

int32_t Graph::get_vertex_owner(int32_t vertex) const {
    if (vertex < 0 || vertex >= num_vertices_) {
        throw std::invalid_argument("Invalid vertex index");
    }
    return vertex_owner_[vertex];
}

int32_t Graph::global_to_local(int32_t global_vertex) const {
    if (global_vertex < 0 || global_vertex >= num_vertices_) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::cout << "Rank " << rank << ": ERROR - Invalid global vertex index " << global_vertex 
                  << " (num_vertices_ = " << num_vertices_ << ")" << std::endl;
        throw std::invalid_argument("Invalid global vertex index");
    }
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (global_to_local_[global_vertex] < 0) {
        std::cout << "Rank " << rank << ": WARNING - Global vertex " << global_vertex 
                  << " has no local mapping (mapping = " << global_to_local_[global_vertex] << ")" << std::endl;
    }
    return global_to_local_[global_vertex];
}

int32_t Graph::local_to_global(int32_t local_vertex) const {
    if (local_vertex < 0 || local_vertex >= local_size_) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::cout << "Rank " << rank << ": ERROR - Invalid local vertex index " << local_vertex 
                  << " (local_size_ = " << local_size_ << ")" << std::endl;
        throw std::invalid_argument("Invalid local vertex index");
    }
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (local_to_global_[local_vertex] < 0) {
        std::cout << "Rank " << rank << ": WARNING - Local vertex " << local_vertex 
                  << " has no global mapping (mapping = " << local_to_global_[local_vertex] << ")" << std::endl;
    }
    return local_to_global_[local_vertex];
}

bool Graph::is_ghost_vertex(int32_t vertex) const {
    return ghost_vertices_.find(vertex) != ghost_vertices_.end();
}

void Graph::setup_vertex_mappings(const std::vector<int32_t>& partition_assignments) {
    if (partition_assignments.size() != num_vertices_) {
        throw std::invalid_argument("Invalid partition assignments size");
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "Rank " << rank << ": Setting up vertex mappings for " << num_vertices_ << " vertices" << std::endl;

    // Clear existing mappings
    std::fill(vertex_owner_.begin(), vertex_owner_.end(), -1);
    std::fill(global_to_local_.begin(), global_to_local_.end(), -1);
    std::fill(local_to_global_.begin(), local_to_global_.end(), -1);
    ghost_vertices_.clear();

    // Get process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Setup mappings for local vertices
    local_size_ = 0;
    for (int32_t v = 0; v < num_vertices_; ++v) {
        vertex_owner_[v] = partition_assignments[v];
        if (partition_assignments[v] == rank) {
            global_to_local_[v] = local_size_;
            local_to_global_[local_size_] = v;
            local_size_++;
        }
    }
    
    std::cout << "Rank " << rank << ": Assigned " << local_size_ << " local vertices" << std::endl;

    // Identify ghost vertices
    identify_ghost_vertices(partition_assignments);
    
    std::cout << "Rank " << rank << ": Identified " << num_ghost_vertices_ << " ghost vertices" << std::endl;
}

void Graph::identify_ghost_vertices(const std::vector<int32_t>& partition_assignments) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Clear existing ghost vertices
    ghost_vertices_.clear();

    std::cout << "Rank " << rank << ": Identifying ghost vertices..." << std::endl;

    // Identify ghost vertices (vertices owned by other processes but connected to local vertices)
    for (int32_t v = 0; v < num_vertices_; ++v) {
        if (partition_assignments[v] == rank) {
            std::cout << "Rank " << rank << ": Checking neighbors of local vertex " << v << std::endl;
            for (const auto& [neighbor, _] : adjacency_list_[v]) {
                if (partition_assignments[neighbor] != rank) {
                    ghost_vertices_.insert(neighbor);
                    std::cout << "Rank " << rank << ": Adding ghost vertex " << neighbor << " (owned by rank " 
                              << partition_assignments[neighbor] << ")" << std::endl;
                }
            }
        }
    }

    num_ghost_vertices_ = ghost_vertices_.size();
    std::cout << "Rank " << rank << ": Ghost vertices identified: " << num_ghost_vertices_ << std::endl;
}

void Graph::distribute_graph(const std::vector<int32_t>& partition_assignments) {
    setup_vertex_mappings(partition_assignments);
}

void Graph::gather_graph(MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Gather adjacency lists from all processes
    for (int r = 0; r < size; ++r) {
        if (r != rank) {
            // Receive number of vertices
            int32_t num_vertices;
            MPI_Recv(&num_vertices, 1, MPI_INT32_T, r, 0, comm, MPI_STATUS_IGNORE);

            // Receive adjacency lists
            for (int32_t v = 0; v < num_vertices; ++v) {
                int32_t num_edges;
                MPI_Recv(&num_edges, 1, MPI_INT32_T, r, 0, comm, MPI_STATUS_IGNORE);

                std::vector<std::pair<int32_t, double>> edges(num_edges);
                MPI_Recv(edges.data(), num_edges * sizeof(std::pair<int32_t, double>),
                        MPI_BYTE, r, 0, comm, MPI_STATUS_IGNORE);

                adjacency_list_[v] = std::move(edges);
            }
        }
    }
}

void Graph::update_ghost_vertices(MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Exchange ghost vertex information
    for (int r = 0; r < size; ++r) {
        if (r != rank) {
            // Send ghost vertices to process r
            std::vector<int32_t> ghost_vertices(ghost_vertices_.begin(), ghost_vertices_.end());
            int32_t num_ghosts = ghost_vertices.size();
            MPI_Send(&num_ghosts, 1, MPI_INT32_T, r, 0, comm);
            MPI_Send(ghost_vertices.data(), num_ghosts, MPI_INT32_T, r, 0, comm);

            // Receive ghost vertices from process r
            MPI_Recv(&num_ghosts, 1, MPI_INT32_T, r, 0, comm, MPI_STATUS_IGNORE);
            std::vector<int32_t> received_ghosts(num_ghosts);
            MPI_Recv(received_ghosts.data(), num_ghosts, MPI_INT32_T, r, 0, comm, MPI_STATUS_IGNORE);

            // Update ghost vertices
            ghost_vertices_.insert(received_ghosts.begin(), received_ghosts.end());
        }
    }

    num_ghost_vertices_ = ghost_vertices_.size();
} 