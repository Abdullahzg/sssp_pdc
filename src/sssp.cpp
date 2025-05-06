#include "sssp.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <omp.h>
#include <mpi.h>
#include <vector>
#include <queue>
#include <unordered_set>
#include <atomic>
#include <fstream>
#include <iostream>
#include <sstream>

// Structure for ghost vertex updates
struct GhostUpdateMessage {
    int32_t vertex_id;
    double distance;
    int32_t parent;
    bool is_deletion;
    double edge_weight;
};

// Structure for managing non-blocking MPI communication
struct MessageBuffer {
    std::vector<GhostUpdateMessage> messages;
    std::vector<MPI_Request> requests;
    std::atomic<int> current_request;

    MessageBuffer() : current_request(0) {}

    void add_message(const GhostUpdateMessage& msg) {
        messages.push_back(msg);
    }

    void start_send(int dest_rank, MPI_Comm comm) {
        int idx = current_request++;
        MPI_Isend(&messages[idx], sizeof(GhostUpdateMessage), MPI_BYTE,
                  dest_rank, 0, comm, &requests[idx]);
    }

    void wait_all() {
        if (current_request > 0) {
            MPI_Waitall(current_request, requests.data(), MPI_STATUSES_IGNORE);
            current_request = 0;
            messages.clear();
        }
    }
};

SSSP::SSSP(const Graph& graph)
    : graph_(graph),
      distances_(graph.get_num_vertices(), std::numeric_limits<double>::infinity()),
      parents_(graph.get_num_vertices(), -1),
      source_(-1),
      has_source_(false) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    ghost_buffers_.resize(world_size);
}

void SSSP::initialize(int32_t source_vertex, MPI_Comm comm) {
    if (source_vertex < 0 || source_vertex >= graph_.get_num_vertices()) {
        throw std::invalid_argument("Invalid source vertex");
    }

    // Reset distances and parents
    std::fill(distances_.begin(), distances_.end(), std::numeric_limits<double>::infinity());
    std::fill(parents_.begin(), parents_.end(), -1);
    affected_vertices_.clear();

    // Set source vertex
    distances_[source_vertex] = 0.0;
    has_source_ = true;
    source_ = source_vertex;

    // Broadcast source vertex to all processes
    broadcast_source(source_vertex, comm);
}

void SSSP::update_sssp(const std::vector<EdgeChange>& edge_changes, MPI_Comm comm) {
    if (!has_source_) {
        throw std::runtime_error("SSSP not initialized with source vertex");
    }

    // Process edge changes
    process_edge_changes(edge_changes);

    // Update affected vertices
    update_affected_vertices(comm);

    // Synchronize ghost distances
    synchronize_ghost_distances();

    // Gather results from all processes
    gather_results(comm);
}

void SSSP::process_edge_changes(const std::vector<EdgeChange>& changes) {
    std::unordered_set<int32_t> affected;

    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < changes.size(); ++i) {
        const auto& change = changes[i];
        if (change.is_deletion) {
            handle_edge_deletion(change.u, change.v, change.weight, affected);
        } else {
            handle_edge_insertion(change.u, change.v, change.weight, affected);
        }
    }

    affected_vertices_ = std::move(affected);
}

void SSSP::handle_edge_deletion(int32_t u, int32_t v, double weight,
                              std::unordered_set<int32_t>& affected) {
    // Check if the deleted edge was part of the shortest path
    if (parents_[v] == u || parents_[u] == v) {
        affected.insert(v);
        affected.insert(u);
        propagate_deletion_effects();
    }
}

void SSSP::handle_edge_insertion(int32_t u, int32_t v, double weight,
                               std::unordered_set<int32_t>& affected) {
    // Check if the new edge creates a shorter path
    double u_dist = distances_[u];
    double v_dist = distances_[v];

    if (u_dist + weight < v_dist) {
        distances_[v] = u_dist + weight;
        parents_[v] = u;
        affected.insert(v);
    }

    if (v_dist + weight < u_dist) {
        distances_[u] = v_dist + weight;
        parents_[u] = v;
        affected.insert(u);
    }
}

void SSSP::update_affected_vertices(MPI_Comm comm) {
    bool changes = true;
    while (changes) {
        changes = false;
        propagate_updates();
        process_ghost_updates();
        synchronize_ghost_distances();

        // Check for global convergence
        int local_changes = changes ? 1 : 0;
        int global_changes;
        MPI_Allreduce(&local_changes, &global_changes, 1, MPI_INT, MPI_SUM, comm);
        changes = global_changes > 0;
    }
}

void SSSP::propagate_deletion_effects() {
    std::queue<int32_t> queue;
    for (int32_t v : affected_vertices_) {
        queue.push(v);
    }

    while (!queue.empty()) {
        int32_t v = queue.front();
        queue.pop();

        // Reset distance and parent
        distances_[v] = std::numeric_limits<double>::infinity();
        parents_[v] = -1;

        // Add children to queue
        for (const auto& [child, weight] : graph_.get_neighbors(v)) {
            if (parents_[child] == v) {
                queue.push(child);
                affected_vertices_.insert(child);
            }
        }
    }
}

void SSSP::propagate_updates() {
    std::queue<int32_t> queue;
    for (int32_t v : affected_vertices_) {
        queue.push(v);
    }

    while (!queue.empty()) {
        int32_t v = queue.front();
        queue.pop();

        double v_dist = distances_[v];
        for (const auto& [neighbor, weight] : graph_.get_neighbors(v)) {
            double new_dist = v_dist + weight;
            if (new_dist < distances_[neighbor]) {
                distances_[neighbor] = new_dist;
                parents_[neighbor] = v;
                queue.push(neighbor);
                affected_vertices_.insert(neighbor);
            }
        }
    }
}

void SSSP::process_ghost_updates() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (int r = 0; r < world_size; ++r) {
        if (r == rank) continue;

        auto& buffer = ghost_buffers_[r];
        for (const auto& msg : buffer.messages) {
            int32_t v = msg.vertex_id;
            double new_dist = msg.distance;
            if (new_dist < distances_[v]) {
                distances_[v] = new_dist;
                parents_[v] = msg.parent;
                affected_vertices_.insert(v);
            }
        }
        buffer.messages.clear();
    }
}

void SSSP::synchronize_ghost_distances() {
    for (int32_t v : graph_.get_ghost_vertices()) {
        GhostUpdateMessage msg;
        msg.vertex_id = v;
        msg.distance = distances_[v];
        msg.parent = parents_[v];
        msg.is_deletion = false;
        msg.edge_weight = 0.0;
        ghost_buffers_[graph_.get_vertex_owner(v)].messages.push_back(msg);
    }
}

double SSSP::get_distance(int32_t vertex) const {
    return distances_[vertex];
}

int32_t SSSP::get_parent(int32_t vertex) const {
    return parents_[vertex];
}

int32_t SSSP::get_num_affected_vertices() const {
    return affected_vertices_.size();
}

void SSSP::gather_results(MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0) {
        // Gather distances and parents from all processes
        std::vector<double> all_distances;
        std::vector<int32_t> all_parents;
        MPI_Gather(distances_.data(), distances_.size(), MPI_DOUBLE,
                  all_distances.data(), distances_.size(), MPI_DOUBLE,
                  0, comm);
        MPI_Gather(parents_.data(), parents_.size(), MPI_INT32_T,
                  all_parents.data(), parents_.size(), MPI_INT32_T,
                  0, comm);
    } else {
        // Send local distances and parents to root
        MPI_Gather(distances_.data(), distances_.size(), MPI_DOUBLE,
                  nullptr, 0, MPI_DOUBLE, 0, comm);
        MPI_Gather(parents_.data(), parents_.size(), MPI_INT32_T,
                  nullptr, 0, MPI_INT32_T, 0, comm);
    }
}

void SSSP::broadcast_source(int32_t source, MPI_Comm comm) {
    MPI_Bcast(&source_, 1, MPI_INT32_T, 0, comm);
    if (has_source_) {
        distances_[source_] = 0.0;
    }
}

bool SSSP::is_ghost_vertex(int32_t global_vertex) const {
    return graph_.is_ghost_vertex(global_vertex);
}

bool SSSP::is_local_vertex(int32_t global_vertex) const {
    return !is_ghost_vertex(global_vertex);
}

int32_t SSSP::get_vertex_owner(int32_t global_vertex) const {
    return graph_.get_vertex_owner(global_vertex);
}

int32_t SSSP::global_to_local(int32_t global_vertex) const {
    return graph_.global_to_local(global_vertex);
}

int32_t SSSP::local_to_global(int32_t local_vertex) const {
    return graph_.local_to_global(local_vertex);
}

// Export graph and SSSP tree to JSON
void SSSP::export_to_json(const std::string& filename) const {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Only rank 0 creates the JSON file
    if (rank == 0) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return;
        }
        
        int num_vertices = graph_.get_num_vertices();
        
        file << "{\n";
        file << "  \"num_vertices\": " << num_vertices << ",\n";
        file << "  \"source\": " << source_ << ",\n";
        
        // Export nodes with their distances
        file << "  \"nodes\": [\n";
        for (int32_t v = 0; v < num_vertices; ++v) {
            file << "    {\n";
            file << "      \"id\": " << v << ",\n";
            
            // Convert infinity to -1 for JSON
            double dist = get_distance(v);
            if (dist == std::numeric_limits<double>::infinity()) {
                file << "      \"distance\": -1,\n";
            } else {
                file << "      \"distance\": " << dist << ",\n";
            }
            
            file << "      \"parent\": " << get_parent(v) << "\n";
            file << "    }";
            if (v < num_vertices - 1) {
                file << ",";
            }
            file << "\n";
        }
        file << "  ],\n";
        
        // Export edges
        file << "  \"edges\": [\n";
        bool first_edge = true;
        for (int32_t v = 0; v < num_vertices; ++v) {
            const auto& neighbors = graph_.get_neighbors(v);
            for (const auto& [neighbor, weight] : neighbors) {
                if (!first_edge) {
                    file << ",\n";
                }
                file << "    {\n";
                file << "      \"source\": " << v << ",\n";
                file << "      \"target\": " << neighbor << ",\n";
                file << "      \"weight\": " << weight << ",\n";
                
                // Determine if this edge is part of the SSSP tree
                bool is_tree_edge = (get_parent(neighbor) == v || get_parent(v) == neighbor);
                file << "      \"is_tree_edge\": " << (is_tree_edge ? "true" : "false") << "\n";
                file << "    }";
                first_edge = false;
            }
        }
        file << "\n  ]\n";
        file << "}\n";
        
        file.close();
        std::cout << "Graph and SSSP tree exported to " << filename << std::endl;
    }
}

// Export edge changes to JSON
void SSSP::export_changes_to_json(const std::vector<EdgeChange>& changes, const std::string& filename) const {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Only rank 0 creates the JSON file
    if (rank == 0) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return;
        }
        
        file << "{\n";
        file << "  \"changes\": [\n";
        for (size_t i = 0; i < changes.size(); ++i) {
            const auto& change = changes[i];
            file << "    {\n";
            file << "      \"source\": " << change.u << ",\n";
            file << "      \"target\": " << change.v << ",\n";
            file << "      \"weight\": " << change.weight << ",\n";
            file << "      \"operation\": \"" << (change.is_deletion ? "delete" : "insert") << "\"\n";
            file << "    }";
            if (i < changes.size() - 1) {
                file << ",";
            }
            file << "\n";
        }
        file << "  ]\n";
        file << "}\n";
        
        file.close();
        std::cout << "Edge changes exported to " << filename << std::endl;
    }
} 