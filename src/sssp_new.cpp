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
#include <iostream>
#include <fstream>
#include <sstream>

SSSP::SSSP(const Graph& graph)
    : graph_(graph),
      distances_(graph.get_num_vertices(), std::numeric_limits<double>::infinity()),
      parents_(graph.get_num_vertices(), -1),
      affected_(graph.get_num_vertices(), false),
      affected_del_(graph.get_num_vertices(), false),
      source_(-1),
      has_source_(false) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    ghost_buffers_.resize(world_size);
}

void SSSP::initialize(int32_t source_vertex, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    if (source_vertex < 0 || source_vertex >= graph_.get_num_vertices()) {
        throw std::invalid_argument("Invalid source vertex");
    }

    // Reset all data structures
    std::fill(distances_.begin(), distances_.end(), std::numeric_limits<double>::infinity());
    std::fill(parents_.begin(), parents_.end(), -1);
    std::fill(affected_.begin(), affected_.end(), false);
    std::fill(affected_del_.begin(), affected_del_.end(), false);

    // Set source vertex
    source_ = source_vertex;
    has_source_ = true;

    // Initialize source vertex on its owner process
    int owner_rank = graph_.get_vertex_owner(source_vertex);
    if (owner_rank == rank) {
        int32_t local_source = graph_.global_to_local(source_vertex);
        if (local_source >= 0) {  // Check if source is local
            distances_[local_source] = 0.0;
            std::cout << "Rank " << rank << ": Initialized source vertex " << source_vertex 
                      << " (local index " << local_source << ") with distance 0.0" << std::endl;
        } else {
            std::cerr << "Rank " << rank << ": ERROR - Cannot map source vertex " << source_vertex 
                      << " to local index even though I am the owner (rank " << rank << ")" << std::endl;
        }
    } else {
        std::cout << "Rank " << rank << ": Source vertex " << source_vertex 
                  << " is owned by process " << owner_rank << std::endl;
    }

    // Broadcast source vertex and synchronize
    MPI_Bcast(&source_, 1, MPI_INT32_T, 0, comm);
    
    // Synchronize initial distances
    synchronize_ghost_distances(comm);
    
    // Ensure all processes have completed initialization
    MPI_Barrier(comm);
}

void SSSP::update_sssp(const std::vector<EdgeChange>& edge_changes, MPI_Comm comm) {
    if (!has_source_) {
        throw std::runtime_error("SSSP not initialized with source vertex");
    }

    // Step 1: Process edge changes and identify affected vertices
    process_edge_changes(edge_changes);

    // Step 2: Update affected vertices iteratively
    update_affected_vertices(comm);

    // Step 3: Synchronize ghost distances
    synchronize_ghost_distances(comm);

    // Step 4: Gather results from all processes
    gather_results(comm);
}

void SSSP::process_edge_changes(const std::vector<EdgeChange>& changes) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    std::cout << "Rank " << rank << ": Processing " << changes.size() << " edge changes" << std::endl;
    
    // Process all local vertex updates first, then handle ghost vertices
    std::vector<GhostUpdateMessage> ghost_updates;
    
    // First pass - process local changes and collect ghost updates
    #pragma omp parallel for schedule(dynamic, 1) 
    for (size_t i = 0; i < changes.size(); ++i) {
        const auto& change = changes[i];
        int32_t u = change.u;
        int32_t v = change.v;
        double weight = change.weight;

        // Determine vertex locality
        bool u_is_local = false;
        bool v_is_local = false;
        int32_t local_u = -1;
        int32_t local_v = -1;
        int u_owner = graph_.get_vertex_owner(u);
        int v_owner = graph_.get_vertex_owner(v);
        
        if (u_owner == rank) {
            local_u = graph_.global_to_local(u);
            if (local_u >= 0) u_is_local = true;
        }
        
        if (v_owner == rank) {
            local_v = graph_.global_to_local(v);
            if (local_v >= 0) v_is_local = true;
        }
        
        std::cout << "Rank " << rank << ": Processing edge (" << u << ", " << v << "), weight: " << weight 
                  << ", deletion: " << (change.is_deletion ? "true" : "false") 
                  << ", u_local: " << (u_is_local ? "true" : "false") 
                  << ", v_local: " << (v_is_local ? "true" : "false") << std::endl;
                  
        if (change.is_deletion) {
            // Edge deletion - if the edge is part of SSSP tree, mark affected
            if (u_is_local && local_u >= 0) {
                if (parents_[local_u] == v) {
                    // u's parent is v, so u (and its subtree) are affected by deletion
                    #pragma omp critical
                    {
                        affected_del_[local_u] = true;
                        affected_[local_u] = true;
                    }
                    std::cout << "Rank " << rank << ": Vertex " << u << " affected by deletion - parent is " << v << std::endl;
                }
            }
            
            if (v_is_local && local_v >= 0) {
                if (parents_[local_v] == u) {
                    // v's parent is u, so v (and its subtree) are affected by deletion
                    #pragma omp critical
                    {
                        affected_del_[local_v] = true;
                        affected_[local_v] = true;
                    }
                    std::cout << "Rank " << rank << ": Vertex " << v << " affected by deletion - parent is " << u << std::endl;
                }
            }
        } else {
            // Edge insertion - check if it provides a shorter path
            if (u_is_local && local_u >= 0) {
                double u_dist = distances_[local_u];
                if (u_dist != std::numeric_limits<double>::infinity()) {
                    // u has a valid distance
                    if (v_is_local && local_v >= 0) {
                        // Both endpoints are local
                        double v_dist = distances_[local_v];
                        if (u_dist + weight < v_dist) {
                            // Found shorter path to v
                            #pragma omp critical
                            {
                                distances_[local_v] = u_dist + weight;
                                parents_[local_v] = u;
                                affected_[local_v] = true;
                            }
                            std::cout << "Rank " << rank << ": Updated vertex " << v << " with distance " 
                                      << u_dist + weight << " from " << u << std::endl;
                        }
                    } else {
                        // v is remote - send update message
                        #pragma omp critical
                        {
                            GhostUpdateMessage msg;
                            msg.vertex_id = v;
                            msg.distance = u_dist + weight;
                            msg.parent = u;
                            msg.is_deletion = false;
                            msg.edge_weight = weight;
                            ghost_updates.push_back(msg);
                        }
                        std::cout << "Rank " << rank << ": Queued update for remote vertex " << v 
                                  << " with distance " << u_dist + weight << " from " << u << std::endl;
                    }
                }
            }
            
            if (v_is_local && local_v >= 0) {
                double v_dist = distances_[local_v];
                if (v_dist != std::numeric_limits<double>::infinity()) {
                    // v has a valid distance
                    if (u_is_local && local_u >= 0) {
                        // Both endpoints are local (already handled above)
                        double u_dist = distances_[local_u]; 
                        if (v_dist + weight < u_dist) {
                            // Found shorter path to u
                            #pragma omp critical
                            {
                                distances_[local_u] = v_dist + weight;
                                parents_[local_u] = v;
                                affected_[local_u] = true;
                            }
                            std::cout << "Rank " << rank << ": Updated vertex " << u << " with distance " 
                                      << v_dist + weight << " from " << v << std::endl;
                        }
                    } else {
                        // u is remote - send update message
                        #pragma omp critical
                        {
                            GhostUpdateMessage msg;
                            msg.vertex_id = u;
                            msg.distance = v_dist + weight;
                            msg.parent = v;
                            msg.is_deletion = false;
                            msg.edge_weight = weight;
                            ghost_updates.push_back(msg);
                        }
                        std::cout << "Rank " << rank << ": Queued update for remote vertex " << u 
                                  << " with distance " << v_dist + weight << " from " << v << std::endl;
                    }
                }
            }
        }
    }
    
    // Process ghost vertex updates
    for (const auto& msg : ghost_updates) {
        int remote_owner = graph_.get_vertex_owner(msg.vertex_id);
        if (remote_owner >= 0 && remote_owner < ghost_buffers_.size()) {
            ghost_buffers_[remote_owner].messages.push_back(msg);
        }
    }
    
    std::cout << "Rank " << rank << ": Finished processing edge changes" << std::endl;
}

void SSSP::update_affected_vertices(MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    
    bool global_changes = true;
    int iteration = 0;
    
    while (global_changes) {
        iteration++;
        
        // Part A: Propagate deletion effects
        bool deletion_changes = true;
        
        while (deletion_changes) {
            deletion_changes = false;
            propagate_deletion_effects(deletion_changes);
            
            // Communicate with other processes to determine global changes
            int local_changes = deletion_changes ? 1 : 0;
            int global_del_changes = 0;
            MPI_Allreduce(&local_changes, &global_del_changes, 1, MPI_INT, MPI_SUM, comm);
            deletion_changes = global_del_changes > 0;
        }

        // Part B: Update distances iteratively
        bool relaxation_changes = true;
        
        while (relaxation_changes) {
            relaxation_changes = false;
            propagate_updates(relaxation_changes);
            
            synchronize_ghost_distances(comm);
            
            // Communicate with other processes to determine global changes
            int local_changes = relaxation_changes ? 1 : 0;
            int global_relax_changes = 0;
            MPI_Allreduce(&local_changes, &global_relax_changes, 1, MPI_INT, MPI_SUM, comm);
            relaxation_changes = global_relax_changes > 0;
        }

        // Check if any changes occurred in this iteration
        global_changes = deletion_changes || relaxation_changes;
    }
}

void SSSP::propagate_deletion_effects(bool& changes) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    std::cout << "Rank " << rank << ": Propagating deletion effects" << std::endl;

    #pragma omp parallel for schedule(dynamic, 1) reduction(|:changes)
    for (int32_t i = 0; i < graph_.get_num_vertices(); ++i) {
        if (affected_del_[i]) {
            affected_del_[i] = false;  // Process once per iteration
            std::cout << "Rank " << rank << ": Processing deletion-affected vertex with local index " << i << std::endl;
            
            try {
                int32_t global_v = graph_.local_to_global(i);
                std::cout << "Rank " << rank << ": Local vertex " << i << " maps to global vertex " << global_v << std::endl;
                
                // Reset distance and parent
                distances_[i] = std::numeric_limits<double>::infinity();
                parents_[i] = -1;
                std::cout << "Rank " << rank << ": Reset distance and parent for vertex " << i << std::endl;

                // Propagate to children in SSSP tree
                const auto& neighbors = graph_.get_neighbors(global_v);
                std::cout << "Rank " << rank << ": Global vertex " << global_v << " has " << neighbors.size() << " neighbors" << std::endl;
                
                for (const auto& [neighbor, weight] : neighbors) {
                    std::cout << "Rank " << rank << ": Checking neighbor " << neighbor << " of vertex " << global_v << std::endl;
                    if (graph_.get_vertex_owner(neighbor) == rank) {
                        std::cout << "Rank " << rank << ": Neighbor " << neighbor << " is local" << std::endl;
                        int32_t local_neighbor = graph_.global_to_local(neighbor);
                        if (local_neighbor >= 0 && parents_[local_neighbor] == global_v) {
                            affected_del_[local_neighbor] = true;
                            affected_[local_neighbor] = true;
                            changes = true;
                            std::cout << "Rank " << rank << ": Marked local neighbor " << local_neighbor 
                                      << " as affected by deletion" << std::endl;
                        }
                    } else {
                        // Send message to neighbor's owner
                        int owner = graph_.get_vertex_owner(neighbor);
                        std::cout << "Rank " << rank << ": Neighbor " << neighbor << " is owned by process " 
                                  << owner << std::endl;
                        GhostUpdateMessage msg;
                        msg.vertex_id = neighbor;
                        msg.is_deletion = true;
                        ghost_buffers_[owner].messages.push_back(msg);
                        std::cout << "Rank " << rank << ": Added deletion message for vertex " << neighbor 
                                  << " to buffer for process " << owner << std::endl;
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Rank " << rank << ": Exception in propagate_deletion_effects: " << e.what() << std::endl;
            }
        }
    }
    
    std::cout << "Rank " << rank << ": Finished propagating deletion effects, changes: " 
              << (changes ? "true" : "false") << std::endl;
}

void SSSP::propagate_updates(bool& changes) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Create a copy of affected flags for this iteration
    std::vector<bool> current_affected = affected_;

    #pragma omp parallel for schedule(dynamic, 1) reduction(|:changes)
    for (int32_t i = 0; i < graph_.get_num_vertices(); ++i) {
        if (current_affected[i]) {
            affected_[i] = false;  // Process once per iteration
            int32_t global_v = graph_.local_to_global(i);
            double v_dist = distances_[i];

            for (const auto& [neighbor, weight] : graph_.get_neighbors(global_v)) {
                if (graph_.get_vertex_owner(neighbor) == rank) {
                    int32_t local_neighbor = graph_.global_to_local(neighbor);
                    if (local_neighbor >= 0) {
                        double new_dist = v_dist + weight;
                        if (new_dist < distances_[local_neighbor]) {
                            distances_[local_neighbor] = new_dist;
                            parents_[local_neighbor] = global_v;
                            affected_[local_neighbor] = true;
                            changes = true;
                        }
                    }
                } else {
                    // Send update proposal to neighbor's owner
                    GhostUpdateMessage msg;
                    msg.vertex_id = neighbor;
                    msg.distance = v_dist + weight;
                    msg.parent = global_v;
                    msg.is_deletion = false;
                    msg.edge_weight = weight;
                    ghost_buffers_[graph_.get_vertex_owner(neighbor)].messages.push_back(msg);
                }
            }
        }
    }
}

void SSSP::synchronize_ghost_distances(MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    std::cout << "Rank " << rank << ": Synchronizing ghost distances" << std::endl;
    
    // Prepare messages for each process
    std::vector<std::vector<GhostUpdateMessage>> messages_to_send(size);
    
    // For each local vertex that might be a ghost in other processes
    // Send its current distance and parent
    for (int32_t local_idx = 0; local_idx < graph_.get_num_vertices(); local_idx++) {
        int32_t global_idx = graph_.local_to_global(local_idx);
        if (global_idx < 0) continue; // Skip invalid mappings
        
        for (int r = 0; r < size; r++) {
            if (r == rank) continue;
            
            GhostUpdateMessage msg;
            msg.vertex_id = global_idx;
            msg.distance = distances_[local_idx];
            msg.parent = parents_[local_idx];
            msg.is_deletion = false;
            
            messages_to_send[r].push_back(msg);
        }
    }
    
    // Send/Receive distance updates to/from each process
    for (int r = 0; r < size; r++) {
        if (r == rank) continue;
        
        // Number of messages to send to rank r
        int send_count = messages_to_send[r].size();
        
        // Exchange counts
        int recv_count = 0;
        MPI_Sendrecv(&send_count, 1, MPI_INT, r, 0,
                    &recv_count, 1, MPI_INT, r, 0,
                    comm, MPI_STATUS_IGNORE);
        
        if (send_count > 0) {
            // Send messages
            MPI_Send(messages_to_send[r].data(), 
                   send_count * sizeof(GhostUpdateMessage),
                   MPI_BYTE, r, 1, comm);
        }
        
        if (recv_count > 0) {
            // Receive messages
            std::vector<GhostUpdateMessage> received_messages(recv_count);
            MPI_Recv(received_messages.data(), 
                    recv_count * sizeof(GhostUpdateMessage),
                    MPI_BYTE, r, 1, comm, MPI_STATUS_IGNORE);
            
            // Process received messages
            for (const auto& msg : received_messages) {
                int32_t local_v = graph_.global_to_local(msg.vertex_id);
                if (local_v >= 0) {
                    if (msg.is_deletion) {
                        // Handle deletion
                        affected_del_[local_v] = true;
                        affected_[local_v] = true;
                        std::cout << "Rank " << rank << ": Received deletion message for vertex " 
                                  << msg.vertex_id << " (local " << local_v << ")" << std::endl;
                    } else if (msg.distance < distances_[local_v]) {
                        // Update distance if better
                        distances_[local_v] = msg.distance;
                        parents_[local_v] = msg.parent;
                        affected_[local_v] = true;
                        std::cout << "Rank " << rank << ": Updated vertex " << msg.vertex_id 
                                  << " (local " << local_v << ") with new distance " 
                                  << msg.distance << " from process " << r << std::endl;
                    }
                }
            }
        }
    }
    
    // Barrier to ensure all updates are processed
    MPI_Barrier(comm);
    
    std::cout << "Rank " << rank << ": Finished synchronizing ghost distances" << std::endl;
}

void SSSP::gather_results(MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        // Gather distances and parents from all processes
        std::vector<double> all_distances(graph_.get_num_vertices());
        std::vector<int32_t> all_parents(graph_.get_num_vertices());

        // Copy local data
        for (int32_t i = 0; i < graph_.get_num_vertices(); ++i) {
            if (graph_.get_vertex_owner(i) == 0) {
                int32_t local_i = graph_.global_to_local(i);
                if (local_i >= 0) {
                    all_distances[i] = distances_[local_i];
                    all_parents[i] = parents_[local_i];
                }
            }
        }

        // Receive data from other processes
        for (int r = 1; r < size; ++r) {
            MPI_Status status;
            MPI_Recv(all_distances.data(), all_distances.size(), MPI_DOUBLE,
                    r, 0, comm, &status);
            MPI_Recv(all_parents.data(), all_parents.size(), MPI_INT32_T,
                    r, 1, comm, &status);
        }

        // Update local data
        distances_ = std::move(all_distances);
        parents_ = std::move(all_parents);
    } else {
        // Send local data to root
        MPI_Send(distances_.data(), distances_.size(), MPI_DOUBLE, 0, 0, comm);
        MPI_Send(parents_.data(), parents_.size(), MPI_INT32_T, 0, 1, comm);
    }
}

void SSSP::broadcast_source(int32_t source, MPI_Comm comm) {
    MPI_Bcast(&source_, 1, MPI_INT32_T, 0, comm);
    if (has_source_) {
        int rank;
        MPI_Comm_rank(comm, &rank);
        if (graph_.get_vertex_owner(source_) == rank) {
            int32_t local_source = graph_.global_to_local(source_);
            if (local_source >= 0) {
                distances_[local_source] = 0.0;
            }
        }
    }
}

double SSSP::get_distance(int32_t vertex) const {
    int32_t local_vertex = graph_.global_to_local(vertex);
    return local_vertex >= 0 ? distances_[local_vertex] : std::numeric_limits<double>::infinity();
}

int32_t SSSP::get_parent(int32_t vertex) const {
    int32_t local_vertex = graph_.global_to_local(vertex);
    return local_vertex >= 0 ? parents_[local_vertex] : -1;
}

int32_t SSSP::get_num_affected_vertices() const {
    return std::count(affected_.begin(), affected_.end(), true);
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