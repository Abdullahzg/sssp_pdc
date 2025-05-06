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
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (graph_.get_vertex_owner(source_) == rank) {
        int32_t local_source = graph_.global_to_local(source_);
        if (local_source >= 0) {  // Check if source is local
            distances_[local_source] = 0.0;
        }
    }

    // Broadcast source vertex to all processes
    broadcast_source(source_, comm);
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

    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < changes.size(); ++i) {
        const auto& change = changes[i];
        int32_t u = change.u;
        int32_t v = change.v;
        double weight = change.weight;

        // Check if either vertex is local
        bool u_local = graph_.get_vertex_owner(u) == rank;
        bool v_local = graph_.get_vertex_owner(v) == rank;
        
        std::cout << "Rank " << rank << ": Processing edge (" << u << ", " << v << "), weight: " << weight 
                  << ", deletion: " << (change.is_deletion ? "true" : "false") 
                  << ", u_local: " << (u_local ? "true" : "false") 
                  << ", v_local: " << (v_local ? "true" : "false") << std::endl;

        try {
            if (change.is_deletion) {
                if (u_local) {
                    int32_t local_u = graph_.global_to_local(u);
                    std::cout << "Rank " << rank << ": Global vertex " << u << " maps to local vertex " << local_u << std::endl;
                    if (local_u >= 0 && parents_[local_u] == v) {
                        affected_del_[local_u] = true;
                        affected_[local_u] = true;
                        std::cout << "Rank " << rank << ": Marking local vertex " << local_u << " (global " << u 
                                  << ") as affected by deletion" << std::endl;
                    }
                }
                if (v_local) {
                    int32_t local_v = graph_.global_to_local(v);
                    std::cout << "Rank " << rank << ": Global vertex " << v << " maps to local vertex " << local_v << std::endl;
                    if (local_v >= 0 && parents_[local_v] == u) {
                        affected_del_[local_v] = true;
                        affected_[local_v] = true;
                        std::cout << "Rank " << rank << ": Marking local vertex " << local_v << " (global " << v 
                                  << ") as affected by deletion" << std::endl;
                    }
                }
            } else {
                if (u_local) {
                    int32_t local_u = graph_.global_to_local(u);
                    std::cout << "Rank " << rank << ": Global vertex " << u << " maps to local vertex " << local_u << std::endl;
                    if (local_u >= 0) {
                        double u_dist = distances_[local_u];
                        
                        // Only proceed if v is also local
                        if (v_local) {
                            int32_t local_v = graph_.global_to_local(v);
                            std::cout << "Rank " << rank << ": Global vertex " << v << " maps to local vertex " << local_v << std::endl;
                            if (local_v >= 0 && u_dist + weight < distances_[local_v]) {
                                distances_[local_v] = u_dist + weight;
                                parents_[local_v] = u;
                                affected_[local_v] = true;
                                std::cout << "Rank " << rank << ": Updated distance for local vertex " << local_v 
                                          << " (global " << v << ") to " << distances_[local_v] << std::endl;
                            }
                        } else {
                            // v is a ghost vertex, send update message
                            std::cout << "Rank " << rank << ": Vertex " << v << " is a ghost vertex, sending update message" << std::endl;
                            GhostUpdateMessage msg;
                            msg.vertex_id = v;
                            msg.distance = u_dist + weight;
                            msg.parent = u;
                            msg.is_deletion = false;
                            msg.edge_weight = weight;
                            int owner = graph_.get_vertex_owner(v);
                            ghost_buffers_[owner].messages.push_back(msg);
                        }
                    }
                }
                if (v_local) {
                    int32_t local_v = graph_.global_to_local(v);
                    std::cout << "Rank " << rank << ": Global vertex " << v << " maps to local vertex " << local_v << std::endl;
                    if (local_v >= 0) {
                        double v_dist = distances_[local_v];
                        
                        // Only proceed if u is also local
                        if (u_local) {
                            int32_t local_u = graph_.global_to_local(u);
                            std::cout << "Rank " << rank << ": Global vertex " << u << " maps to local vertex " << local_u << std::endl;
                            if (local_u >= 0 && v_dist + weight < distances_[local_u]) {
                                distances_[local_u] = v_dist + weight;
                                parents_[local_u] = v;
                                affected_[local_u] = true;
                                std::cout << "Rank " << rank << ": Updated distance for local vertex " << local_u 
                                          << " (global " << u << ") to " << distances_[local_u] << std::endl;
                            }
                        } else {
                            // u is a ghost vertex, send update message
                            std::cout << "Rank " << rank << ": Vertex " << u << " is a ghost vertex, sending update message" << std::endl;
                            GhostUpdateMessage msg;
                            msg.vertex_id = u;
                            msg.distance = v_dist + weight;
                            msg.parent = v;
                            msg.is_deletion = false;
                            msg.edge_weight = weight;
                            int owner = graph_.get_vertex_owner(u);
                            ghost_buffers_[owner].messages.push_back(msg);
                        }
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Rank " << rank << ": Exception in process_edge_changes: " << e.what() << std::endl;
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

    // Use a simpler, more robust implementation with blocking MPI calls
    for (int r = 0; r < size; ++r) {
        if (r == rank) continue;
        
        // Pack messages for this process
        int msg_count = ghost_buffers_[r].messages.size();
        
        // Exchange message counts with the receiving process
        int other_msg_count = 0;
        MPI_Sendrecv(&msg_count, 1, MPI_INT, r, 0,
                    &other_msg_count, 1, MPI_INT, r, 0,
                    comm, MPI_STATUS_IGNORE);
        
        // Send messages if we have any
        if (msg_count > 0) {
            MPI_Send(ghost_buffers_[r].messages.data(), 
                   msg_count * sizeof(GhostUpdateMessage),
                   MPI_BYTE, r, 1, comm);
        }
        
        // Receive messages if the other process has any to send us
        if (other_msg_count > 0) {
            std::vector<GhostUpdateMessage> received_messages(other_msg_count);
            MPI_Recv(received_messages.data(), 
                    other_msg_count * sizeof(GhostUpdateMessage),
                    MPI_BYTE, r, 1, comm, MPI_STATUS_IGNORE);
            
            // Process received messages
            for (const auto& msg : received_messages) {
                if (graph_.get_vertex_owner(msg.vertex_id) == rank) {
                    int32_t local_v = graph_.global_to_local(msg.vertex_id);
                    if (local_v >= 0) {
                        if (msg.is_deletion) {
                            affected_del_[local_v] = true;
                            affected_[local_v] = true;
                        } else if (msg.distance < distances_[local_v]) {
                            distances_[local_v] = msg.distance;
                            parents_[local_v] = msg.parent;
                            affected_[local_v] = true;
                        }
                    }
                }
            }
        }
    }

    // Clear message buffers
    for (auto& buffer : ghost_buffers_) {
        buffer.messages.clear();
    }
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