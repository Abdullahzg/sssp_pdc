#pragma once

#include <vector>
#include <cstdint>
#include <mpi.h>

// Edge change structure
struct EdgeChange {
    int32_t u;
    int32_t v;
    double weight;
    bool is_deletion;
};

// Message for ghost vertex updates
struct GhostUpdateMessage {
    int32_t vertex_id;
    double distance;
    int32_t parent;
    bool is_deletion;
    double edge_weight;
};

// Buffer for non-blocking MPI communication
struct MessageBuffer {
    std::vector<GhostUpdateMessage> messages;
    std::vector<MPI_Request> requests;
}; 