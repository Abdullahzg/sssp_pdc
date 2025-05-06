#pragma once

#include "graph.hpp"
#include <cstdint>
#include <vector>
#include <metis.h>

class GraphPartitioner {
public:
    explicit GraphPartitioner(Graph& graph);
    ~GraphPartitioner() = default;

    // Partition the graph into num_partitions parts
    void partition(int num_partitions);

private:
    Graph& graph_;

    // Convert graph to METIS format
    void convert_to_metis_format(std::vector<int32_t>& xadj,
                               std::vector<int32_t>& adjncy,
                               std::vector<int32_t>& adjwgt);
}; 