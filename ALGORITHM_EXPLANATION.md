# SSSP Update Algorithm Explanation

## 1. Problem Definition

The Single-Source Shortest Path (SSSP) update problem is defined as follows:

**Given:**
- A directed weighted graph G = (V, E, w), where V is the set of vertices, E is the set of edges, and w is a weight function w: E → R⁺
- A source vertex s ∈ V
- An existing SSSP solution (distances and parents) from s to all other vertices in G
- A set of edge changes ΔE consisting of:
  - Edge insertions (u, v, w) where (u, v) ∉ E
  - Edge weight decreases (u, v, w_new) where (u, v) ∈ E and w_new < w(u, v)
  - Edge deletions (u, v) where (u, v) ∈ E
  - Edge weight increases (u, v, w_new) where (u, v) ∈ E and w_new > w(u, v)

**Find:**
- Updated shortest path distances and parent pointers from s to all vertices in G after applying the changes in ΔE

The goal is to compute these updates more efficiently than recomputing the entire SSSP solution from scratch.

## 2. Key Insights

The efficiency of the SSSP update algorithm relies on several key insights:

1. **Locality of Effects**: Edge changes typically affect only a small portion of the graph
2. **Affected Vertex Identification**: We can precisely identify which vertices might be affected by each type of change
3. **Incremental Processing**: We can process changes by only updating affected vertices and their descendants
4. **Separation of Concerns**: Edge deletions/increases and edge insertions/decreases can be handled separately

## 3. Algorithm Details

### 3.1 Core Algorithm Structure

The algorithm follows a two-phase approach:

```
Algorithm: SSSP_Update(G, s, ΔE)
1. ProcessEdgeChanges(G, s, ΔE)  // Identify affected vertices
2. UpdateAffectedVertices(G, s)  // Update distances and parents
```

### 3.2 Phase 1: Identify Affected Vertices

The first phase identifies vertices directly affected by edge changes:

```
Algorithm: ProcessEdgeChanges(G, s, ΔE)
1. for each change (u, v, w) in ΔE:
2.     if change is a deletion or weight increase:
3.         if edge (u, v) is in the current SSSP tree:
4.             Mark v and its subtree as affected by deletion
5.     else: // change is an insertion or weight decrease
6.         if d[u] + w < d[v]:
7.             d[v] = d[u] + w
8.             parent[v] = u
9.             Mark v as affected
10.        else if d[v] + w < d[u]:
11.            d[u] = d[v] + w
12.            parent[u] = v
13.            Mark u as affected
```

### 3.3 Phase 2: Update Affected Vertices

The second phase updates all vertices affected by the changes:

```
Algorithm: UpdateAffectedVertices(G, s)
1. // Part A: Propagate deletion effects
2. while there are vertices affected by deletion:
3.     for each vertex v affected by deletion:
4.         Mark v as no longer affected by deletion
5.         Set d[v] = infinity
6.         Set parent[v] = null
7.         Mark all children of v in the SSSP tree as affected by deletion
8.
9. // Part B: Update distances
10. while there are affected vertices:
11.     current_affected = affected
12.     Clear affected set
13.     for each vertex v in current_affected:
14.         for each outgoing edge (v, u) with weight w:
15.             if d[v] + w < d[u]:
16.                 d[u] = d[v] + w
17.                 parent[u] = v
18.                 Mark u as affected
19.         for each incoming edge (u, v) with weight w:
20.             if d[u] + w < d[v]:
21.                 d[v] = d[u] + w
22.                 parent[v] = u
23.                 Mark v as affected
```

## 4. Parallel Algorithm

The algorithm can be parallelized across multiple processes using the following approach:

### 4.1 Graph Partitioning

The graph is partitioned into P parts (one per process) using a graph partitioning algorithm like METIS:

```
Algorithm: PartitionGraph(G, P)
1. Partition G into P subgraphs G₁, G₂, ..., Gₚ
2. Assign each subgraph to one process
3. Identify ghost vertices at partition boundaries
4. Set up communication patterns between processes
```

### 4.2 Parallel Process Edge Changes

Each process processes changes affecting its local vertices:

```
Algorithm: ParallelProcessChanges(Gᵢ, ΔE, comm)
1. Filter ΔE to extract locally relevant changes ΔEᵢ
2. #pragma omp parallel for
3. for each change (u, v, w) in ΔEᵢ:
4.     Process change locally (similar to sequential algorithm)
5.     If change affects ghost vertices, queue updates for communication
6. Exchange ghost vertex updates with other processes
```

### 4.3 Parallel Update Affected Vertices

The update process is also performed in parallel:

```
Algorithm: ParallelUpdateAffected(Gᵢ, comm)
1. // Part A: Parallel propagate deletion effects
2. global_changes = true
3. while global_changes:
4.     local_changes = false
5.     #pragma omp parallel for reduction(||:local_changes)
6.     for each local vertex v affected by deletion:
7.         Process deletion effects (similar to sequential algorithm)
8.         If effects propagate to ghost vertices, queue updates
9.     Exchange ghost vertex updates with other processes
10.    Perform MPI_Allreduce to determine if any process has changes
11.
12. // Part B: Parallel update distances
13. global_changes = true
14. while global_changes:
15.    local_changes = false
16.    Synchronize ghost vertex distances with other processes
17.    #pragma omp parallel for reduction(||:local_changes)
18.    for each local affected vertex v:
19.        Process edge relaxations (similar to sequential algorithm)
20.        If updates affect ghost vertices, queue updates
21.    Exchange ghost vertex updates with other processes
22.    Perform MPI_Allreduce to determine if any process has changes
```

## 5. Theoretical Analysis

### 5.1 Time Complexity

The time complexity of the algorithm depends on several factors:

- |V| = number of vertices
- |E| = number of edges
- |ΔE| = number of edge changes
- |A| = number of affected vertices
- d = maximum degree of a vertex
- P = number of processes

**Sequential Algorithm:**
- Worst case (all vertices affected): O(|V| + |E|)
- Typical case (few vertices affected): O(|ΔE| + |A|·d)

**Parallel Algorithm:**
- Ideal speedup: O((|ΔE| + |A|·d)/P)
- With communication overhead: O((|ΔE| + |A|·d)/P + C·log P)
  where C is the communication cost per iteration

### 5.2 Space Complexity

The space complexity includes:

- O(|V| + |E|) for graph storage
- O(|V|) for distances and parents
- O(|V|) for affected sets
- O(|Vg|) for ghost vertex information, where |Vg| is the number of ghost vertices

### 5.3 Correctness

The algorithm maintains the following invariants:

1. **Distance Invariant**: At all times, d[v] ≤ d[u] + w(u,v) for every edge (u,v)
2. **Parent Invariant**: parent[v] always represents the first hop of the shortest path from s to v
3. **Connectivity Invariant**: If a vertex v is reachable from s, it will have d[v] < ∞

## 6. Algorithm Optimizations

Several optimizations enhance the performance of the algorithm:

### 6.1 Priority-Based Processing

Vertices can be processed in order of increasing distance to minimize redundant work:

```
Algorithm: PriorityUpdateAffected(G, s)
1. Initialize a priority queue Q with affected vertices
2. while Q is not empty:
3.     Extract vertex v with minimum d[v] from Q
4.     for each neighbor u of v:
5.         if d[v] + w(v,u) < d[u]:
6.             d[u] = d[v] + w(v,u)
7.             parent[u] = v
8.             Insert or update u in Q
```

### 6.2 Early Termination

The algorithm can terminate early when no further changes occur:

```
Algorithm: EarlyTerminateUpdateAffected(G, s)
1. while there are affected vertices:
2.     if no distances changed in the last iteration:
3.         break
4.     // Process affected vertices as before
```

### 6.3 Delta-Stepping Approach

The Delta-Stepping approach groups vertices by distance ranges for more efficient processing:

```
Algorithm: DeltaSteppingUpdate(G, s, Δ)
1. Divide vertices into buckets B₀, B₁, B₂, ... where
   Bᵢ contains vertices v with i·Δ ≤ d[v] < (i+1)·Δ
2. i = 0
3. while there are non-empty buckets:
4.     while Bᵢ is not empty:
5.         Process all vertices in Bᵢ in parallel
6.         Update buckets based on new distances
7.     i = i + 1
```

## 7. Comparison with Other Approaches

### 7.1 From-Scratch SSSP Algorithms

| Algorithm | Time Complexity | Advantages | Disadvantages |
|-----------|-----------------|------------|--------------|
| Dijkstra  | O(|V|log|V| + |E|) | Optimal for static graphs | Doesn't leverage existing solution |
| Bellman-Ford | O(|V|·|E|) | Handles negative edges | Much slower than incremental |
| SSSP Update | O(|ΔE| + |A|·d) | Efficient for small changes | More complex implementation |

### 7.2 Other Dynamic SSSP Algorithms

| Algorithm | Key Approach | Strengths | Weaknesses |
|-----------|--------------|-----------|------------|
| RR Algorithm | Edge-centric updates | Simple implementation | Less parallelizable |
| FMN Algorithm | Affected subgraph identification | Theoretically optimal | Complex data structures |
| Our Algorithm | Vertex-centric & two-phase | Highly parallelizable | Requires careful synchronization |

## 8. Conclusion

The SSSP update algorithm provides an efficient way to maintain shortest paths in dynamic graphs without recomputing from scratch. Its parallel implementation using MPI and OpenMP enables scaling to large graphs across multiple compute nodes. The algorithm's efficiency stems from its ability to identify and update only those vertices affected by edge changes, making it particularly effective for scenarios where changes are localized and affect a small portion of the graph. 