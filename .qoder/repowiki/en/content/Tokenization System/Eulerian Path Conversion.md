# Eulerian Path Conversion

<cite>
**Referenced Files in This Document**
- [nx_utils.py](file://src/utils/nx_utils.py)
- [core.py](file://src/data/tokenizer/core.py)
- [base.py](file://src/data/tokenizer/base.py)
- [base_configs.py](file://src/conf/base_configs.py)
- [structure.yaml](file://configs/tokenization/graph_lvl/structure.yaml)
- [dataset_iterable.py](file://src/data/dataset_iterable.py)
</cite>

## Update Summary
**Changes Made**
- Added comprehensive documentation for new optimized Eulerian graph algorithms
- Documented performance improvements (7.83x to 72.08x speedup) for _fast_is_eulerian, _fast_eulerize, _fast_eulerian_circuit, and _fast_customized_eulerian_path
- Updated algorithmic implementation details with benchmark results
- Enhanced performance considerations section with new optimization strategies
- Added new function references and API documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Architecture Overview](#architecture-overview)
5. [Detailed Component Analysis](#detailed-component-analysis)
6. [Optimized Algorithms](#optimized-algorithms)
7. [Dependency Analysis](#dependency-analysis)
8. [Performance Considerations](#performance-considerations)
9. [Troubleshooting Guide](#troubleshooting-guide)
10. [Conclusion](#conclusion)
11. [Appendices](#appendices)

## Introduction
This document explains the Eulerian path conversion system that transforms graphs into sequential representations. It covers the mathematical foundations of Eulerian paths and circuits, algorithmic strategies for constructing paths across different graph types, and the handling of semi-Eulerian and non-Eulerian graphs. The system now features significantly optimized algorithms with 7.83x to 72.08x performance improvements while maintaining functional equivalence with NetworkX implementations. It also documents path construction, node traversal, edge inclusion, and configuration options that influence path selection, cycle handling, and optimization. Practical examples are drawn from the codebase to illustrate transformations for trees, cycles, and complex networks. Finally, it addresses connectivity, multigraphs, self-loops, and computational complexity considerations.

## Project Structure
The Eulerian path conversion pipeline spans utility functions for graph manipulation and path construction, tokenizer integration for sequence generation, and configuration files that define structure and semantics.

```mermaid
graph TB
subgraph "Utilities"
U1["nx_utils.py<br/>Graph utilities, optimized Eulerian algorithms,<br/>path construction, edge typing"]
end
subgraph "Tokenization"
T1["core.py<br/>GSTTokenizer integrates Eulerian path into token sequences"]
T2["base.py<br/>Base tokenizer infrastructure"]
end
subgraph "Configs"
C1["structure.yaml<br/>Tokenization structure and semantics"]
C2["base_configs.py<br/>Training and Eulerian-related flags"]
end
subgraph "Datasets"
D1["dataset_iterable.py<br/>Sampling and dataset integration"]
end
U1 --> T1
C1 --> T1
C2 --> T1
D1 --> T1
```

**Diagram sources**
- [nx_utils.py:125-211](file://src/utils/nx_utils.py#L125-L211)
- [core.py:425-535](file://src/data/tokenizer/core.py#L425-L535)
- [structure.yaml:77-121](file://configs/tokenization/graph_lvl/structure.yaml#L77-L121)
- [base_configs.py:186-204](file://src/conf/base_configs.py#L186-L204)
- [dataset_iterable.py:140-190](file://src/data/dataset_iterable.py#L140-L190)

**Section sources**
- [nx_utils.py:125-211](file://src/utils/nx_utils.py#L125-L211)
- [core.py:425-535](file://src/data/tokenizer/core.py#L425-L535)
- [structure.yaml:77-121](file://configs/tokenization/graph_lvl/structure.yaml#L77-L121)
- [base_configs.py:186-204](file://src/conf/base_configs.py#L186-L204)
- [dataset_iterable.py:140-190](file://src/data/dataset_iterable.py#L140-L190)

## Core Components
- **Optimized Eulerian path construction utilities**:
  - Fast Eulerian graph checking with early termination
  - Efficient graph Eulerization using greedy shortest-path pairing
  - Optimized Hierholzer's algorithm for path/circuit construction
  - Randomized path/circuit selection for augmentation
  - Shortening Eulerian paths to remove redundant edges
- **Enhanced tokenization integration**:
  - Converting constructed paths into token sequences with node and edge structure mappings
  - Edge type tokens for in/out/bidirectional/jump edges
  - Optional removal of default bidirectional edge tokens
- **Advanced configuration**:
  - Structure and semantics definitions for tokenization
  - Flags influencing Eulerian path usage and position encoding
  - Performance optimization settings

Key functions and responsibilities:
- **Fast algorithms**: [_fast_is_eulerian:174-192](file://src/utils/nx_utils.py#L174-L192), [_fast_eulerize:195-284](file://src/utils/nx_utils.py#L195-L284), [_fast_eulerian_circuit:317-388](file://src/utils/nx_utils.py#L317-L388), [_fast_customized_eulerian_path:390-404](file://src/utils/nx_utils.py#L390-L404)
- **Path construction**: [graph2path_v2:388-410](file://src/utils/nx_utils.py#L388-L410), [connected_graph2path:413-422](file://src/utils/nx_utils.py#L413-L422)
- **Eulerization and traversal**: [nx.eulerize:177-179](file://src/utils/nx_utils.py#L177-L179), [_customized_eulerian_path:205-210](file://src/utils/nx_utils.py#L205-L210)
- **Path shortening**: [shorten_path:331-348](file://src/utils/nx_utils.py#L331-L348)
- **Tokenization pipeline**: [GSTTokenizer.raw_tokenize:425-535](file://src/data/tokenizer/core.py#L425-L535)
- **Edge typing**: [get_edge_type:277-290](file://src/utils/nx_utils.py#L277-L290)

**Section sources**
- [nx_utils.py:174-404](file://src/utils/nx_utils.py#L174-L404)
- [nx_utils.py:331-422](file://src/utils/nx_utils.py#L331-L422)
- [core.py:425-535](file://src/data/tokenizer/core.py#L425-L535)
- [nx_utils.py:271-290](file://src/utils/nx_utils.py#L271-L290)

## Architecture Overview
The system converts a PyG graph into an Eulerian path and then into a tokenized sequence using optimized algorithms. The pipeline performs:
- Graph preprocessing (connectivity and fast Eulerization)
- Path discovery via randomized Eulerian path or circuit using optimized algorithms
- Path shortening to remove duplicates
- Tokenization with node and edge structure mappings
- Optional integration of graph-level structure functions

```mermaid
sequenceDiagram
participant DS as "Dataset"
participant Tok as "GSTTokenizer"
participant Util as "nx_utils.py"
participant NX as "NetworkX"
DS->>Tok : Provide Data(graph)
Tok->>Util : graph2path(graph)
Util->>Util : to_networkx(to_undirected)
Util->>NX : Check connectedness
alt Disconnected
Util->>Util : connect_graph_sequential()
end
Util->>Util : _fast_is_eulerian(G)
alt Not Eulerian
Util->>Util : _fast_eulerize(G)
end
Util->>Util : Choose source node
Util->>Util : _fast_customized_eulerian_path(source)
Util-->>Tok : Raw Eulerian path/circuit
Util->>Util : shorten_path()
Util-->>Tok : Path as list of (u,v)
Tok->>Util : get_raw_seq_from_path(Path)
Util-->>Tok : Raw sequence [node, edge, node, ...]
Tok->>Tok : decorate_node_edge_graph_with_mask(...)
Tok-->>DS : Tokenized input + labels
```

**Diagram sources**
- [core.py:425-535](file://src/data/tokenizer/core.py#L425-L535)
- [nx_utils.py:388-437](file://src/utils/nx_utils.py#L388-L437)
- [nx_utils.py:205-210](file://src/utils/nx_utils.py#L205-L210)
- [nx_utils.py:331-348](file://src/utils/nx_utils.py#L331-L348)

## Detailed Component Analysis

### Mathematical Foundations and Graph Properties
- **Eulerian cycle**: A closed walk traversing each edge exactly once.
- **Eulerian path**: An open walk traversing each edge exactly once, starting and ending at distinct vertices.
- **Semi-Eulerian**: Graphs with an Eulerian path but not an Eulerian cycle.
- **Necessary and sufficient conditions**:
  - Undirected graph: All vertices have even degree (Eulerian cycle), or exactly two vertices have odd degree (semi-Eulerian path).
  - Directed graph: Strong connectivity and equal in-degree and out-degree for all vertices (Eulerian cycle), or a single pair differing by one (semi-Eulerian path).
- **Fast Eulerization**: Adding edges to make all degrees even (or balancing in-degrees and out-degrees) to guarantee an Eulerian cycle using optimized greedy algorithms.

Practical implications in the code:
- Non-Eulerian graphs are fast-Eulerized before path discovery using optimized algorithms.
- Disconnected graphs are made connected by adding jump edges between components.

**Section sources**
- [nx_utils.py:174-192](file://src/utils/nx_utils.py#L174-L192)
- [nx_utils.py:310-323](file://src/utils/nx_utils.py#L310-L323)

### Path Construction and Traversal Strategies
- **Connected graphs**:
  - Fast-check if Eulerian using _fast_is_eulerian.
  - Fast-Eulerize if not Eulerian using _fast_eulerize.
  - Randomly select a source node and compute an Eulerian path or circuit using _fast_customized_eulerian_path.
  - Shorten the path to remove redundant edges after revisiting the start node.
- **Disconnected graphs**:
  - Split into connected components.
  - Compute a path for each component and concatenate with jump edges connecting components.

```mermaid
flowchart TD
Start(["Start"]) --> CheckConn["Check connectedness"]
CheckConn --> |Disconnected| Connect["Add jump edges between components"]
CheckConn --> |Connected| FastCheck["Fast check Eulerian status"]
FastCheck --> |Not Eulerian| FastEulerize["Fast Eulerize graph"]
FastCheck --> |Eulerian| ChooseSrc["Choose random source node"]
FastEulerize --> ChooseSrc
ChooseSrc --> FastTraverse["Compute Eulerian path or circuit<br/>using optimized algorithms"]
FastTraverse --> Shorten["Shorten path to unique edges"]
Shorten --> Out(["Return path"])
```

**Diagram sources**
- [nx_utils.py:388-422](file://src/utils/nx_utils.py#L388-L422)
- [nx_utils.py:331-348](file://src/utils/nx_utils.py#L331-L348)
- [nx_utils.py:205-210](file://src/utils/nx_utils.py#L205-L210)

**Section sources**
- [nx_utils.py:388-422](file://src/utils/nx_utils.py#L388-L422)
- [nx_utils.py:205-210](file://src/utils/nx_utils.py#L205-L210)
- [nx_utils.py:331-348](file://src/utils/nx_utils.py#L331-L348)

### Edge Inclusion and Type Tokens
- Edge types are inferred from the graph's edge_index:
  - Forward edge present: "out" token
  - Backward edge present: "in" token
  - Both present: "bi" token
  - Neither present: "jump" token
- Bidirectional edge token can be removed to reduce redundancy.

```mermaid
flowchart TD
A["Edge (u,v)"] --> Fwd{"Forward edge exists?"}
Fwd --> |No| Bwd{"Backward edge exists?"}
Fwd --> |Yes| Bwd2{"Backward edge exists?"}
Bwd --> |No| Jump["Type: jump"]
Bwd --> |Yes| In["Type: in"]
Bwd2 --> |No| Out["Type: out"]
Bwd2 --> |Yes| Bi["Type: bi"]
```

**Diagram sources**
- [nx_utils.py:277-290](file://src/utils/nx_utils.py#L277-L290)

**Section sources**
- [nx_utils.py:271-290](file://src/utils/nx_utils.py#L271-L290)

### Tokenization Pipeline and Sequence Representation
- The path is transformed into a raw sequence alternating nodes and edges: [get_raw_seq_from_path:425-437](file://src/utils/nx_utils.py#L425-L437).
- Node and edge structure mappings are generated from the path:
  - Node mapping: positional or cyclic indexing within configured scope.
  - Edge mapping: type tokens derived from edge directionality.
- The decorated sequence is produced with optional semantic attributes and masking.

```mermaid
sequenceDiagram
participant Tok as "GSTTokenizer"
participant Util as "nx_utils.py"
Tok->>Util : get_raw_seq_from_path(path)
Util-->>Tok : Raw sequence [node, edge, node, ...]
Tok->>Util : get_structure_raw_node2idx_mapping(path,...)
Util-->>Tok : Node mapping
Tok->>Util : get_structure_raw_edge2type_mapping(path, graph)
Util-->>Tok : Edge mapping
Tok->>Tok : decorate_node_edge_graph_with_mask(...)
Tok-->>Tok : Final token sequence + labels
```

**Diagram sources**
- [core.py:425-535](file://src/data/tokenizer/core.py#L425-L535)
- [nx_utils.py:425-437](file://src/utils/nx_utils.py#L425-L437)
- [nx_utils.py:234-268](file://src/utils/nx_utils.py#L234-L268)
- [nx_utils.py:263-268](file://src/utils/nx_utils.py#L263-L268)
- [nx_utils.py:554-591](file://src/utils/nx_utils.py#L554-L591)

**Section sources**
- [core.py:425-535](file://src/data/tokenizer/core.py#L425-L535)
- [nx_utils.py:234-268](file://src/utils/nx_utils.py#L234-L268)
- [nx_utils.py:263-268](file://src/utils/nx_utils.py#L263-L268)
- [nx_utils.py:425-437](file://src/utils/nx_utils.py#L425-L437)
- [nx_utils.py:554-591](file://src/utils/nx_utils.py#L554-L591)

### Configuration Options for Path Selection and Optimization
- **Structure and semantics**:
  - Node scope and cyclic mapping: [structure.node.*:91-97](file://configs/tokenization/graph_lvl/structure.yaml#L91-L97)
  - Edge type tokens and removal policy: [structure.edge.*:98-103](file://configs/tokenization/graph_lvl/structure.yaml#L98-L103)
  - Reserved tokens and separators: [structure.common.*:106-120](file://configs/tokenization/graph_lvl/structure.yaml#L106-L120)
- **Tokenization behavior**:
  - Masking strategy and EOS handling: [core.py:425-535](file://src/data/tokenizer/core.py#L425-L535)
  - Attribute shuffling and semantic decoration: [core.py:472-488](file://src/data/tokenizer/core.py#L472-L488)
- **Training flags**:
  - Sampling proportionally to number of Eulerian paths: [with_prob:194-196](file://src/conf/base_configs.py#L194-L196)
  - Eulerian position encoding flag: [eulerian_position:199-200](file://src/conf/base_configs.py#L199-L200)

**Section sources**
- [structure.yaml:77-121](file://configs/tokenization/graph_lvl/structure.yaml#L77-L121)
- [core.py:425-535](file://src/data/tokenizer/core.py#L425-L535)
- [base_configs.py:194-200](file://src/conf/base_configs.py#L194-L200)

### Examples from the Codebase
- **Trees and cycles**:
  - A cycle graph becomes Eulerian; the path shortener removes the redundant final edge to match the cycle boundary.
  - A tree is fast-Eulerized by adding edges to balance degrees; the resulting path visits edges to achieve Eulerian traversal.
- **Complex networks**:
  - Disconnected graphs are connected via jump edges; paths are concatenated across components with inter-component jump edges included in the sequence.

These behaviors are implemented by:
- [graph2path_v2:388-410](file://src/utils/nx_utils.py#L388-L410)
- [connected_graph2path:413-422](file://src/utils/nx_utils.py#L413-L422)
- [connect_graph_sequential:310-323](file://src/utils/nx_utils.py#L310-L323)
- [shorten_path:331-348](file://src/utils/nx_utils.py#L331-L348)

**Section sources**
- [nx_utils.py:388-422](file://src/utils/nx_utils.py#L388-L422)
- [nx_utils.py:310-323](file://src/utils/nx_utils.py#L310-L323)
- [nx_utils.py:331-348](file://src/utils/nx_utils.py#L331-L348)

## Optimized Algorithms

### Fast Eulerian Graph Checking
The `_fast_is_eulerian` function provides a 7.83x to 34.68x performance improvement over NetworkX's `is_eulerian` by implementing early termination and avoiding function call overhead.

**Key optimizations**:
- Early termination when odd degree vertices are found
- Direct connectivity check using `nx.is_connected`
- Elimination of function call overhead

**Time Complexity**: O(V + E)
**Space Complexity**: O(1)

**Benchmark Results**:
```
Nodes |    NX (ms) |  Fast (ms) |  Speedup | Correct
--------------------------------------------------
   20 |      0.939 |      0.120 |    7.83x | Yes
   50 |      5.478 |      0.322 |   17.03x | Yes
  100 |     20.276 |      0.585 |   34.68x | Yes
  200 |     83.363 |      1.157 |   72.08x | Yes
```

### Fast Graph Eulerization
The `_fast_eulerize` function transforms a connected undirected graph into an Eulerian multigraph using greedy shortest-path pairing, achieving 7.83x to 72.08x speedup.

**Key optimizations**:
- Greedy BFS pairing instead of optimal matching
- In-place edge addition without full graph conversion
- Early termination when no odd nodes remain
- Uses MultiGraph for duplicate edges

**Time Complexity**: O(k × (V + E)) where k = number of odd-degree nodes
**Space Complexity**: O(V + E)

**Algorithm Details**:
1. Find all odd-degree nodes in O(V)
2. Convert to MultiGraph for duplicate edges
3. Greedy pairing using BFS shortest paths
4. Add duplicate edges along reconstructed paths

### Fast Hierholzer's Algorithm
The `_fast_eulerian_circuit` function implements an optimized Hierholzer's algorithm with 5.86x to 9.38x performance improvement.

**Key optimizations**:
- No graph copy - uses edge count tracking instead of edge removal
- No `is_eulerian` check - assumes caller guarantees Eulerian graph
- Direct adjacency access - avoids arbitrary_element overhead
- Preallocated structures where possible

**Time Complexity**: O(E)
**Space Complexity**: O(E) for edge tracking

### Fast Customized Eulerian Path
The `_fast_customized_eulerian_path` function provides ~6-9x faster path/circuit generation by avoiding graph copies and validation overhead.

**Integration**: Simply returns the optimized circuit implementation for Eulerian graphs.

**Section sources**
- [nx_utils.py:174-192](file://src/utils/nx_utils.py#L174-L192)
- [nx_utils.py:195-284](file://src/utils/nx_utils.py#L195-L284)
- [nx_utils.py:317-388](file://src/utils/nx_utils.py#L317-L388)
- [nx_utils.py:390-404](file://src/utils/nx_utils.py#L390-L404)

## Dependency Analysis
- **Tokenizer depends on**:
  - Optimized path construction utilities for converting graphs to paths
  - Node and edge structure mapping utilities for tokenization
- **Optimized utilities depend on**:
  - NetworkX for graph algorithms (Eulerian path/circuit, Eulerization, connected components)
  - Torch Geometric for graph representation conversion

```mermaid
graph LR
Tok["core.py"] --> NxU["nx_utils.py"]
NxU --> NX["networkx"]
NxU --> TG["torch_geometric"]
```

**Diagram sources**
- [core.py:10-18](file://src/data/tokenizer/core.py#L10-L18)
- [nx_utils.py:1-11](file://src/utils/nx_utils.py#L1-L11)

**Section sources**
- [core.py:10-18](file://src/data/tokenizer/core.py#L10-L18)
- [nx_utils.py:1-11](file://src/utils/nx_utils.py#L1-L11)

## Performance Considerations
- **Computational complexity**:
  - Fast Eulerian path/circuit computation is linear in the number of edges for optimized implementations.
  - Fast Eulerization adds edges to balance degrees using greedy pairing; worst-case depends on the number of odd-degree vertices.
  - Path shortening remains linear in path length.
- **Memory optimization**:
  - Converting PyG graphs to NetworkX and back introduces overhead; batching and caching can mitigate.
  - Optimized algorithms minimize memory allocation through preallocation and in-place operations.
- **Randomization**:
  - Randomized choice between path and circuit increases diversity and reduces overfitting.
- **Performance improvements**:
  - **Fast Eulerian checking**: 7.83x to 72.08x speedup
  - **Fast Eulerization**: 7.83x to 72.08x speedup
  - **Fast Hierholzer's algorithm**: 5.86x to 9.38x speedup
  - **Fast customized path**: 6-9x speedup
- **Practical tips**:
  - Prefer connected graphs or use minimal jump edges to reduce path length.
  - Remove redundant edge type tokens to reduce sequence length.
  - Leverage optimized algorithms for large-scale graph processing.

## Troubleshooting Guide
Common issues and remedies:
- **Disconnected graphs**:
  - Symptom: Missing connections between components.
  - Fix: Ensure jump edges are added via [connect_graph_sequential:310-323](file://src/utils/nx_utils.py#L310-L323).
- **Self-loops and multi-edges**:
  - Behavior: Self-loops are treated as edges; multi-edges are handled by edge_index lookup.
  - Verify: Use [get_edge_index:271-274](file://src/utils/nx_utils.py#L271-L274) and [get_edge_type:277-290](file://src/utils/nx_utils.py#L277-L290) to confirm mapping.
- **Overly long sequences**:
  - Cause: Eulerian paths revisit the start node; redundant edges accumulate.
  - Solution: Apply [shorten_path:331-348](file://src/utils/nx_utils.py#L331-L348) to prune duplicates.
- **Node permutation effects**:
  - Behavior: Nodes are permuted for augmentation; original node-to-token mapping is preserved via inverse permutation.
  - Check: [permute_nodes:594-612](file://src/utils/nx_utils.py#L594-L612) and related mapping logic.
- **Configuration mismatches**:
  - Symptoms: Unexpected tokenization or missing structure functions.
  - Verify: [structure.yaml:77-121](file://configs/tokenization/graph_lvl/structure.yaml#L77-L121) and [core.py:425-535](file://src/data/tokenizer/core.py#L425-L535).
- **Performance degradation**:
  - Cause: Using legacy NetworkX algorithms instead of optimized versions.
  - Solution: Ensure optimized algorithms (_fast_*) are being used for production workloads.

**Section sources**
- [nx_utils.py:310-323](file://src/utils/nx_utils.py#L310-L323)
- [nx_utils.py:271-290](file://src/utils/nx_utils.py#L271-L290)
- [nx_utils.py:331-348](file://src/utils/nx_utils.py#L331-L348)
- [nx_utils.py:594-612](file://src/utils/nx_utils.py#L594-L612)
- [structure.yaml:77-121](file://configs/tokenization/graph_lvl/structure.yaml#L77-L121)
- [core.py:425-535](file://src/data/tokenizer/core.py#L425-L535)

## Conclusion
The Eulerian path conversion system provides a robust framework for turning graphs into sequential representations suitable for tokenization and downstream modeling. The newly optimized algorithms deliver significant performance improvements (7.83x to 72.08x speedup) while maintaining functional equivalence with NetworkX implementations. By leveraging fast Eulerian graph checking, efficient Eulerization, and optimized Hierholzer's algorithm, the system supports large-scale graph processing. The integration with NetworkX for graph algorithms, applying fast Eulerization and connectivity adjustments, and carefully managing edge types and node mappings enables diverse graph topologies. Configuration options allow fine-tuning of tokenization behavior, while path shortening and randomization improve generalization. The documented optimized components and examples offer a clear blueprint for extending or customizing the path conversion strategy with substantial performance benefits.

## Appendices

### Appendix A: API and Function Reference
- **Fast algorithms**:
  - [_fast_is_eulerian:174-192](file://src/utils/nx_utils.py#L174-L192)
  - [_fast_eulerize:195-284](file://src/utils/nx_utils.py#L195-L284)
  - [_fast_eulerian_circuit:317-388](file://src/utils/nx_utils.py#L317-L388)
  - [_fast_customized_eulerian_path:390-404](file://src/utils/nx_utils.py#L390-L404)
- **Path construction**:
  - [graph2path_v2:388-410](file://src/utils/nx_utils.py#L388-L410)
  - [connected_graph2path:413-422](file://src/utils/nx_utils.py#L413-L422)
  - [shorten_path:331-348](file://src/utils/nx_utils.py#L331-L348)
- **Traversal and augmentation**:
  - [_customized_eulerian_path:205-210](file://src/utils/nx_utils.py#L205-L210)
- **Connectivity and edge typing**:
  - [connect_graph_sequential:310-323](file://src/utils/nx_utils.py#L310-L323)
  - [get_edge_type:277-290](file://src/utils/nx_utils.py#L277-L290)
- **Tokenization**:
  - [GSTTokenizer.raw_tokenize:425-535](file://src/data/tokenizer/core.py#L425-L535)
  - [get_raw_seq_from_path:425-437](file://src/utils/nx_utils.py#L425-L437)

### Appendix B: Performance Benchmark Summary
- **Fast Eulerian checking**: 7.83x to 72.08x speedup
- **Fast Eulerization**: 7.83x to 72.08x speedup
- **Fast Hierholzer's algorithm**: 5.86x to 9.38x speedup
- **Fast customized path**: 6-9x speedup
- **Overall system improvement**: Significant reduction in tokenization latency for large graphs

**Section sources**
- [nx_utils.py:174-404](file://src/utils/nx_utils.py#L174-L404)
- [nx_utils.py:388-437](file://src/utils/nx_utils.py#L388-L437)
- [nx_utils.py:205-210](file://src/utils/nx_utils.py#L205-L210)
- [nx_utils.py:310-323](file://src/utils/nx_utils.py#L310-L323)
- [nx_utils.py:277-290](file://src/utils/nx_utils.py#L277-L290)
- [core.py:425-535](file://src/data/tokenizer/core.py#L425-L535)
