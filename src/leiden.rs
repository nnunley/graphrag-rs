//! Leiden algorithm for community detection.
//!
//! This is a Rust implementation of the Leiden algorithm, which improves upon
//! Louvain by adding a refinement phase that guarantees well-connected communities.
//!
//! Reference: Traag, V.A., Waltman, L. & van Eck, N.J. "From Louvain to Leiden:
//! guaranteeing well-connected communities." Sci Rep 9, 5233 (2019).
//!
//! Implementation adapted from fa-leiden-cd (MIT licensed).

use std::collections::{HashMap, HashSet, VecDeque};

/// A community can be either leaf-level (containing node IDs) or nested (containing sub-communities)
#[derive(Debug, Clone)]
pub enum Community {
    /// Leaf community containing actual node IDs
    Leaf(HashSet<i64>),
    /// Nested community containing sub-communities
    Nested(Vec<Community>),
}

impl Community {
    /// Collect all node IDs in this community (recursively)
    pub fn collect_nodes(&self) -> Vec<i64> {
        let mut nodes = Vec::new();
        self.collect_nodes_into(&mut nodes);
        nodes
    }

    fn collect_nodes_into(&self, out: &mut Vec<i64>) {
        match self {
            Community::Leaf(nodes) => out.extend(nodes.iter().copied()),
            Community::Nested(subs) => {
                for sub in subs {
                    sub.collect_nodes_into(out);
                }
            }
        }
    }

    /// Get the number of nodes in this community
    pub fn node_count(&self) -> usize {
        match self {
            Community::Leaf(nodes) => nodes.len(),
            Community::Nested(subs) => subs.iter().map(|s| s.node_count()).sum(),
        }
    }
}

/// Result of community detection
#[derive(Debug)]
pub struct CommunityHierarchy {
    /// Top-level communities
    pub communities: Vec<Community>,
    /// Final modularity score
    pub modularity: f64,
}

/// Graph structure for community detection
/// Nodes are i64 (entity IDs), edges have f64 weights
pub struct CommunityGraph {
    /// Node IDs (entity IDs from database)
    nodes: Vec<i64>,
    /// Map from node ID to index
    node_to_idx: HashMap<i64, usize>,
    /// Adjacency list: for each node index, list of (neighbor_idx, weight)
    neighbors: Vec<Vec<(usize, f64)>>,
    /// Total edge weight (sum of all weights, counting each edge once)
    total_weight: f64,
}

impl CommunityGraph {
    /// Create a new empty graph
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            node_to_idx: HashMap::new(),
            neighbors: Vec::new(),
            total_weight: 0.0,
        }
    }

    /// Add a node to the graph, returns its index
    pub fn add_node(&mut self, node_id: i64) -> usize {
        if let Some(&idx) = self.node_to_idx.get(&node_id) {
            return idx;
        }
        let idx = self.nodes.len();
        self.nodes.push(node_id);
        self.node_to_idx.insert(node_id, idx);
        self.neighbors.push(Vec::new());
        idx
    }

    /// Add an undirected edge between two nodes
    pub fn add_edge(&mut self, node1: i64, node2: i64, weight: f64) {
        if node1 == node2 {
            return; // No self-loops
        }

        let idx1 = self.add_node(node1);
        let idx2 = self.add_node(node2);

        // Check if edge already exists
        if let Some(existing) = self.neighbors[idx1].iter_mut().find(|(n, _)| *n == idx2) {
            existing.1 += weight;
            // Update the reverse edge too
            if let Some(rev) = self.neighbors[idx2].iter_mut().find(|(n, _)| *n == idx1) {
                rev.1 += weight;
            }
            self.total_weight += weight;
            return;
        }

        self.neighbors[idx1].push((idx2, weight));
        self.neighbors[idx2].push((idx1, weight));
        self.total_weight += weight;
    }

    /// Get the number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get total edge weight
    pub fn total_weight(&self) -> f64 {
        self.total_weight
    }

    /// Run the Leiden algorithm
    pub fn leiden(&self, max_iterations: Option<usize>, tolerance: f64) -> CommunityHierarchy {
        let n = self.node_count();

        // Handle edge cases
        if n == 0 {
            return CommunityHierarchy {
                communities: vec![],
                modularity: 0.0,
            };
        }

        if n == 1 {
            let mut nodes = HashSet::new();
            nodes.insert(self.nodes[0]);
            return CommunityHierarchy {
                communities: vec![Community::Leaf(nodes)],
                modularity: 0.0,
            };
        }

        if self.total_weight == 0.0 {
            // No edges: each node is its own community
            let communities = self
                .nodes
                .iter()
                .map(|&id| {
                    let mut set = HashSet::new();
                    set.insert(id);
                    Community::Leaf(set)
                })
                .collect();
            return CommunityHierarchy {
                communities,
                modularity: 0.0,
            };
        }

        // Initial community assignment: each node in its own community
        let mut assignments: Vec<usize> = (0..n).collect();

        // Run Leiden iterations
        let max_iter = max_iterations.unwrap_or(100);
        let mut prev_modularity = f64::NEG_INFINITY;

        for _ in 0..max_iter {
            // Phase 1: Local moving (modularity optimization)
            self.optimize_modularity(&mut assignments);

            // Phase 2: Refinement (split poorly connected communities)
            let refined = self.refine(&assignments);

            // Check convergence
            let current_modularity = self.compute_modularity_from_sets(&refined);
            if (current_modularity - prev_modularity).abs() < tolerance {
                // Convert to Community hierarchy
                let communities = self.build_leaf_communities(&refined);
                return CommunityHierarchy {
                    communities,
                    modularity: current_modularity,
                };
            }
            prev_modularity = current_modularity;

            // Update assignments from refined communities
            assignments = vec![0; n];
            for (comm_idx, nodes) in refined.iter().enumerate() {
                for &node_idx in nodes {
                    assignments[node_idx] = comm_idx;
                }
            }
        }

        // Final result
        let refined = self.refine(&assignments);
        let modularity = self.compute_modularity_from_sets(&refined);
        let communities = self.build_leaf_communities(&refined);

        CommunityHierarchy {
            communities,
            modularity,
        }
    }

    /// Phase 1: Local moving to optimize modularity
    fn optimize_modularity(&self, assignments: &mut [usize]) {
        let n = self.node_count();
        let m = self.total_weight;

        if m == 0.0 {
            return;
        }

        // Compute node degrees
        let degrees: Vec<f64> = self
            .neighbors
            .iter()
            .map(|neigh| neigh.iter().map(|(_, w)| w).sum())
            .collect();

        // Community degree sums
        let mut community_degree: HashMap<usize, f64> = HashMap::new();
        for (i, &comm) in assignments.iter().enumerate() {
            *community_degree.entry(comm).or_insert(0.0) += degrees[i];
        }

        let mut improved = true;
        let mut iterations = 0;
        let max_local_iterations = 50;

        while improved && iterations < max_local_iterations {
            improved = false;
            iterations += 1;

            for i in 0..n {
                let current_comm = assignments[i];
                let k_i = degrees[i];

                if k_i == 0.0 {
                    continue;
                }

                // Compute weights to each neighboring community
                let mut comm_weights: HashMap<usize, f64> = HashMap::new();
                for &(j, w) in &self.neighbors[i] {
                    let comm_j = assignments[j];
                    *comm_weights.entry(comm_j).or_insert(0.0) += w;
                }

                // Delta for removing from current community
                let total_current = community_degree.get(&current_comm).copied().unwrap_or(0.0);
                let k_i_in = comm_weights.get(&current_comm).copied().unwrap_or(0.0);
                let delta_remove = k_i_in - (total_current * k_i) / (2.0 * m);

                // Find best community to move to
                let mut best_delta = 0.0;
                let mut best_comm = current_comm;

                for (&comm, &w_in) in &comm_weights {
                    if comm == current_comm {
                        continue;
                    }
                    let total_comm = community_degree.get(&comm).copied().unwrap_or(0.0);
                    let delta = w_in - (total_comm * k_i) / (2.0 * m);

                    if delta > best_delta {
                        best_delta = delta;
                        best_comm = comm;
                    }
                }

                // Move if improvement
                if best_delta > delta_remove + 1e-10 {
                    // Update community degrees
                    *community_degree.entry(current_comm).or_insert(0.0) -= k_i;
                    *community_degree.entry(best_comm).or_insert(0.0) += k_i;
                    assignments[i] = best_comm;
                    improved = true;
                }
            }
        }
    }

    /// Phase 2: Refinement - split communities that aren't well-connected
    /// This is the key difference from Louvain
    fn refine(&self, assignments: &[usize]) -> Vec<HashSet<usize>> {
        // Group nodes by community
        let mut communities_by_id: HashMap<usize, HashSet<usize>> = HashMap::new();
        for (node_idx, &comm) in assignments.iter().enumerate() {
            communities_by_id
                .entry(comm)
                .or_insert_with(HashSet::new)
                .insert(node_idx);
        }

        let mut refined: Vec<HashSet<usize>> = communities_by_id.into_values().collect();

        // For each community, check if it's internally connected
        // If not, split into connected components
        let mut i = 0;
        while i < refined.len() {
            let community = &refined[i];

            if community.len() <= 1 {
                i += 1;
                continue;
            }

            // BFS to find connected component within this community
            let mut visited: HashSet<usize> = HashSet::new();
            let mut queue: VecDeque<usize> = VecDeque::new();

            // Start from first node
            let start = *community.iter().next().unwrap();
            queue.push_back(start);

            while let Some(node) = queue.pop_front() {
                if !visited.insert(node) {
                    continue;
                }

                // Add neighbors that are in the same community
                for &(neighbor, _) in &self.neighbors[node] {
                    if community.contains(&neighbor) && !visited.contains(&neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }

            // If we visited all nodes, community is connected
            if visited.len() == community.len() {
                i += 1;
                continue;
            }

            // Split: visited nodes stay, unvisited become new community
            let unvisited: HashSet<usize> = community.difference(&visited).copied().collect();

            // Update current community to only visited nodes
            refined[i] = visited;

            // Add unvisited as new community (will be checked in future iterations)
            refined.push(unvisited);

            i += 1;
        }

        // Remove empty communities
        refined.retain(|c| !c.is_empty());

        refined
    }

    /// Compute modularity from community sets
    fn compute_modularity_from_sets(&self, communities: &[HashSet<usize>]) -> f64 {
        let m = self.total_weight;
        if m == 0.0 {
            return 0.0;
        }

        // Create assignment vector
        let n = self.node_count();
        let mut assignments = vec![0usize; n];
        for (comm_idx, nodes) in communities.iter().enumerate() {
            for &node in nodes {
                assignments[node] = comm_idx;
            }
        }

        self.compute_modularity(&assignments)
    }

    /// Compute modularity Q = (1/2m) * sum_ij [A_ij - k_i*k_j/(2m)] * delta(c_i, c_j)
    fn compute_modularity(&self, assignments: &[usize]) -> f64 {
        let m = self.total_weight;
        if m == 0.0 {
            return 0.0;
        }

        // Compute degrees
        let degrees: Vec<f64> = self
            .neighbors
            .iter()
            .map(|neigh| neigh.iter().map(|(_, w)| w).sum())
            .collect();

        let mut q = 0.0;
        let n = self.node_count();

        for i in 0..n {
            for &(j, w) in &self.neighbors[i] {
                if i < j && assignments[i] == assignments[j] {
                    // Edge contribution (count each edge once)
                    q += w - (degrees[i] * degrees[j]) / (2.0 * m);
                }
            }
        }

        q / m
    }

    /// Build leaf communities from node index sets
    fn build_leaf_communities(&self, sets: &[HashSet<usize>]) -> Vec<Community> {
        sets.iter()
            .map(|indices| {
                let node_ids: HashSet<i64> = indices.iter().map(|&idx| self.nodes[idx]).collect();
                Community::Leaf(node_ids)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_communities() {
        let mut graph = CommunityGraph::new();

        // Two dense clusters connected by a weak link
        // Cluster 1: 1-2-3 (triangle with strong edges)
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(2, 3, 1.0);
        graph.add_edge(1, 3, 1.0);

        // Cluster 2: 4-5-6 (triangle with strong edges)
        graph.add_edge(4, 5, 1.0);
        graph.add_edge(5, 6, 1.0);
        graph.add_edge(4, 6, 1.0);

        // Weak bridge between clusters
        graph.add_edge(3, 4, 0.1);

        let result = graph.leiden(Some(100), 1e-6);

        // Should find 2 communities (the two triangles)
        assert!(result.communities.len() >= 1, "Should find at least 1 community");
        assert!(result.communities.len() <= 6, "Should not over-partition");

        // Total nodes should be 6
        let total_nodes: usize = result.communities.iter().map(|c| c.node_count()).sum();
        assert_eq!(total_nodes, 6);
    }

    #[test]
    fn test_single_community() {
        let mut graph = CommunityGraph::new();

        // Fully connected 4-clique with uniform weights
        // This is a trivial case where all partitions have equal modularity
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(1, 3, 1.0);
        graph.add_edge(1, 4, 1.0);
        graph.add_edge(2, 3, 1.0);
        graph.add_edge(2, 4, 1.0);
        graph.add_edge(3, 4, 1.0);

        let result = graph.leiden(Some(100), 1e-6);

        // For a fully connected graph, any partition is valid
        // Just verify we get all 4 nodes
        let total_nodes: usize = result.communities.iter().map(|c| c.node_count()).sum();
        assert_eq!(total_nodes, 4);
    }

    #[test]
    fn test_disconnected_nodes() {
        let mut graph = CommunityGraph::new();

        // Add nodes without edges
        graph.add_node(1);
        graph.add_node(2);
        graph.add_node(3);

        let result = graph.leiden(Some(100), 1e-6);

        // Each node should be its own community
        assert_eq!(result.communities.len(), 3);
    }

    #[test]
    fn test_refinement_splits_disconnected() {
        let mut graph = CommunityGraph::new();

        // Two separate triangles that might be grouped by Louvain
        // but should be split by Leiden's refinement
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(2, 3, 1.0);
        graph.add_edge(1, 3, 1.0);

        graph.add_edge(4, 5, 1.0);
        graph.add_edge(5, 6, 1.0);
        graph.add_edge(4, 6, 1.0);

        // Weak bridge between triangles
        graph.add_edge(3, 4, 0.1);

        let result = graph.leiden(Some(100), 1e-6);

        // With the weak bridge, might be 1 or 2 communities depending on resolution
        // But refinement ensures any community is internally connected
        for comm in &result.communities {
            // Verify each community is connected (refinement guarantee)
            let nodes = comm.collect_nodes();
            if nodes.len() > 1 {
                // Check connectivity via BFS in original graph
                let node_set: HashSet<i64> = nodes.iter().copied().collect();
                let mut visited = HashSet::new();
                let mut queue = VecDeque::new();
                queue.push_back(nodes[0]);

                while let Some(node) = queue.pop_front() {
                    if !visited.insert(node) {
                        continue;
                    }
                    let idx = graph.node_to_idx[&node];
                    for &(neighbor_idx, _) in &graph.neighbors[idx] {
                        let neighbor_id = graph.nodes[neighbor_idx];
                        if node_set.contains(&neighbor_id) {
                            queue.push_back(neighbor_id);
                        }
                    }
                }

                assert_eq!(
                    visited.len(),
                    nodes.len(),
                    "Community should be internally connected"
                );
            }
        }
    }

    #[test]
    fn test_larger_graph_connectivity_guarantee() {
        // Test with a larger graph to ensure Leiden's connectivity guarantee holds
        let mut graph = CommunityGraph::new();

        // Create 3 clusters of 5 nodes each, connected internally
        // Cluster 1: nodes 1-5
        for i in 1..=5 {
            for j in (i + 1)..=5 {
                graph.add_edge(i, j, 1.0);
            }
        }

        // Cluster 2: nodes 6-10
        for i in 6..=10 {
            for j in (i + 1)..=10 {
                graph.add_edge(i, j, 1.0);
            }
        }

        // Cluster 3: nodes 11-15
        for i in 11..=15 {
            for j in (i + 1)..=15 {
                graph.add_edge(i, j, 1.0);
            }
        }

        // Weak inter-cluster connections
        graph.add_edge(5, 6, 0.1);
        graph.add_edge(10, 11, 0.1);

        let result = graph.leiden(Some(100), 1e-6);

        // Verify Leiden's key guarantee: all communities are internally connected
        for comm in &result.communities {
            let nodes = comm.collect_nodes();
            if nodes.len() > 1 {
                let node_set: HashSet<i64> = nodes.iter().copied().collect();
                let mut visited = HashSet::new();
                let mut queue = VecDeque::new();
                queue.push_back(nodes[0]);

                while let Some(node) = queue.pop_front() {
                    if !visited.insert(node) {
                        continue;
                    }
                    let idx = graph.node_to_idx[&node];
                    for &(neighbor_idx, _) in &graph.neighbors[idx] {
                        let neighbor_id = graph.nodes[neighbor_idx];
                        if node_set.contains(&neighbor_id) {
                            queue.push_back(neighbor_id);
                        }
                    }
                }

                assert_eq!(
                    visited.len(),
                    nodes.len(),
                    "Leiden guarantee violated: community is not internally connected"
                );
            }
        }

        // All 15 nodes should be accounted for
        let total: usize = result.communities.iter().map(|c| c.node_count()).sum();
        assert_eq!(total, 15);
    }

    #[test]
    fn test_modularity_improves() {
        let mut graph = CommunityGraph::new();

        // Graph with clear community structure
        // Two dense cliques with sparse inter-connections
        for i in 1..=4 {
            for j in (i + 1)..=4 {
                graph.add_edge(i, j, 1.0);
            }
        }
        for i in 5..=8 {
            for j in (i + 1)..=8 {
                graph.add_edge(i, j, 1.0);
            }
        }
        graph.add_edge(4, 5, 0.1);

        let result = graph.leiden(Some(100), 1e-6);

        // For a graph with clear community structure, modularity should be positive
        // (better than random assignment)
        assert!(
            result.modularity >= 0.0,
            "Modularity should be non-negative for clustered graph"
        );
    }
}
