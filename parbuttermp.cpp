#include <omp.h>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>
#include <random>
#include <chrono>
#include <cstdlib>
#include <climits>
#include <unordered_map>
using namespace std;
using namespace std::chrono;
// Type definitions
using Vertex = int64_t;
using Edge = pair<Vertex, Vertex>;
using Wedge = std::tuple<pair<Vertex, Vertex>, Vertex>;
#define BATCH_SIZE 10000

// Hash function for std::pair<Vertex, Vertex>
struct pair_hash {
    std::size_t operator()(const pair<Vertex, Vertex>& p) const {
        auto h1 = hash<Vertex>{}(p.first);
        auto h2 = hash<Vertex>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

size_t estimate_memory_usage(const std::vector<Wedge>& W, const std::unordered_map<std::pair<Vertex, Vertex>, std::vector<Vertex>, pair_hash>& wedge_freq) {
    size_t total = 0;
    total += W.size() * sizeof(Wedge);
    for (const auto& [_, centers] : wedge_freq) {
        total += centers.size() * sizeof(Vertex) + sizeof(std::pair<Vertex, Vertex>);
    }
    std::cout << "Estimated memory usage: " << total / (1024.0 * 1024.0) << " MB\n" << std::flush;
    return total;
}

// Graph in CSR format
struct Graph {
    vector<Vertex> xadj; // Offsets
    vector<Vertex> adjncy; // CSR neighbors
    Vertex nvtxs; // Total vertices (|U| + |V|)
    Vertex nedges; // Total number of edges
    Vertex nU; // Vertices in U
    Vertex nV; // Vertices in V
    vector<Vertex> U_vertices; // Vertices in U
    vector<Vertex> V_vertices; // Vertices in V
};

// Load the bipartite graph
Graph extract(const string& filename) {
    Graph g;
    vector<Edge> edges;
    set<Vertex> U_set, V_set;
    Vertex max_vertex = -1;
    Vertex total_edges = 0;
    vector<bool> is_U, is_V;
    vector<Vertex> temp_xadj;
    vector<Vertex> final_xadj;
    double weight;
    string stmp;

    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Cannot open file " << filename << endl;
        exit(1);
    }

    // First pass: count degrees and collect vertices
    temp_xadj.resize(4040, 0);
    is_U.resize(4040, false);
    is_V.resize(4040, false);
    edges.reserve(BATCH_SIZE);
    Vertex u, v;
    while (file >> u >> v >> weight >> stmp) {
        if (u < 0 || v < 0) {
            cerr << "Error: Negative vertex ID (" << u << "," << v << ")" << endl;
            exit(1);
        }
        edges.emplace_back(u, v);
        U_set.insert(u);
        V_set.insert(v);
        max_vertex = max({max_vertex, u, v});
        if (edges.size() >= BATCH_SIZE) {
            cout << "Processing batch" << endl;
            if (max_vertex >= temp_xadj.size()) {
                temp_xadj.resize(max_vertex + 1, 0);
                is_U.resize(max_vertex + 1, false);
                is_V.resize(max_vertex + 1, false);
            }
            #pragma omp parallel for
            for (size_t i = 0; i < edges.size(); i++) {
                Vertex bu = edges[i].first;
                Vertex bv = edges[i].second;
                #pragma omp atomic
                temp_xadj[bu + 1]++;
                #pragma omp atomic
                temp_xadj[bv + 1]++;
                #pragma omp critical
                {
                    is_U[bu] = true;
                    is_V[bv] = true;
                }
            }
            total_edges += edges.size();
            edges.clear();
        }
    }
    // Process final batch
    if (!edges.empty()) {
        if (max_vertex >= temp_xadj.size()) {
            temp_xadj.resize(max_vertex + 1, 0);
            is_U.resize(max_vertex + 1, false);
            is_V.resize(max_vertex + 1, false);
        }
        #pragma omp parallel for
        for (size_t i = 0; i < edges.size(); i++) {
            Vertex bu = edges[i].first;
            Vertex bv = edges[i].second;
            #pragma omp atomic
            temp_xadj[bu + 1]++;
            #pragma omp atomic
            temp_xadj[bv + 1]++;
            #pragma omp critical
            {
                is_U[bu] = true;
                is_V[bv] = true;
            }
        }
        total_edges += edges.size();
        edges.clear();
    }
    file.close();

    // Finalize processed data
    g.nedges = total_edges;
    g.nvtxs = max_vertex + 1;
    for (Vertex v = 0; v <= max_vertex; ++v) {
        if (is_V[v] && !is_U[v]) g.V_vertices.push_back(v);
        if (is_U[v]) g.U_vertices.push_back(v);
    }
    g.nU = g.U_vertices.size();
    g.nV = g.V_vertices.size();
    if (g.nU + g.nV > g.nvtxs) {
        g.nvtxs = g.nU + g.nV;
    }

    // Compute prefix sum
    final_xadj.resize(g.nvtxs + 1, 0);
    for (Vertex v = 1; v <= g.nvtxs; v++) {
        final_xadj[v] = final_xadj[v - 1] + (v < temp_xadj.size() ? temp_xadj[v] : 0);
    }
    g.xadj = final_xadj;

    // Second pass: build adjacency list
    g.adjncy.resize(g.nedges * 2);
    vector<size_t> offsets(g.nvtxs, 0);
    ifstream file2(filename);
    if (!file2.is_open()) {
        cerr << "Error: Cannot open file " << filename << endl;
        exit(1);
    }
    edges.clear();
    edges.reserve(BATCH_SIZE);
    while (file2 >> u >> v >> weight >> stmp) {
        edges.emplace_back(u, v);
        if (edges.size() >= BATCH_SIZE) {
            #pragma omp parallel for
            for (size_t i = 0; i < edges.size(); ++i) {
                Vertex bu = edges[i].first;
                Vertex bv = edges[i].second;
                size_t u_offset, v_offset;
                #pragma omp critical
                {
                    u_offset = g.xadj[bu] + offsets[bu]++;
                    v_offset = g.xadj[bv] + offsets[bv]++;
                }
                g.adjncy[u_offset] = bv;
                g.adjncy[v_offset] = bu;
            }
            edges.clear();
        }
    }
    // Process final batch
    if (!edges.empty()) {
        #pragma omp parallel for
        for (size_t i = 0; i < edges.size(); ++i) {
            Vertex bu = edges[i].first;
            Vertex bv = edges[i].second;
            size_t u_offset, v_offset;
            #pragma omp critical
            {
                u_offset = g.xadj[bu] + offsets[bu]++;
                v_offset = g.xadj[bv] + offsets[bv]++;
            }
            g.adjncy[u_offset] = bv;
            g.adjncy[v_offset] = bu;
        }
    }
    file2.close();
    return g;
}

// Graph statistics
struct GraphStats {
    Vertex max_degU, max_degV;
    Vertex min_degU, min_degV;
    double avg_degU, avg_degV;
    bool is_bipartite;
};

GraphStats analyzeGraph(const Graph& g) {
    GraphStats stats;
    stats.max_degU = 0;
    stats.max_degV = 0;
    stats.min_degU = g.nedges;
    stats.min_degV = g.nedges;
    stats.avg_degU = 0;
    stats.avg_degV = 0;
    stats.is_bipartite = true;

    vector<Vertex> degrees(g.nvtxs);
    #pragma omp parallel for
    for (Vertex v = 0; v < g.nvtxs; v++) {
        degrees[v] = g.xadj[v + 1] - g.xadj[v];
    }

    // Compute stats for U
    Vertex local_max_degU = 0, local_min_degU = g.nedges;
    double local_avg_degU = 0;
    bool local_bipartite = true;
    #pragma omp parallel for reduction(max:local_max_degU) reduction(min:local_min_degU) reduction(+:local_avg_degU) reduction(&:local_bipartite)
    for (size_t i = 0; i < g.U_vertices.size(); ++i) {
        Vertex u = g.U_vertices[i];
        local_max_degU = max(local_max_degU, degrees[u]);
        local_min_degU = min(local_min_degU, degrees[u]);
        local_avg_degU += degrees[u];
        for (size_t j = g.xadj[u]; j < g.xadj[u + 1]; ++j) {
            Vertex v = g.adjncy[j];
            if (find(g.U_vertices.begin(), g.U_vertices.end(), v) != g.U_vertices.end()) {
                local_bipartite = false;
            }
        }
    }
    stats.max_degU = local_max_degU;
    stats.min_degU = local_min_degU;
    stats.avg_degU = local_avg_degU;

    // Compute stats for V
    Vertex local_max_degV = 0, local_min_degV = g.nedges;
    double local_avg_degV = 0;
    #pragma omp parallel for reduction(max:local_max_degV) reduction(min:local_min_degV) reduction(+:local_avg_degV) reduction(&:local_bipartite)
    for (size_t i = 0; i < g.V_vertices.size(); ++i) {
        Vertex v = g.V_vertices[i];
        local_max_degV = max(local_max_degV, degrees[v]);
        local_min_degV = min(local_min_degV, degrees[v]);
        local_avg_degV += degrees[v];
        for (size_t j = g.xadj[v]; j < g.xadj[v + 1]; ++j) {
            Vertex u = g.adjncy[j];
            if (find(g.V_vertices.begin(), g.V_vertices.end(), u) != g.V_vertices.end()) {
                local_bipartite = false;
            }
        }
    }
    stats.max_degV = local_max_degV;
    stats.min_degV = local_min_degV;
    stats.avg_degV = local_avg_degV;
    stats.is_bipartite = local_bipartite;

    stats.avg_degU = g.nU ? stats.avg_degU / g.nU : 0;
    stats.avg_degV = g.nV ? stats.avg_degV / g.nV : 0;

    return stats;
}

// Preprocessing algorithm
struct PreprocessedGraph {
    vector<Vertex> xadj;
    vector<Vertex> adjncy;
    vector<Vertex> global_ranks; // Vertex to rank mapping
    vector<Vertex> vertex_map; // Rank to vertex map
};

PreprocessedGraph Preprocessing(const Graph& g) {
    PreprocessedGraph pg;
    pg.global_ranks.resize(g.nvtxs, -1);
    pg.vertex_map.resize(g.nvtxs, -1);

    // Compute approximate degree (log-degree)
    vector<pair<double, Vertex>> degree_order;
    degree_order.reserve(g.nvtxs);
    #pragma omp parallel
    {
        vector<pair<double, Vertex>> local_degree_order;
        local_degree_order.reserve(g.nvtxs / omp_get_num_threads() + 1) ; 
        #pragma omp for
        for (Vertex v = 0; v < g.nvtxs; v++) {
            double logDeg = log2(g.xadj[v + 1] - g.xadj[v] + 1.0);
            local_degree_order.emplace_back(-logDeg, v);
        }
        #pragma omp critical
        degree_order.insert(degree_order.end(), local_degree_order.begin(), local_degree_order.end());
    }
    sort(degree_order.begin(), degree_order.end());

    for (size_t i = 0; i < degree_order.size(); i++) {
        pg.global_ranks[degree_order[i].second] = i;
        pg.vertex_map[i] = degree_order[i].second;
    }

    // Rebuild graph with ranked vertices
    pg.xadj.resize(g.nvtxs + 1);
    pg.adjncy.resize(g.nedges * 2);
    vector<size_t> edgeCounts(g.nvtxs, 0);
    #pragma omp parallel for schedule(dynamic)
    for (Vertex v = 0; v < g.nvtxs; ++v) {
        Vertex vr = pg.global_ranks[v];
        vector<Vertex> neighbors;
        neighbors.reserve(g.xadj[v + 1] - g.xadj[v]);
        for (size_t i = g.xadj[v]; i < g.xadj[v + 1]; ++i) {
            neighbors.push_back(pg.global_ranks[g.adjncy[i]]);
        }
        sort(neighbors.begin(), neighbors.end(), greater<Vertex>());
        pg.xadj[vr + 1] = neighbors.size();
        for (Vertex u : neighbors) {
            pg.adjncy[pg.xadj[vr] + edgeCounts[vr]++] = u;
        }
    }

    // Compute prefix sum for xadj
    for (Vertex v = 1; v <= g.nvtxs; ++v) {
        pg.xadj[v] += pg.xadj[v - 1];
    }

    return pg;
}

// Bucket partitioning for OpenMP threads
struct Bucket {
    vector<Vertex> vertices;
    Vertex wedgeCount;
};

vector<Bucket> bucket_partition(const Graph& g, const PreprocessedGraph& pg, int nparts) {
    vector<Bucket> buckets(nparts);
    vector<pair<Vertex, Vertex>> wedge_order;
    wedge_order.reserve(g.nvtxs);
    #pragma omp parallel
    {
        vector<pair<Vertex, Vertex>> local_wedge_order;
        local_wedge_order.reserve(g.nvtxs / omp_get_num_threads() + 1);
        #pragma omp for
        for (Vertex v = 0; v < g.nvtxs; ++v) {
            Vertex degree = g.xadj[v + 1] - g.xadj[v];
            Vertex wedgeCount = degree * (degree - 1) / 2;
            local_wedge_order.emplace_back(wedgeCount, v);
        }
        #pragma omp critical
        wedge_order.insert(wedge_order.end(), local_wedge_order.begin(), local_wedge_order.end());
    }
    sort(wedge_order.begin(), wedge_order.end(), greater<>());

    // Compute wedge counts
    Vertex totalCount = 0;
    for (const auto& [wc, v] : wedge_order) {
        totalCount += wc;
    }
    Vertex maxWedgesPerBucket = totalCount / nparts + 1;
    vector<Vertex> wedge_counts(nparts, 0);
    for (int i = 0; i < nparts; ++i) {
        buckets[i].wedgeCount = 0;
    }

    // Assign vertices to buckets
    for (const auto& [wc, v] : wedge_order) {
        int min_bucket = 0;
        Vertex min_wedges = wedge_counts[0];
        for (int i = 1; i < nparts; ++i) {
            if (wedge_counts[i] < min_wedges) {
                min_wedges = wedge_counts[i];
                min_bucket = i;
            }
        }
        buckets[min_bucket].vertices.push_back(v);
        buckets[min_bucket].wedgeCount += wc;
        wedge_counts[min_bucket] += wc;
    }

    // Debug: Print wedge counts
    for (int i = 0; i < nparts; ++i) {
        cout << "Bucket " << i << ": " << buckets[i].vertices.size() 
             << " vertices, " << buckets[i].wedgeCount << " wedges\n" << flush;
    }

    return buckets;
}

// Output graph structure and analysis
void output_analysis(const Graph& g, const GraphStats& stats, const PreprocessedGraph& pg, const vector<Bucket>& buckets, vector<Vertex>& butterfly_counts) {
    cout << "Graph Structure Analysis:\n";
    cout << "  Total Vertices: " << g.nvtxs << " (|U| = " << g.nU << ", |V| = " << g.nV << ")\n";
    cout << "  Total Edges: " << g.nedges << "\n";
    cout << "  Is Bipartite: " << (stats.is_bipartite ? "Yes" : "No") << "\n";
    cout << "  U Partition:\n";
    cout << "    Max Degree: " << stats.max_degU << "\n";
    cout << "    Min Degree: " << stats.min_degU << "\n";
    cout << "    Avg Degree: " << stats.avg_degU << "\n";
    cout << "  V Partition:\n";
    cout << "    Max Degree: " << stats.max_degV << "\n";
    cout << "    Min Degree: " << stats.min_degV << "\n";
    cout << "    Avg Degree: " << stats.avg_degV << "\n";

    // Rank distribution
    vector<int> rank_buckets(10, 0);
    for (Vertex vr : pg.global_ranks) {
        if (vr >= 0) {
            int bucket = min(static_cast<int>(vr * 10 / g.nvtxs), 9);
            rank_buckets[bucket]++;
        }
    }
    cout << "  Rank Distribution (10 buckets):\n";
    for (int i = 0; i < 10; ++i) {
        cout << "    Bucket " << i << " (" << (i * g.nvtxs / 10) << "-" << ((i + 1) * g.nvtxs / 10 - 1) << "): " << rank_buckets[i] << " vertices\n";
    }

    // Bucket partitioning
    cout << "  Bucket Partitioning:\n";
    Vertex total_wedges = 0;
    for (size_t i = 0; i < buckets.size(); ++i) {
        total_wedges += buckets[i].wedgeCount;
    }
    for (size_t i = 0; i < buckets.size(); ++i) {
        double percentage = buckets[i].wedgeCount * 100.0 / total_wedges;
        cout << "    Bucket " << i << ": " << buckets[i].vertices.size() 
             << " vertices, " << buckets[i].wedgeCount << " wedges (" 
             << percentage << "%)\n";
    }

    Vertex total_butterflies = 0;
    for (Vertex count : butterfly_counts) {
        total_butterflies += count;
    }
    total_butterflies /= 4; // Each butterfly counted 4 times
    cout << "  Butterfly Count: " << total_butterflies << "\n";
}

// Wedge aggregation
void get_wedges(const Graph& g, const PreprocessedGraph& pg, const vector<Bucket>& buckets, int thread_id, 
                unordered_map<pair<Vertex, Vertex>, vector<Vertex>, pair_hash>& wedge_freq) {
    const vector<Vertex>& local_vertices = buckets[thread_id].vertices;
    const size_t batch_size = 1000000;

    // Batch processing of wedges
    size_t total_wedges = 0;
    vector<size_t> prefix_sum(local_vertices.size() + 1, 0);
    #pragma omp parallel for reduction(+:total_wedges)
    for (size_t idx = 0; idx < local_vertices.size(); ++idx) {
        Vertex u = local_vertices[idx];
        size_t wedge_count = 0;
        for (size_t i = g.xadj[u]; i < g.xadj[u + 1]; ++i) {
            Vertex v = g.adjncy[i];
            Vertex degree_v = g.xadj[v + 1] - g.xadj[v];
            wedge_count += degree_v - 1;
        }
        prefix_sum[idx + 1] = prefix_sum[idx] + wedge_count;
        total_wedges += wedge_count;
    }
    cout << "Thread " << thread_id << ": Total wedges to process: " << total_wedges << "\n" << flush;

    wedge_freq.reserve(total_wedges / 2);
    vector<Wedge> W_batch(batch_size);
    size_t processed_wedges = 0;

    #pragma omp parallel
    {
        vector<Wedge> thread_W(batch_size);
        size_t thread_processed = 0;
        #pragma omp for schedule(dynamic) nowait
        for (size_t idx = 0; idx < local_vertices.size(); ++idx) {
            Vertex u = local_vertices[idx];
            Vertex degree_u = g.xadj[u + 1] - g.xadj[u];
            if (degree_u > 1000) {
                #pragma omp critical
                cout << "Thread " << thread_id << ": Processing high-degree vertex " << u << " with degree " << degree_u << "\n" << flush;
            }
            size_t wedge_idx = 0;
            for (size_t i = g.xadj[u]; i < g.xadj[u + 1]; ++i) {
                Vertex v = g.adjncy[i];
                for (size_t j = g.xadj[v]; j < g.xadj[v + 1]; ++j) {
                    Vertex u2 = g.adjncy[j];
                    if (u2 == u) continue;
                    thread_W[wedge_idx++] = {{u, u2}, v};
                    if (wedge_idx >= batch_size) {
                        #pragma omp critical
                        {
                            for (size_t k = 0; k < wedge_idx; ++k) {
                                wedge_freq[std::get<0>(thread_W[k])].push_back(std::get<1>(thread_W[k]));
                            }
                            processed_wedges += wedge_idx;
                        }
                        thread_processed += wedge_idx;
                        wedge_idx = 0;
                    }
                }
            }
            if (wedge_idx > 0) {
                #pragma omp critical
                {
                    for (size_t k = 0; k < wedge_idx; ++k) {
                        wedge_freq[std::get<0>(thread_W[k])].push_back(std::get<1>(thread_W[k]));
                    }
                    processed_wedges += wedge_idx;
                }
                thread_processed += wedge_idx;
            }
            if (idx % (local_vertices.size() / 10 + 1) == 0) {
                #pragma omp critical
                cout << "Thread " << thread_id << ": Processed " << idx << " of " << local_vertices.size() 
                     << " vertices, " << processed_wedges << " wedges\n" << flush;
            }
        }
        #pragma omp critical
        cout << "Thread " << thread_id << ": Thread processed " << thread_processed << " wedges\n" << flush;
    }
    cout << "Thread " << thread_id << ": Completed wedge processing, total wedges: " << processed_wedges << "\n" << flush;
}

// Butterfly counting
vector<Vertex> ButterflyCount(const Graph& g, const PreprocessedGraph& pg, const vector<Bucket>& buckets) {
    vector<unordered_map<pair<Vertex, Vertex>, vector<Vertex>, pair_hash>> wedge_freqs(buckets.size());
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        if (thread_id < buckets.size()) {
            get_wedges(g, pg, buckets, thread_id, wedge_freqs[thread_id]);
        }
    }

    // Merge wedge frequencies
    unordered_map<pair<Vertex, Vertex>, vector<Vertex>, pair_hash> wedge_freq;
    size_t total_centers = 0;
    for (const auto& wf : wedge_freqs) {
        for (const auto& [endpoints, centers] : wf) {
            wedge_freq[endpoints].insert(wedge_freq[endpoints].end(), centers.begin(), centers.end());
            total_centers += centers.size();
        }
    }

    // Log memory usage
    estimate_memory_usage({}, wedge_freq);

    // Prepare R and F
    vector<pair<pair<Vertex, Vertex>, Vertex>> R;
    R.reserve(wedge_freq.size());
    vector<size_t> F(wedge_freq.size() + 1, 0);
    size_t idx = 0;
    for (const auto& [endpoints, centers] : wedge_freq) {
        R.emplace_back(endpoints, centers.size());
        F[idx + 1] = F[idx] + centers.size();
        idx++;
    }
    cout << "Computed " << R.size() << " unique wedge endpoint pairs, total centers " << total_centers << "\n" << flush;

    // Count butterflies
    vector<Vertex> butterfly_counts(g.nvtxs, 0);
    #pragma omp parallel
    {
        vector<Vertex> thread_counts(g.nvtxs, 0);
        // Count for endpoints
        #pragma omp for schedule(dynamic) nowait
        for (size_t i = 0; i < R.size(); ++i) {
            auto [endpoints, d] = R[i];
            Vertex u1 = endpoints.first;
            Vertex u2 = endpoints.second;
            Vertex count = (d * (d - 1)) / 2;
            thread_counts[u1] += count;
            thread_counts[u2] += count;
        }
        // Count for centers
        #pragma omp for schedule(dynamic) nowait
        for (size_t i = 0; i < R.size(); ++i) {
            Vertex d = R[i].second;
            for (size_t j = F[i]; j < F[i + 1]; ++j) {
                Vertex v = wedge_freq[R[i].first][j - F[i]];
                thread_counts[v] += d - 1;
            }
        }
        // Manual reduction
        #pragma omp critical
        for (Vertex v = 0; v < g.nvtxs; ++v) {
            butterfly_counts[v] += thread_counts[v];
        }
    }

    return butterfly_counts;
}

int main(int argc, char* argv[]) {
    // Set OpenMP threads
    int num_threads = std::getenv("OMP_NUM_THREADS") ? atoi(std::getenv("OMP_NUM_THREADS")) : 2;
    if (argc > 1) {
        num_threads = atoi(argv[1]);
    }
    if (num_threads < 1) num_threads = 1;
    omp_set_num_threads(num_threads);
    cout << "Using " << num_threads << " OpenMP threads\n";

    // Load graph
    auto t0 = chrono::high_resolution_clock::now();
    string filename = "movies.edges";
    Graph g = extract(filename);
    auto t1 = chrono::high_resolution_clock::now();
    cout << "Graph loading time: " << chrono::duration_cast<milliseconds>(t1 - t0).count() << " ms\n";

    // Analyze graph
    t0 = chrono::high_resolution_clock::now();
    GraphStats stats = analyzeGraph(g);
    t1 = chrono::high_resolution_clock::now();
    cout << "Graph analysis time: " << duration_cast<milliseconds>(t1 - t0).count() << " ms\n";

    // Preprocess graph
    t0 = chrono::high_resolution_clock::now();
    PreprocessedGraph pg = Preprocessing(g);
    t1 = chrono::high_resolution_clock::now();
    cout << "Graph preprocessing time: " << duration_cast<milliseconds>(t1 - t0).count() << " ms\n";

    // Partition graph into buckets
    t0 = chrono::high_resolution_clock::now();
    vector<Bucket> buckets = bucket_partition(g, pg, num_threads);
    t1 =chrono::high_resolution_clock::now();
    cout << "Bucket partitioning time: " << duration_cast<milliseconds>(t1 - t0).count() << " ms\n";

    // Count butterflies
    t0 = chrono::high_resolution_clock::now();
    vector<Vertex> butterfly_counts = ButterflyCount(g, pg, buckets);
    t1 = chrono::high_resolution_clock::now();
    cout << "Butterfly counting time: " << duration_cast<milliseconds>(t1 - t0).count() << " ms\n";

    // Output analysis
    t0 = chrono::high_resolution_clock::now();
    output_analysis(g, stats, pg, buckets, butterfly_counts);
    t1 = chrono::high_resolution_clock::now();
    cout << "Output analysis time: " << duration_cast<milliseconds>(t1 - t0).count() << " ms\n";

    return 0;
}