#include <mpi.h>
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
#include <unistd.h>
using namespace std;

//type definition
using Vertex = int64_t;
using Edge = pair<Vertex , Vertex>;

#define BATCH_SIZE 10000

//Graph in CSR format
struct Graph {
    vector<Vertex> xadj; //Offsets
    vector<Vertex> adjncy; //CSR neighbors
    Vertex nvtxs; //total vertices (|U| + |V|)
    Vertex nedges; //total number of edges
    Vertex nU; //vertices in U
    Vertex nV; //vertices in V
    vector<Vertex> U_vertices; //vertices in U
    vector<Vertex> V_vertices; //vertices in V
};

//Load the bipartite graph
Graph extract(const string& filename,int rank){
    Graph g;
    vector<Edge> edges;
    set<Vertex> U_set, V_set;
    Vertex max_vertex = -1;
    Vertex total_edges = 0;
    vector<bool> is_U, is_V;
    //temp offsets for degree counting.
    vector<Vertex> temp_xadj;
    vector<Vertex> final_xadj;
    //for now temp
    double weight;
    string stmp;
    if(rank == 0) {
        ifstream file(filename);
        if(!file.is_open()) {
            cerr << "Erro: Cannot open the file" << filename << endl;
        }

        //First pass: count degrees and collect the vertices.
        temp_xadj.resize(4040,0); //this is the initial size.
        is_U.resize(4040, false);
        is_V.resize(4040, false);
        edges.reserve(BATCH_SIZE);
        Vertex u,v;
        while( file >> u >> v >> weight >> stmp) {
            u--;
            v--;
            if(u < 0 || v < 0) {
                cerr<<"Error: Negative vertex ID ("<<u<<","<<v<<")"<<endl;
                MPI_Abort(MPI_COMM_WORLD,1);
            }

            edges.emplace_back(u,v);
            U_set.insert(u);
            V_set.insert(v);

            max_vertex = max({max_vertex, u, v}); //get the maximum value from current , u or v value.

            if(edges.size() >= BATCH_SIZE) {
                //process the batch.
                cout<<"Processing batch"<<endl;
                if( max_vertex >= temp_xadj.size()) {
                    temp_xadj.resize(max_vertex + 1, 0);
                    is_U.resize(max_vertex + 1, false);
                    is_V.resize(max_vertex + 1, false);
                }
                //#pragma omp parallel for
                for (int i = 0; i < edges.size(); i++) {
                    Vertex bu = edges[i].first;
                    Vertex bv = edges[i].second;
                    // //update the global histogram of both vertices.
                    // #pragma omp atomic
                    // temp_xadj[bu + 1]++;
                    // #pragma omp atomic
                    // temp_xadj[bv + 1]++;
                    temp_xadj[bu + 1]++;
                    temp_xadj[bv + 1]++;
                    is_U[bu] = true;
                    is_V[bv] = true;
                }
                // Broadcast batch
                Vertex batch_size_local = edges.size();
                cout << "Rank 0: Broadcasting batch of size " << batch_size_local << "\n" << flush;
                MPI_Bcast(&batch_size_local, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
                MPI_Bcast(edges.data(), batch_size_local * 2, MPI_INT64_T, 0, MPI_COMM_WORLD);
                total_edges += edges.size();
                edges.clear();
            }
        }
        //Process the final batch.
        if(!edges.empty()) {
            if(max_vertex >= temp_xadj.size()) {
                temp_xadj.resize(max_vertex+1,0);
                is_U.resize(max_vertex + 1, false);
                is_V.resize(max_vertex + 1, false);
            }
            //#pragma omp parallel for
            for (int i=0; i< edges.size(); i++) {
                Vertex bu = edges[i].first;
                Vertex bv = edges[i].second;
                // //update the global histogram of both vertices.
                // #pragma omp atomic
                // temp_xadj[bu + 1]++;
                // #pragma omp atomic
                // temp_xadj[bv + 1]++;
                temp_xadj[bu + 1]++;
                temp_xadj[bv + 1]++;
                is_U[bu] = true;
                is_V[bv] = true;
            }
            // Broadcast final batch
            Vertex batch_size_local = edges.size();
            cout << "Rank 0: Broadcasting batch of size " << batch_size_local << "\n" << flush;
            MPI_Bcast(&batch_size_local, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
            MPI_Bcast(edges.data(), batch_size_local * 2, MPI_INT64_T, 0, MPI_COMM_WORLD);
            total_edges += edges.size();
            edges.clear();
        }
        // Signal end of batches
        Vertex end_batch = 0;
        cout << "Rank 0: Broadcasting end signal\n" << flush;
        MPI_Bcast(&end_batch, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
        file.close();

        //finalize the processed data.
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

        //compute the prefx sum
        final_xadj.resize(g.nvtxs + 1,0);
        for (Vertex v = 1; v <= g.nvtxs; v++) {
            final_xadj[v] = final_xadj[v-1] + (v < temp_xadj.size() ? temp_xadj[v] : 0);
        }
        
    } else {
        // Non-root nodes receive batches
        temp_xadj.resize(4040, 0);
        is_U.resize(4040, false);
        is_V.resize(4040, false);
        while (true) {
            Vertex batch_size_local;
            MPI_Bcast(&batch_size_local, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
            cout << "Rank " << rank << ": Received batch size " << batch_size_local << "\n" << flush;
            if (batch_size_local <= 0) {
                cout << "Rank " << rank << ": Exiting batch loop\n" << flush;
                break;
            }
            edges.resize(batch_size_local);
            MPI_Bcast(edges.data(), batch_size_local * 2, MPI_INT64_T, 0, MPI_COMM_WORLD);

            // Compute max vertex ID in batch
            Vertex batch_max_vertex = -1;
            for (size_t i = 0; i < edges.size(); ++i) {
                batch_max_vertex = max({batch_max_vertex, edges[i].first, edges[i].second});
            }
            if (batch_max_vertex > max_vertex) {
                max_vertex = batch_max_vertex;
                if (max_vertex >= temp_xadj.size()) {
                    temp_xadj.resize(max_vertex + 1, 0);
                    is_U.resize(max_vertex + 1, false);
                    is_V.resize(max_vertex + 1, false);
                }
            }
            // Process batch
            cout << "Rank " << rank << ": Processing batch of size " << edges.size() << "\n" << flush;
            for (size_t i = 0; i < edges.size(); ++i) {
                Vertex bu = edges[i].first;
                Vertex bv = edges[i].second;
                temp_xadj[bu + 1]++;
                temp_xadj[bv + 1]++;
                is_U[bu] = true;
                is_V[bv] = true;
            }
            total_edges += edges.size();
        }
    }

    //now that the rank 0 processing has completed for reading the graph we move towards Bcast.
    MPI_Barrier(MPI_COMM_WORLD);
    //for metadata
    MPI_Bcast(&g.nU, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&g.nV, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&g.nvtxs, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&g.nedges, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);

    // for the U and V vertices.
    Vertex u_size = g.nU;
    Vertex v_size = g.nV;
    MPI_Bcast(&u_size, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&v_size, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    g.U_vertices.resize(u_size);
    g.V_vertices.resize(v_size);
    MPI_Bcast(g.U_vertices.data(), u_size, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(g.V_vertices.data(), v_size, MPI_INT64_T, 0, MPI_COMM_WORLD);

    //broadcast the xadj
    g.xadj.resize(g.nvtxs + 1);
    if( rank == 0 ) {
        g.xadj = final_xadj;
    }
    MPI_Bcast(g.xadj.data(), g.nvtxs + 1, MPI_INT64_T, 0, MPI_COMM_WORLD);

    //proceed to the second pass for creating the list.
    g.adjncy.resize(g.nedges * 2);
    vector<size_t> offsets(g.nvtxs, 0);
    if(rank == 0) {
        ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        edges.clear();
        edges.reserve(BATCH_SIZE);
        size_t batch_count = 0;
        Vertex u, v;
        //double weight,stmp;
        while (file >> u >> v >> weight >> stmp) {
            u--;
            v--;
            edges.emplace_back(u, v);
            if (edges.size() >= BATCH_SIZE) {
                // Broadcast batch
                Vertex batch_size_local = edges.size();
                cout << "Rank 0: Broadcasting second-pass batch of size " << batch_size_local << "\n" << flush;
                MPI_Bcast(&batch_size_local, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
                MPI_Bcast(edges.data(), batch_size_local * 2, MPI_INT64_T, 0, MPI_COMM_WORLD);

                // Process batch
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
                batch_count++;
            }

        }
        //again processing the final batch.
        if (!edges.empty()) {
            Vertex batch_size_local = edges.size();
            cout << "Rank 0: Broadcasting second-pass final batch of size " << batch_size_local << "\n" << flush;
            MPI_Bcast(&batch_size_local, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
            MPI_Bcast(edges.data(), batch_size_local * 2, MPI_INT64_T, 0, MPI_COMM_WORLD);

            //#pragma omp parallel for
            for (size_t i = 0; i < edges.size(); ++i) {
                // Vertex bu = edges[i].first;
                // Vertex bv = edges[i].second;
                // size_t u_offset, v_offset;
                // #pragma omp critical
                // {
                //     u_offset = g.xadj[bu] + offsets[bu]++;
                //     v_offset = g.xadj[bv] + offsets[bv]++;
                // }
                // g.adjncy[u_offset] = bv;
                // g.adjncy[v_offset] = bu;
                Vertex bu = edges[i].first;
                Vertex bv = edges[i].second;
                size_t u_offset = g.xadj[bu] + offsets[bu]++;
                size_t v_offset = g.xadj[bv] + offsets[bv]++;
                g.adjncy[u_offset] = bv;
                g.adjncy[v_offset] = bu;
            }
        }
        Vertex end_batch = 0;
        cout << "Rank 0: Broadcasting second-pass end signal\n" << flush;
        MPI_Bcast(&end_batch, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
        file.close();

    } else {
        // receive the batches in the non root nodes.
        while (true) {
            Vertex batch_size_local;
            MPI_Bcast(&batch_size_local, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
            cout << "Rank " << rank << ": Received second-pass batch size " << batch_size_local << "\n" << flush;
            if (batch_size_local == 0) {
                cout << "Rank " << rank << ": Exiting second-pass batch loop\n" << flush;
                break;
            }
            edges.resize(batch_size_local);
            MPI_Bcast(edges.data(), batch_size_local * 2, MPI_INT64_T, 0, MPI_COMM_WORLD);

            //#pragma omp parallel for
            for (size_t i = 0; i < edges.size(); ++i) {
                Vertex bu = edges[i].first;
                Vertex bv = edges[i].second;
                size_t u_offset = g.xadj[bu] + offsets[bu]++;
                size_t v_offset = g.xadj[bv] + offsets[bv]++;
                //size_t u_offset, v_offset;
                // #pragma omp critical
                // {
                //     u_offset = g.xadj[bu] + offsets[bu]++;
                //     v_offset = g.xadj[bv] + offsets[bv]++;
                // }
                 g.adjncy[u_offset] = bv;
                 g.adjncy[v_offset] = bu;
            }
        }
    }
    // Optional: Sparsification (uncomment to enable for total count approximation)
    /*
    if (rank == 0) {
        double p = 0.5;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        std::vector<Vertex> sparse_adjncy;
        std::vector<Vertex> sparse_xadj(g.nvtxs + 1, 0);
        size_t new_edge_count = 0;
        for (Vertex v = 0; v < g.nvtxs; ++v) {
            for (size_t i = g.xadj[v]; i < g.xadj[v + 1]; ++i) {
                if (dis(gen) < p) {
                    sparse_adjncy.push_back(g.adjncy[i]);
                    sparse_xadj[v + 1]++;
                    new_edge_count++;
                }
            }
        }
        for (Vertex v = 1; v <= g.nvtxs; ++v) {
            sparse_xadj[v] += sparse_xadj[v - 1];
        }
        g.xadj = std::move(sparse_xadj);
        g.adjncy = std::move(sparse_adjncy);
        g.nedges = new_edge_count / 2;
    }
    MPI_Bcast(&g.nedges, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(g.xadj.data(), g.nvtxs + 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(g.adjncy.data(), g.nedges * 2, MPI_INT64_T, 0, MPI_COMM_WORLD);
    */
    //cout<<"Exiting the first read"<<endl;
    return g;
}


struct GraphStats {
    Vertex max_degU, max_degV;
    Vertex min_degU, min_degV;
    double avg_degU, avg_degV;
    bool is_bipartite;
};

GraphStats analyzeGraph(const Graph& g, int rank) {
    GraphStats stats;
    stats.max_degU = 0;
    stats.max_degV = 0;
    stats.min_degU = g.nedges;
    stats.min_degV = g.nedges;
    stats.avg_degU = 0;
    stats.avg_degV = 0;
    stats.is_bipartite = true;

    /// useing open mp to compute the degrees in parallel . using only 2 threads for hardware limits.
    vector<Vertex> degrees(g.nvtxs);
    #pragma omp parallel for
    for ( Vertex v = 0; v < g.nvtxs; v++ ) {
        degrees[v] = g.xadj[v+1] - g.xadj[v];
    }

    // Check bipartiteness and compute stats for U
    for (Vertex u : g.U_vertices) {
        stats.max_degU = max(stats.max_degU, degrees[u]);
        stats.min_degU = min(stats.min_degU, degrees[u]);
        stats.avg_degU += degrees[u];
        for (size_t i = g.xadj[u]; i < g.xadj[u + 1]; ++i) {
            Vertex v = g.adjncy[i];
            if (std::find(g.U_vertices.begin(), g.U_vertices.end(), v) != g.U_vertices.end()) {
                stats.is_bipartite = false;
            }
        }
    }
    // Compute stats for V
    for (Vertex v : g.V_vertices) {
        stats.max_degV = max(stats.max_degV, degrees[v]);
        stats.min_degV = min(stats.min_degV, degrees[v]);
        stats.avg_degV += degrees[v];
        for (size_t i = g.xadj[v]; i < g.xadj[v + 1]; ++i) {
            Vertex u = g.adjncy[i];
            if (std::find(g.V_vertices.begin(), g.V_vertices.end(), u) != g.V_vertices.end()) {
                stats.is_bipartite = false;
            }
        }
    }
    stats.avg_degU = g.nU ? stats.avg_degU / g.nU : 0;
    stats.avg_degV = g.nV ? stats.avg_degV / g.nV : 0;

    // Aggregate global stats
    Vertex global_max_U, global_max_V, global_min_U, global_min_V;
    double global_avg_U, global_avg_V;
    int local_bipartite = stats.is_bipartite ? 1 : 0;
    int global_bipartite;
    MPI_Reduce(&stats.max_degU, &global_max_U, 1, MPI_INT64_T, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats.max_degV, &global_max_V, 1, MPI_INT64_T, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats.min_degU, &global_min_U, 1, MPI_INT64_T, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats.min_degV, &global_min_V, 1, MPI_INT64_T, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats.avg_degU, &global_avg_U, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats.avg_degV, &global_avg_V, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_bipartite, &global_bipartite, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        stats.max_degU = global_max_U;
        stats.max_degV = global_max_V;
        stats.min_degU = global_min_U;
        stats.min_degV = global_min_V;
        stats.avg_degU = global_avg_U;
        stats.avg_degV = global_avg_V;
        stats.is_bipartite = global_bipartite == 1;
    }

    return stats;
}


//The preprocessing algorithm
struct PreprocessedGraph {
    vector<Vertex> xadj;
    vector<Vertex> adjncy;
    vector<Vertex> global_ranks; // vertex to rank mapping
    vector<Vertex> vertex_map; //rank to vertex map.
};


PreprocessedGraph Preprocessing(const Graph& g, int rank) {
    PreprocessedGraph pg;
    pg.global_ranks.resize(g.nvtxs, -1);
    pg.vertex_map.resize(g.nvtxs, -1);

    //compute approximate degree (log-degreee)
    vector<pair<double,Vertex>> degree_order;

    if( rank == 0 ) {
        for(Vertex v = 0; v < g.nvtxs; v++ ) {
            double logDeg = log2(g.xadj[v+1] - g.xadj[v] + 1.0);
            degree_order.emplace_back(-logDeg, v); //in decreasing order..
        }
        sort(degree_order.begin(), degree_order.end());

        for(int i = 0; i < degree_order.size(); i++) {
            pg.global_ranks[degree_order[i].second] = i;
            pg.vertex_map[i] = degree_order[i].second;
        }
    }
    MPI_Bcast(pg.global_ranks.data(), g.nvtxs, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(pg.vertex_map.data(), g.nvtxs, MPI_INT64_T, 0, MPI_COMM_WORLD);


    //rebuild the graph with the ranked vertices.
    pg.xadj.resize(g.nvtxs+1);
    pg.adjncy.resize(g.nedges * 2);
    vector<size_t> edgeCounts(g.nvtxs,0);

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


//bucket partitioning instead of metis due to distribution issues with bipartite graphs.
struct Bucket {
    vector<Vertex> vertices;
    Vertex edgeCount;
};

vector<Bucket> bucket_partition(const Graph& g, const PreprocessedGraph& pg, int nparts, int rank) {
    vector<Bucket> buckets(nparts);
    if (rank == 0) {
        // Sort vertices by degree
        vector<std::pair<Vertex, Vertex>> degree_order;
        for (Vertex v = 0; v < g.nvtxs; ++v) {
            Vertex degree = g.xadj[v + 1] - g.xadj[v];
            degree_order.emplace_back(degree, v);
        }
        sort(degree_order.begin(), degree_order.end(), std::greater<>());

        // Distribute vertices with edge cap
        vector<Vertex> edgeCounts(nparts, 0);
        Vertex max_edges_per_bucket = g.nedges / nparts + 1;
        int current_bucket = 0;
        for (const auto& [degree, v] : degree_order) {
            if (current_bucket < nparts - 1 && edgeCounts[current_bucket] >= max_edges_per_bucket) {
                current_bucket++;
            }
            buckets[current_bucket].vertices.push_back(v);
            buckets[current_bucket].edgeCount += degree;
            edgeCounts[current_bucket] += degree;
        }
    }

    // Broadcast bucket assignments
    for (int i = 0; i < nparts; ++i) {
        Vertex size = buckets[i].vertices.size();
        MPI_Bcast(&size, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
        buckets[i].vertices.resize(size);
        MPI_Bcast(buckets[i].vertices.data(), size, MPI_INT64_T, 0, MPI_COMM_WORLD);
        MPI_Bcast(&buckets[i].edgeCount, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    }

    return buckets;
}

// Output graph structure and analysis
void output_analysis(const Graph& g, const GraphStats& stats, const PreprocessedGraph& pg, const std::vector<Bucket>& buckets, int rank) {
    if (rank == 0) {
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
                int bucket = std::min(static_cast<int>(vr * 10 / g.nvtxs), 9);
                rank_buckets[bucket]++;
            }
        }
        cout << "  Rank Distribution (10 buckets):\n";
        for (int i = 0; i < 10; ++i) {
            cout << "    Bucket " << i << " (" << (i * g.nvtxs / 10) << "-" << ((i + 1) * g.nvtxs / 10 - 1) << "): " << rank_buckets[i] << " vertices\n";
        }

        // Bucket partitioning
        cout << "  Bucket Partitioning:\n";
        for (size_t i = 0; i < buckets.size(); ++i) {
            cout << "    Bucket " << i << ": " << buckets[i].vertices.size() << " vertices, " << buckets[i].edgeCount / 2 << " edges\n";
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 3) {
        if (rank == 0) {
            cerr << "This program requires exactly 3 or more MPI processes.\n";
        }
        MPI_Finalize();
        return 1;
    }

   char hostname[256];
   gethostname(hostname, sizeof(hostname));
   cout << "Rank " << rank << " running on " << hostname << "\n" << std::flush;

   // Set OpenMP threads dynamically
   // Default for 6-core machine
   int num_threads = std::getenv("OMP_NUM_THREADS") ? atoi(std::getenv("OMP_NUM_THREADS")) : 2;
   if (argc > 1) {
    num_threads = std::atoi(argv[1]);
    }
    if (num_threads < 1) num_threads = 1;
    omp_set_num_threads(num_threads);
    //omp_set_proc_bind(OMP_PROC_BIND_TRUE);
    if (rank == 0) {
        std::cout << "Using " << num_threads << " OpenMP threads per node\n";
    }

    // Load graph
    string filename = "movies.edges";
    Graph g = extract(filename, rank);

    // Analyze graph
    GraphStats stats = analyzeGraph(g, rank);

    // Preprocess graph
    PreprocessedGraph pg = Preprocessing(g, rank);

    // Partition graph into buckets
    vector<Bucket> buckets = bucket_partition(g, pg, size, rank);

    // Output analysis
    output_analysis(g, stats, pg, buckets, rank);

    MPI_Finalize();
    return 0;
}
