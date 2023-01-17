from os import listdir
from os.path import isfile, join
import pandas as pd
from data_collection.reddit_user_dataset import RedditUserDataset
import abc
import pickle as pkl
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.centrality import betweenness_centrality
from utils.file_sort import path_sort

############################################## Graph Measures ##########################################################

class PartitionedGraphMeasure(abc.ABC):
    @abc.abstractmethod
    def get_measurement_map(self, nx_graph, partition, num_doc_list):
        pass

class PartitionedConductanceMeasure(PartitionedGraphMeasure):
    def get_measurement_map(self, nx_graph, partition, num_doc_list):
        outbound_edges = {}
        intra_edges = {}

        for edge in nx_graph.edges():
            if partition[edge[0]] == partition[edge[1]]:
                if partition[edge[0]] in intra_edges.keys():
                    intra_edges[partition[edge[0]]] += 1
                else:
                    intra_edges[partition[edge[0]]] = 1
            else:
                if partition[edge[0]] in outbound_edges.keys():
                    outbound_edges[partition[edge[0]]] += 1
                else:
                    outbound_edges[partition[edge[0]]] = 1

        res_map = {}
        for community, int_edges in intra_edges.items():
            res_map[community] = outbound_edges[community]/(2*int_edges+outbound_edges[community])
        return res_map

class PartitionedDegreeMeasure(PartitionedGraphMeasure):
    def get_measurement_map(self, nx_graph, partition, num_doc_list):
        degree_sums = {}
        node_amounts = {}
        for node, community in partition.items():
            if num_doc_list[node] == 0:
                continue
            if community not in degree_sums.keys():
                degree_sums[community] = nx_graph.degree[node]
                node_amounts[community] = 1
            else:
                degree_sums[community] += nx_graph.degree[node]
                node_amounts[community] += 1

        res_map = {}

        average = sum(degree_sums.values()) / sum(node_amounts.values())
        for community, d_sum in degree_sums.items():
            res_map[community] = (d_sum / node_amounts[community]) - average

        return res_map

class PartitionedNormalizedDegreeMeasure():
    def get_measurement_map(self, nx_graph, partition, num_doc_list):
        degree_sums = {}
        node_amounts = {}
        for node, community in partition.items():
            if num_doc_list[node] == 0:
                continue
            if community not in degree_sums.keys():
                degree_sums[community] = nx_graph.degree[node]/num_doc_list[node]
                node_amounts[community] = 1
            else:
                degree_sums[community] += nx_graph.degree[node]/num_doc_list[node]
                node_amounts[community] += 1

        res_map = {}

        average = sum(degree_sums.values()) / sum(node_amounts.values())
        for community, d_sum in degree_sums.items():
            res_map[community] = (d_sum / node_amounts[community])

        return res_map

class PartitionedSeparabilityMeasure(PartitionedGraphMeasure):
    def get_measurement_map(self, nx_graph, partition, num_docs_list):
        outbound_edges = {}
        intra_edges = {}

        for edge in nx_graph.edges():
            if partition[edge[0]] == partition[edge[1]]:
                if partition[edge[0]] in intra_edges.keys():
                    intra_edges[partition[edge[0]]] += 1
                else:
                    intra_edges[partition[edge[0]]] = 1
            else:
                if partition[edge[0]] in outbound_edges.keys():
                    outbound_edges[partition[edge[0]]] += 1
                else:
                    outbound_edges[partition[edge[0]]] = 1

        res_map = {}
        for community, int_edges in intra_edges.items():
            res_map[community] = int_edges/outbound_edges[community]
        return res_map

class PartitionedDensityMeasure(PartitionedGraphMeasure):
    def get_measurement_map(self, nx_graph, partition, num_docs_list):
        intra_edges = {}
        comm_nodes = {}

        for node, community in partition.items():
            if num_docs_list[node] == 0:
                continue
            if partition[node] in comm_nodes.keys():
                comm_nodes[partition[node]] += 1
            else:
                comm_nodes[partition[node]] = 1

        for edge in nx_graph.edges():
            if partition[edge[0]] == partition[edge[1]]:
                if partition[edge[0]] in intra_edges.keys():
                    intra_edges[partition[edge[0]]] += 1
                else:
                    intra_edges[partition[edge[0]]] = 1

        res_map = {}
        for community, int_edges in intra_edges.items():
            res_map[community] = int_edges/(comm_nodes[community] * (comm_nodes[community]-1) * 0.5)
        return res_map

class PartitionedCentralityMeasure(PartitionedGraphMeasure):
    def get_measurement_map(self, nx_graph, partition, num_docs_list):
        print("New Graph")
        centrality_vals = betweenness_centrality(nx_graph, k=1000, normalized=False)

        val_sums = {}
        node_amounts = {}
        for node, community in partition.items():
            if num_docs_list[node] == 0:
                continue
            if community not in val_sums.keys():
                val_sums[community] = centrality_vals[node]/num_docs_list[node]
                node_amounts[community] = 1
            else:
                val_sums[community] += centrality_vals[node]/num_docs_list[node]
                node_amounts[community] += 1

        res_map = {}
        average = sum(val_sums.values()) / sum(node_amounts.values())
        for community, val_sum in val_sums.items():
            res_map[community] = (val_sum / node_amounts[community])

        print(res_map)
        return res_map

class PartitionedNumDocsMeasure():
    def get_measurement_map(self, nx_graph, partition, num_docs_list):
        val_sums = {}
        node_amounts = {}
        for node, community in partition.items():
            #if num_docs_list[node] == 0:
            #    continue
            if community not in val_sums.keys():
                val_sums[community] = num_docs_list[node]
                node_amounts[community] = 1
            else:
                val_sums[community] += num_docs_list[node]
                node_amounts[community] += 1

        res = {}
        print(val_sums)
        average = sum(val_sums.values()) / sum(node_amounts.values())
        print(node_amounts)
        for community, val_sum in val_sums.items():
            res[community] = (val_sums[community] / node_amounts[community])
        return res


################################################ Pipeline ##############################################################

graph_dir = "data/core_dataset/linguistic/cosine/avg/delta30/t_0.8p_1em_avg"
#graph_dir = "data/core_dataset/social/delta30/source"
base_dataset = RedditUserDataset.load_from_file("data/core_dataset/reddit_corpus_final_balanced.gzip", compression='gzip')
res_file = "results/figures/lingu_centrality_normalized.png"
GRAPH_TYPE = 'LINGUISTIC'

# Build ground truth
ground_truth = {}
for index, row in base_dataset.data_frame.iterrows():
    ground_truth[index] = row['fake_news_spreader']

graph_measure = PartitionedCentralityMeasure()

def load_graphs_from_dir(dir_path: str, compression='gzip'):
    files = path_sort([join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f))])
    print(files)
    if GRAPH_TYPE == 'LINGUISTIC':
        return [pd.read_pickle(path, compression=compression) for path in files]
    if GRAPH_TYPE == 'SOCIAL':
        return [RedditUserDataset.load_from_instance_file(path).data_frame for path in files if not 'source_graph_descriptor.data' in path]

def pandas_to_networkX(dataframe):
    graph = nx.Graph()

    for index, row in dataframe.iterrows():
        graph.add_node(index)
        for node, weight_dict in row['social_graph'].items():
            if node in dataframe['user_id']:
                # Use directed or not???
                graph.add_edge(index, node)
    return graph

def pandas_to_num_docs(dataframe):
    step_dict = {}
    for index, row in dataframe.iterrows():
        step_dict[index] = row['num_docs']
    return step_dict

graphs = load_graphs_from_dir(graph_dir)

nX_graphs = [pandas_to_networkX(graph) for graph in graphs]
num_doc_dicts = [pandas_to_num_docs(graph) for graph in graphs]
measurements = [graph_measure.get_measurement_map(graph, ground_truth, num_doc_dicts[index]) for index, graph in enumerate(nX_graphs)]

fig, ax1 = plt.subplots()

color = 'black'
ax1.set_xlabel('Timeframe')
ax1.set_ylabel('Appr. Centrality Normalized', color=color)
ax1.bar([val-0.15 for val in range(len(measurements))], [dp[0] for dp in measurements], 0.3, label="real_news")
ax1.bar([val+0.15 for val in range(len(measurements))], [dp[1] for dp in measurements], 0.3, label="fake_news")
ax1.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.legend()
plt.savefig(res_file)

