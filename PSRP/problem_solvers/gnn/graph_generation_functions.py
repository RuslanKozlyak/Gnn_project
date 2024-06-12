import random
from PSRP.data_generators.GraphGenerator import GraphGenerator


def create_wheel_noised(num_nodes):
  graph_generator = GraphGenerator()

  g = graph_generator.build_graph( num_nodes, graph_type="wheel", layout="kamada-kawai")
  g = graph_generator.alter_graph( g, shift=(1.3, 1.3), scale=(100.0, 100.0), rotation_angle=0, convert_to_int=True,
                                  noised_nodes_part=0.8, node_noise_strength=0.2, random_seed=random.randint(0,100000))
  g = graph_generator.init_edge_weights( g, round_to = 1 )
  weight_matrix = graph_generator.build_edge_weight_matrix(g, fill_empty_policy ="shortest", make_int=True)

  graph_data = {}
  graph_data["graph"] = g
  graph_data["weight_matrix"] = weight_matrix
  return graph_data


def create_tree_noised(num_nodes):

  graph_generator = GraphGenerator()

  g = graph_generator.build_graph( num_nodes, graph_type="tree", layout="spring", random_seed=random.randint(0,100000))
  g = graph_generator.alter_graph( g, shift=(1.0, 1.0), scale=(100.0, 100.0), rotation_angle=0, convert_to_int=True )
  g = graph_generator.alter_graph( g, shift=(1.0, 1.0), scale=(1.0, 1.0),
                                  noised_nodes_part=1.0, node_noise_strength=0.1, random_seed=random.randint(0,100000),
                                  rotation_angle=0, convert_to_int=True )
  g = graph_generator.alter_graph( g, shift=(1.0, 1.0), scale=(1.0, 1.0),
                                  noised_nodes_part=0.4, node_noise_strength=0.2, random_seed=random.randint(0,100000),
                                  rotation_angle=0, convert_to_int=True )

  composed_graph = graph_generator.init_edge_weights( g, round_to=1 )
  weight_matrix = graph_generator.build_edge_weight_matrix(composed_graph, fill_empty_policy="shortest")

  graph_data = {}
  graph_data["graph"] = composed_graph
  graph_data["weight_matrix"] = weight_matrix

  return graph_data

def create_grid_noised(num_nodes):
  graph_generator = GraphGenerator()

  g = graph_generator.build_graph( num_nodes, graph_type="cube", layout="kamada-kawai" )
  g = graph_generator.alter_graph( g, shift=(1.0, 1.0), scale=(100.0, 100.0), rotation_angle=0, convert_to_int=True )
  g = graph_generator.alter_graph( g, shift=(1.0, 1.0), scale=(1.0, 1.0),
                                  noised_nodes_part=1.0, node_noise_strength=0.05, random_seed=45,
                                  rotation_angle=0, convert_to_int=True )
  g = graph_generator.alter_graph( g, shift=(1.0, 1.0), scale=(1.0, 1.0),
                                  noised_nodes_part=0.2, node_noise_strength=0.2, random_seed=46,
                                  rotation_angle=0, convert_to_int=True )

  g = graph_generator.init_edge_weights( g, round_to = 3 )
  weight_matrix = graph_generator.build_edge_weight_matrix(g, fill_empty_policy="shortest")

  graph_data = {}
  graph_data["graph"] = g
  graph_data["weight_matrix"] = weight_matrix

  return graph_data