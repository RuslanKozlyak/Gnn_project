from pathlib import Path
import shutil
import numpy as np
from tqdm import tqdm
from PSRP.data_generators.DataBuilder_MPPSRP_Simple import DataBuilder_MPPSRP_Simple
import networkx as nx

from PSRP.utils import load, save

def get_graph_dict(synth_data, parameters_dict):

    data_builder = DataBuilder_MPPSRP_Simple(synth_data["weight_matrix"],
                                         distance_multiplier=1, travel_time_multiplier=60*60,
                                         planning_horizon=parameters_dict['planning_horizon'],
                                         safety_level=0.05, max_level=0.95,
                                         initial_inventory_level=0.5, tank_capacity=100,
                                         depot_service_time = 0*60, station_service_time=0*60,
                                         demand=10, products_count=parameters_dict['products_count'],
                                         k_vehicles=parameters_dict["k_vehicles"],
                                        compartments=[parameters_dict["products_count"] * [parameters_dict["compartment_capacity"]]],
                                         mean_vehicle_speed=60, vehicle_time_windows=[[9*60*60, 18*60*60]],
                                         noise_initial_inventory=0.0, noise_tank_capacity=0.0,
                                         noise_compartments=0.0, noise_demand=0.0,
                                         noise_vehicle_time_windows=0.0,
                                         noise_restrictions = 0.0,
                                         random_seed=45)

    graph_data = data_builder.build_data_model()

    # get node position
    graph = synth_data['graph']
    positions = nx.get_node_attributes(graph, 'pos').values()
    positions = np.asarray(list(positions))

    # get depots
    depots = np.array([0])

    # get demands and normalize by compartment_capacity
    plan_horizon_prods = []
    min_capacities = []
    max_capacities = []
    init_capacities = []
    for day_num in range(parameters_dict['planning_horizon']):

      #demands
      demand_idx = 5
      prods = []
      for i in range(parameters_dict['products_count']):
        products_type_mask = graph_data['station_data'][:,1] == i
        demands = graph_data['station_data'][products_type_mask][:, demand_idx + day_num]
        demands = np.insert(demands, 0, 0)
        demands = demands
        prods.append(demands)
      prods = np.stack(prods, axis=0).T
      plan_horizon_prods.append(prods)

    for i in range(parameters_dict['products_count']):
      products_type_mask = graph_data['station_data'][:,1] == i
      #min_capacities
      min_cap_idx = 2
      min_capacitie = graph_data['station_data'][products_type_mask][:, min_cap_idx]
      min_capacitie = np.insert(min_capacitie, 0, 0)
      min_capacitie = min_capacitie
      min_capacities.append(min_capacitie)

      #max_capacities
      max_cap_idx = 3
      max_capacitie = graph_data['station_data'][products_type_mask][:, max_cap_idx]
      max_capacitie = np.insert(max_capacitie, 0, 0)
      max_capacitie = max_capacitie
      max_capacities.append(max_capacitie)

      #init_capacities
      init_cap_idx = 4
      init_capacitie = graph_data['station_data'][products_type_mask][:, init_cap_idx]
      init_capacitie = np.insert(init_capacitie, 0, 0)
      init_capacitie = init_capacitie
      init_capacities.append(init_capacitie)


    max_capacities = np.stack(max_capacities, axis=0).T
    min_capacities = np.stack(min_capacities, axis=0).T
    init_capacities = np.stack(init_capacities, axis=0).T

    time_windows = graph_data['vehicle_time_windows']
    time_deltas = time_windows[:,1] - time_windows[:,0]
    time_deltas = time_deltas * parameters_dict["planning_horizon"]

    vehicle_compartments = []
    for i in range(parameters_dict['k_vehicles']):
      compartments = np.stack([graph_data['vehicle_compartments'][i] for _ in range(parameters_dict['num_nodes'])], axis=0)
      vehicle_compartments.append(compartments)
    vehicle_compartments = np.stack(vehicle_compartments, axis=0)

    min_capacities = min_capacities / max_capacities
    init_capacities = init_capacities / max_capacities
    vehicle_compartments = vehicle_compartments / max_capacities
    node_demand = np.array(plan_horizon_prods) / max_capacities
    load = np.ones((parameters_dict['num_nodes'], parameters_dict['products_count'])) * parameters_dict['compartment_capacity']
    load = load / max_capacities 
    max_capacities = max_capacities / max_capacities

    max_capacities = np.nan_to_num(max_capacities, 0)
    min_capacities = np.nan_to_num(min_capacities, 0)
    init_capacities = np.nan_to_num(init_capacities, 0)
    vehicle_compartments = np.nan_to_num(vehicle_compartments, 0)
    load = np.nan_to_num(load, 0)
    node_demand = np.nan_to_num(node_demand, 0)

    days_count = parameters_dict['planning_horizon']
    dummy_day = np.zeros((1, parameters_dict["num_nodes"], parameters_dict["products_count"]))
    daily_demands = np.vstack((node_demand, dummy_day))

    remaining_time = 18*60*60 - 9*60*60

    # get and normalize weight matrix
    weight_matrix = graph_data['travel_time_matrix']
    weight_matrix = np.divide(weight_matrix, remaining_time)
    service_times = graph_data['service_times'] / remaining_time

    model_for_nn = {
        'positions':positions,
        'travel_time_matrix':weight_matrix,
        'node_demand':daily_demands,
        'products_count':parameters_dict["products_count"],
        'vehicle_compartments':vehicle_compartments,
        'num_nodes':parameters_dict["num_nodes"],
        'depots':depots,
        'k_vehicles':parameters_dict["k_vehicles"],
        'vehicle_time_windows':time_deltas,
        'restriction_matrix':graph_data['restriction_matrix'],
        'service_times':service_times,
        'min_capacities':min_capacities,
        'max_capacities':max_capacities,
        'init_capacities':init_capacities,
        'load':load
    }

    return model_for_nn, graph_data

def generate_dataset(graph_creation_method,
                     parameters_dict,
                     dataset_name=None,
                     dataset_dir='/content/drive/MyDrive/GraphDataset',
                     del_previous_dir=False):

  if del_previous_dir:
    shutil.rmtree(dataset_dir, ignore_errors=True)

  if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

  num_nodes = parameters_dict['num_nodes']
  num_graphs = parameters_dict['num_graphs']

  if dataset_name == None:
    dataset_name = f'{graph_creation_method.__name__}_{num_nodes}'

  synth_data = graph_creation_method(num_nodes)
  model_for_nn, model_for_cpsat = get_graph_dict(synth_data=synth_data, parameters_dict=parameters_dict)

  for idx in tqdm(range(num_graphs)):
    save( model_for_nn, Path(dataset_dir, f"{dataset_name}_{idx}.pkl"))
  return model_for_cpsat

import os
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset):
    def __init__(self, dataset_dir,
                 generation_function,
                 parameters_dict):

      self.dataset_dir = dataset_dir
      self.parameters_dict = parameters_dict

      self.model_for_cpsat = generate_dataset(generation_function,
                 parameters_dict,
                 dataset_dir = dataset_dir,
                 del_previous_dir=True)

      self.graphs = os.listdir(dataset_dir)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        file = self.graphs[idx]
        graph = load(Path(self.dataset_dir, file))

        return graph['positions'],\
          graph['travel_time_matrix'], \
          graph['node_demand'],\
          graph['depots'],\
          graph['vehicle_time_windows'],\
          graph['restriction_matrix'],\
          graph['service_times'],\
          graph['min_capacities'],\
          graph['max_capacities'],\
          graph['init_capacities'],\
          graph['load']