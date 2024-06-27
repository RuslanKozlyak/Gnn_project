from collections import defaultdict
from json import load
import tempfile
from typing import Tuple
from matplotlib.pylab import f
import numpy as np
import torch


class IRPEnv_Custom:

    def __init__(
        self,
        batch,
        parameters_dict,
        seed: int = 69,
    ):
        self.num_nodes = parameters_dict['num_nodes']
        self.batch_size = parameters_dict['batch_size']
        self.products_count = parameters_dict['products_count']
        self.k_vehicles = parameters_dict['k_vehicles']
        self.max_trips = parameters_dict['max_trips']
        self.num_depots = parameters_dict['num_depots']
        self.compartment_capacity = parameters_dict['compartment_capacity']
        self.days_count = parameters_dict['planning_horizon']

        np.random.seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.positions, self.weight_matrixes, self.daily_demands,\
        self.depots, self.remaining_time,\
        self.restriction_matrix, self.service_times,\
        self.min_capacities, self.max_capacities,\
        self.init_capacities, self.vehicle_compartments, self.load = batch

        self.batch_size = self.positions.shape[0]

        if not isinstance(self.positions, np.ndarray):
          self.positions = self.positions.numpy()
          self.depots = self.depots.numpy()
          self.daily_demands = np.array([demand.numpy() for demand in self.daily_demands])
          self.weight_matrixes = self.weight_matrixes.numpy()
          self.service_times = self.service_times.numpy()
          self.min_capacities = self.min_capacities.numpy()
          self.max_capacities = self.max_capacities.numpy()
          self.init_capacities = self.init_capacities.numpy()
          self.restriction_matrix = self.restriction_matrix.numpy()
          self.load = self.load.numpy()
          
        self.depots = self.depots.astype(int)

        self.cur_day = np.zeros(shape=(self.batch_size)).astype(int)
        self.current_location = self.depots

        self.cur_remaining_time = np.ones((self.batch_size))

        self.vehicles = np.ones(shape=(self.batch_size)) * self.k_vehicles * self.max_trips
        self.demands = self.daily_demands[np.arange(len(self.daily_demands)), self.cur_day.astype(int)]

        self.actions_list = []
        self.percent_list = []
        self.avarage_stocks = defaultdict(float)
        self.dry_runs_dict = defaultdict(int)
        self.actions_daily = defaultdict(list)
        self.vehicle_utilisation = 0
        self.loss_dry_runs = np.zeros(self.batch_size)

        self.dist_to_depot =  self.weight_matrixes[np.arange(self.batch_size)[:, None], self.depots].squeeze()
        cur_to_any_time =  self.weight_matrixes[np.arange(self.batch_size)[:, None], self.current_location].squeeze()
        self.possible_action_time = cur_to_any_time + self.service_times + self.dist_to_depot
        self.temp_load = self.load.copy()
        
    def step(self, actions: np.ndarray, load_percent: np.ndarray) -> Tuple[np.ndarray, dict, bool]:
        
        traversed_edges = np.hstack([self.current_location, actions]).astype(int)

        # time spent on action
        action_time = self.get_distances(zip(self.current_location, actions))
        action_time = action_time.T.squeeze()
        
        # moved to new location
        self.current_location = np.array(actions)
        
        # time spent on service on new station
        selected_service_times = self.service_times[np.arange(len(self.service_times)),actions.T].squeeze()
        self.cur_remaining_time -= action_time + selected_service_times
        
        # calculating time to move to any station from cur location
        cur_to_any_time =  self.weight_matrixes[np.arange(self.batch_size)[:, None], self.current_location].squeeze()

        # calculation time for each action
        self.possible_action_time = cur_to_any_time + self.service_times + self.dist_to_depot

        # update load of each vehicle
        # load for 3 day further
        # days_count = 3
        # full_fill_up = self.max_capacities - self.init_capacities
        # selected_delivery = self.daily_demands[np.arange(self.batch_size), :days_count, actions.T, :].squeeze(0).sum(axis=1)
        # selected_temp_load = self.temp_load[np.arange(self.batch_size), actions.T,:].squeeze(0)
        # self.capacity_reduction = np.minimum(selected_delivery, selected_temp_load)

        # load by percent
        full_fill_up = self.max_capacities - self.init_capacities
        selected_delivery = full_fill_up[np.arange(self.batch_size), actions.T,:].squeeze(0)
        selected_temp_load = self.temp_load[np.arange(self.batch_size), actions.T,:].squeeze(0) 
        self.capacity_reduction = np.minimum(selected_delivery, selected_temp_load)

        self.init_capacities[np.arange(self.batch_size), actions.T] += self.capacity_reduction

        # cause on each station different normalization for load, changin load by percent
        percent = self.capacity_reduction / self.load[np.arange(self.batch_size), actions.T,:].squeeze(0)
        percent = np.expand_dims(percent, axis=1)
        percent_reduction = self.load * percent

        self.temp_load -= percent_reduction
        self.temp_load[self.temp_load <= 0.01] = 0

        vehicle_in_depot = np.where(actions == self.depots)[0]
        self.temp_load[vehicle_in_depot] = self.load[vehicle_in_depot]

        self.cur_remaining_time[vehicle_in_depot] = np.ones(shape=(self.batch_size))[vehicle_in_depot]

        self.vehicles[vehicle_in_depot] -= 1
     
        self.day_end = np.where(self.vehicles <= 0.0)[0]

        self.vehicles[self.day_end] = np.ones(shape=(self.batch_size))[self.day_end] * self.k_vehicles * self.max_trips

        self.demands = self.daily_demands[np.arange(len(self.daily_demands)), self.cur_day]
        self.init_capacities[self.day_end] -= self.demands[self.day_end]
        self.init_capacities[self.init_capacities < 0] = 0

        self.cur_day[self.day_end] += 1
        self.cur_day = np.clip(self.cur_day, 0, self.days_count)
        done = self.is_done()

        self.calc_step_kpis(actions, load_percent)

        return self.get_loss(traversed_edges), self.get_kpis(), done

    def get_loss(self, traversed_edges):
        return -self.get_distances(traversed_edges)

    def calc_step_kpis(self, actions, load_percent):
        self.actions_list.append(actions)
        self.percent_list.append(load_percent)

        dry_runs = np.einsum('ijk -> i', (self.init_capacities < self.min_capacities).astype(int))

        self.loss_dry_runs[self.day_end] += dry_runs[self.day_end]

        avarage_stock = np.zeros((self.batch_size,))
        avarage_stock = (np.einsum('ijk->',self.init_capacities[self.day_end]) / np.einsum('ijk->', self.max_capacities[self.day_end]))

        for day in self.cur_day :
          if len(self.day_end) != 0:
            self.avarage_stocks[day] += avarage_stock
            self.dry_runs_dict[day] += dry_runs.sum()
          self.actions_daily[day].append(actions)

        self.vehicle_utilisation += self.capacity_reduction.mean()

    def get_kpis(self):
        for key in self.avarage_stocks:
          self.avarage_stocks[key] = self.avarage_stocks[key] / self.batch_size

        kpi = {
            'actions_list':self.actions_list,
            'load_percents': self.percent_list,
            'actions_daily':self.actions_daily,
            'avarage_stocks':self.avarage_stocks,
            'dry_runs':self.dry_runs_dict,
            'vehicle_utilisation':self.vehicle_utilisation / self.batch_size,
        }

        return kpi

    def is_done(self):
        return np.all(self.cur_day >= self.days_count)

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        depots = np.zeros((self.batch_size, self.num_nodes))
        depots[np.arange(len(depots)), self.depots.T] = 1

        depots = torch.tensor(depots, dtype=torch.float, device=self.device)
        temp_vehicle_index =  (self.vehicles - 1) % self.max_trips
        temp_vehicle_index = temp_vehicle_index.astype(int)
        
        local_features = np.dstack(
            [
                self.positions,
                self.demands,
                self.init_capacities - self.min_capacities,
                self.temp_load,
                self.restriction_matrix[np.arange(self.batch_size), temp_vehicle_index, :]
            ]
        )
        
        global_features = [
            torch.tensor(self.vehicles, dtype=torch.float, device=self.device)[:,None,None],
            torch.tensor(self.cur_remaining_time, dtype=torch.float, device=self.device)[:,None,None],
            torch.tensor(self.service_times, dtype=torch.float, device=self.device)[:,None],
            # torch.tensor(self.cur_day / self.days_count, dtype=torch.float, device=self.device)[:,None,None]
            ]

        mask = self.generate_mask()

        local_features = torch.tensor(local_features, dtype=torch.float, device=self.device)
        mask = torch.tensor(mask, dtype=torch.float, device=self.device)

        return local_features, global_features, mask, depots

    def generate_mask(self):
        mask = np.zeros(shape=(self.batch_size, self.num_nodes), dtype=np.int32)
        
        # disallow staying in cur position
        mask[np.arange(self.batch_size)[:, None], self.current_location] = 1

        # disallow staying in nodes with max capacity
        filled_nodes = np.all(self.init_capacities == self.max_capacities, axis=2)
        mask[filled_nodes] |= 1
        
        # disallow visit node if not anought time to reach it
        not_enough_time = self.possible_action_time >= self.cur_remaining_time[:,None]
        mask[not_enough_time] |= 1

        # go to base if empty
        empty_loads = np.where(np.all(self.temp_load <= 0.0, axis=1))[0]
        mask[empty_loads] |= 1

        # go to base if all nodes filled
        all_filled = np.where(np.all(self.init_capacities == self.max_capacities, axis=1))[0]
        mask[all_filled] |= 1

        # disallow by restriction matrix
        temp_vehicle_index =  (self.vehicles - 1) % self.max_trips
        temp_vehicle_index = temp_vehicle_index.astype(int)
        cur_vehicle_restriction = self.restriction_matrix[np.arange(self.batch_size), temp_vehicle_index]
        mask |= cur_vehicle_restriction

        # force staying on a depot if the graph is solved.
        done_graphs = np.where(self.cur_day == self.days_count)[0]
        mask[done_graphs] = 1
        
        # always allow staing in depot
        mask[np.arange(len(mask)), self.depots.squeeze()] = 0

        return mask

    def get_distance(self, graph_idx: int, node_idx_1: int, node_idx_2: int) -> float:
        return self.weight_matrixes[graph_idx][node_idx_1, node_idx_2]

    def get_distances(self, paths) -> np.ndarray:
        return np.array(
            [
                self.get_distance(index, source, dest)
                for index, (source, dest) in enumerate(paths)
            ]
        )
        