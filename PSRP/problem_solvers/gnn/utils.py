from matplotlib import pyplot as plt
import networkx as nx

import matplotlib.colors as mcolors

def draw_paths(data_nn, routes):
  G = nx.Graph()
  edges = []
  for r in routes:
      route_edges = [(r[n],r[n+1]) for n in range(len(r)-1)]
      G.add_nodes_from(r)
      G.add_edges_from(route_edges)
      edges.append(route_edges)

  print("Graph has %d nodes with %d edges" %(G.number_of_nodes(),
  G.number_of_edges()))

  pos = data_nn[0]
  nx.draw_networkx_nodes(G,pos=pos)
  nx.draw_networkx_labels(G,pos=pos)
  colors = list(mcolors.TABLEAU_COLORS.keys())
  for ctr, edgelist in enumerate(edges):
      nx.draw_networkx_edges(G,pos=pos,edgelist=edgelist,edge_color = colors[ctr % len(colors)], width=5)
  plt.savefig('this.png')
  
  
def divide_to_paths_nn(actions):

  particular_value = 0
  result = []
  temp_list = []
  for i in actions:
      if i == particular_value:
          temp_list.append(i)
          result.append(temp_list)
          temp_list.insert(0,0)
          temp_list = []
      else:
          temp_list.append(i)
  result.append(temp_list)
  result.pop(0)
  result.pop(-1)
  return result

def divide_to_path_cpsat(routes_schedule):
  result_paths = []
  for car in routes_schedule[0]:
    for e in car:
      result_paths.append(e[0])
  return result_paths