# import networkx as nx
# from karateclub import DeepWalk
#
# g = nx.newman_watts_strogatz_graph(100, 20, 0.05)
#
# model = DeepWalk()
# model.fit(g)
# embedding = model.get_embedding()
# print(embedding)
#
# import networkx as nx
# from karateclub.node_embedding.neighbourhood import Walklets
#
# g = nx.newman_watts_strogatz_graph(100, 20, 0.05)
#
# model = Walklets()
# model.fit(g)
# embedding = model.get_embedding()
# print(embedding)

import networkx as nx
from karateclub import LabelPropagation

graph = nx.newman_watts_strogatz_graph(100, 20, 0.05)

model = LabelPropagation()
model.fit(graph)
cluster_membership = model.get_memberships()