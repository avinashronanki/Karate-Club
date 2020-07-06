from karateclub import GraphReader
from karateclub import LabelPropagation
from sklearn.metrics.cluster import normalized_mutual_info_score

reader = GraphReader("facebook")

graph = reader.get_graph()
target = reader.get_target()
# print(graph)
model = LabelPropagation()
model.fit(graph)
cluster_membership = model.get_memberships()

cluster_membership = [cluster_membership[node] for node in range(len(cluster_membership))]

nmi = normalized_mutual_info_score(target, cluster_membership)
print('NMI: {:.4f}'.format(nmi))