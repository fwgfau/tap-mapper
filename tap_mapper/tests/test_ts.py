import unittest
import numpy as np
import networkx as nx
from tap_mapper.ts_algos import approx_ts


class TestTS(unittest.TestCase):

    def test_connectivity(self):
        for k in range(10):
            with self.subTest(i=k):
                graph = nx.random_regular_graph(5, 30, seed=k)
                np.random.seed(k)
                initial_mapping = list(np.random.permutation(graph.number_of_nodes()))
                final_mapping = list(np.random.permutation(graph.number_of_nodes()))

                ml, sl = approx_ts(graph, initial_mapping, final_mapping)

                self.assertTrue(len(ml) == len(sl) + 1)

                for i, s in enumerate(sl):
                    self.assertTrue(np.all(s == np.nonzero(np.array(ml[i]) - np.array(ml[i + 1]))[0]))
                    self.assertTrue(list(s) in graph.edges())


if __name__ == "__main__":

    unittest.main()
