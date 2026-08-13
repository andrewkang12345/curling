import numpy as np

from world.search.vec_tree import VecTree, _Dec


def test_modeled_opponent_integrates_instead_of_minimizing():
    node = _Dec(np.zeros(24, np.float32), np.array([1.0, 0.0, 1.0], np.float32),
                h=3, depth=1, integrate_actions=True)
    node.set_pool(np.zeros((2, 4), np.float32), np.array([0.75, 0.25]), open_all=True)
    # Action 0 looks very good for the root and action 1 very bad.  A minimizer
    # would always choose action 1; the modeled node must follow its 3:1 prior.
    node.edge_n[:] = [3.0, 1.0]
    node.edge_sum[:] = [30.0, -10.0]
    tree = object.__new__(VecTree)
    assert tree._select(node) == 0
    node.edge_vl[0] = 2.0
    assert tree._select(node) == 1
