import matplotlib
import numpy as np
import pyro

matplotlib.use("Agg")


def pytest_runtest_setup(item):
    np.random.seed(20220324)
    pyro.set_rng_seed(20220324)
    pyro.enable_validation(True)
    pyro.clear_param_store()
