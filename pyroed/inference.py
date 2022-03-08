from typing import Callable, Dict

import pyro
import torch
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.infer.autoguide import AutoLowRankMultivariateNormal
from pyro.infer.mcmc import MCMC, NUTS
from pyro.optim import ClippedAdam


def fit_svi(
    model: Callable,
    *,
    lr=0.01,
    num_steps=201,
    jit_compile=False,
    log_every=100,
    plot=False,
) -> Callable[[], Dict[str, torch.Tensor]]:
    pyro.clear_param_store()
    guide = AutoLowRankMultivariateNormal(model)
    optim = ClippedAdam({"lr": lr, "lrd": 0.1 ** (1 / num_steps)})
    elbo = (JitTrace_ELBO if jit_compile else Trace_ELBO)()
    svi = SVI(model, guide, optim, elbo)
    losses = []
    for step in range(num_steps):
        loss = svi.step()
        losses.append(loss)
        if log_every and step % log_every == 0:
            print(f"svi step {step} loss = {loss:0.6g}")

    if plot:
        import matplotlib.pyplot as plt

        plt.plot(losses)
        plt.xlabel("SVI step")
        plt.ylabel("loss")

    return guide


def fit_mcmc(
    model: Callable,
    *,
    num_samples=500,
    warmup_steps=500,
    num_chains=1,
    jit_compile=False,
) -> Callable[[], Dict[str, torch.Tensor]]:
    kernel = NUTS(model, jit_compile=jit_compile)
    mcmc = MCMC(
        kernel,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        num_chains=num_chains,
    )
    mcmc.run()
    samples = mcmc.get_samples()
    return Sampler(samples)


class Sampler:
    def __init__(self, samples: Dict[str, torch.Tensor]):
        self.samples = samples
        self.num_samples = len(next(iter(samples.values())))

    def __call__(self) -> Dict[str, torch.Tensor]:
        i = torch.randint(0, self.num_samples, ())
        return {k: v[i] for k, v in self.samples.items()}
