import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoLowRankMultivariateNormal
from pyro.optim import ClippedAdam


def fit_svi(
    model,
    dataset,
    *,
    lr=0.01,
    num_steps=201,
    log_every=100,
    plot=False,
):
    pyro.clear_param_store()
    guide = AutoLowRankMultivariateNormal(model)
    optim = ClippedAdam({"lr": lr, "lrd": 0.1 ** (1 / num_steps)})
    svi = SVI(model, guide, optim, Trace_ELBO())
    losses = []
    for step in range(num_steps):
        loss = svi.step(**dataset) / len(dataset["experiment_response"])
        losses.append(loss)
        if log_every and step % log_every == 0:
            print(f"svi step {step} loss = {loss:0.6g}")

    if plot:
        import matplotlib.pyplot as plt

        plt.plot(losses)
        plt.xlabel("SVI step")
        plt.ylabel("loss")

    # TODO return samples rather than a guide, to be consistent with SVI
    return guide
