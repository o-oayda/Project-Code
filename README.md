# Project Code: A Novel Test of the Cosmological Principle

## Summary of Code
Below is a copy of Table B.1 from the report:

| Script | Function |
| --- | --- |
`points-required.py` | Computes the evidences via `dynesty` for each hypothesis after generating a sample of $n$ quasars with intrinsic timescale $\tau$ and uncertainty $\Delta \tau_0$.|
| `bayes-factors.py` | Computes the Bayes factors for each experiment given evidences. Also responsible for producing the plots of $\ln B$ against $n$ in Section 3.1. |
| `funcs.py` | Houses core functions e.g. implementation of equations, sampling, etc. |
| `plotting.py` | Houses core plotting functions, responsible for producing Figure 2.4 and Figure 2.7. |
| `light-curve-MCMC.py` | Responsible for generating light curves via the DRW process and extracting $\tau$ and $\hat{\sigma}$ through `JAVELIN`. |
| `light-curve-distribution.py` | Responsible for inspecting the distribution of timescales and producing the $\tau$ distribution plots in Section 3.2. |
| `points-visualisation.py` | Responsible for producing Figure 2.3 |
| `locations-tested.py` | Responsible for producing Figure 3.1 |