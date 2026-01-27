# Automated framework for the design of photonic circuit to violate Bell inequalities using homodyne measurements

Source code used in [2410.19670](https://arxiv.org/abs/2410.19670).

# Structure of the repo

## `notebook`

The `noise_analysis.jl` Pluto notebook contains all the code to produce and plot the data relevant for the analysis of the circuit depicted in Fig.1 of [2410.19670](https://arxiv.org/abs/2410.19670).
`cd` in the notebook directory and run it using

```
julia --project=. -e "import Pluto; Pluto.run(notebook="noise_analysis.jl")
```

## `src`

Contains the source code used to automatically search for photonic circuit that violates Bell inequalities using homodyne measurements.

`measures_homodyne.jl` contains functions to compute the CHSH score from a given circuit. These are implementations of equations detailed in Appendix A.

### `Agent`

`Agent.jl` contains the PPO agent while `RandomPolicy.jl` provides a random search.

### `Environment`

Each file provides the environment matching a strategy:  
    - `Custom_Env_with_allgates.jl` : Strategy 1  
    - `Custom_Env_with_initialSMS.jl` : Strategy 2 / 3  
    - `Custom_Env_with_initialTMS.jl` : Strategy 4 / 5  
    - `Env.jl` : flexible environment  
