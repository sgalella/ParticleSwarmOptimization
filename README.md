# Particle Swarm Optimization
Implementation of the particle swarm optimization (PSO) algorithm proposed in [_Particle Swarm Optimization_](https://ieeexplore.ieee.org/document/488968) to find the global minimum in a landscape.

> _"PSO solves a problem by having a population of candidates of solutions, called particles, and moving these particles around in the search-space according to the particle's position and velocity"_ — From [Wikipedia](https://en.wikipedia.org/wiki/Particle_swarm_optimization)

<p align="center">
    <img width="512" height="304" src="images/pso.gif">
</p>



## Installation

To install the dependencies, run the following command:

```bash
pip install -r requirements.txt
```

If using Conda, you can also create an environment with the requirements:

```bash
conda env create -f environment.yml
```

By default the environment name is `particle-swarm`. To activate it run:

```bash
conda activate particle-swarm
```



## Usage

Run the algorithm from the command line with:

```python
python particle_swarm_optimization.py
```