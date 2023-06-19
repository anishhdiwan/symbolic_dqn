# Lunar Lander w/ GP

Evolving a lunar lander with differentiable genetic programming as part of CS4205 Evolutionary Algorithms at TU Delft

## Installation
Since this project requires older versions of some packages, it is recommended to install them in a virtual environment. You could either create a native python venv using the `requirement.txt` file in this repo (untested) or create a conda environment with the provided `environment.yml` file (tested). To do this, follow these steps.

1. Assuming you have conda installed. Download the environment.yml file 
2. The older versions of some packages are not available on the common conda channels. Moreover, different systems/hardware usually need different torch variants. These will be installed via pip in the next step. For now, run `conda env create -f environment.yml`
3. Activate the new env using `conda activate EA_env` and install the pending packages
4. `pip install pygame=2.1.0` | `pip install pyglet==1.5.21`
5. Find out the correct torch version for your system and GPU variant (cuda version). Install it from the [Torch Website](https://pytorch.org/)
6. Install git via `conda install -c anaconda git` and clone this repo. Enable the python kernel for this environment `python -m ipykernel install --user --name=python3` and start the notebook via `jupyter notebook` 
7. You can now start playing around with our solution

## Model Training & Inference

The directory `symbolic_dqn` holds all scripts relevant to neural guided population initialisation using DQN. The two scripts of interest are `main.py` and `inference.py`. Run `main.py` to train the model to learn a policy that can generate symbolic regression multitrees. Similarly run `inference.py` to use the saved policy to generate a population of multitrees. The config file `GP_symbolic_DQN_config.ini` within the `symbolic_dqn` directory holds the hyperparameters relevant to training, inference, and subsequent integration with the GP code within `solution.ipynb`. The script `actions.py` holds the operator instances. The operator set can be modified by modifying the `node_indices`, `node_instances`, and `node_vectors` dictionaries within `actions.py`.

We have already set up some trained models in this repository. The GP code within `solution.ipynb` loads the population generated from this policy to further evolve trees. Be sure to specify parameters such as pop_size or max_gens through the `GP_symbolic_DQN_config.ini` config file. 