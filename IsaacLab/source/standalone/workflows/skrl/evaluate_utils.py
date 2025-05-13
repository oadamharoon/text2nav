from typing import Any, Mapping, Type, Union
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
import gymnasium as gym
import numpy as np
from skrl.models.torch import Model
import copy
from skrl import logger
import yaml
from skrl.resources.preprocessors.torch import RunningStandardScaler
import torch

def _component(name: str) -> Type:
        """Get skrl component (e.g.: agent, trainer, etc..) from string identifier

        :return: skrl component
        """
        component = None
        name = name.lower()
        # model
        if name == "gaussianmixin":
            from skrl.utils.model_instantiators.torch import gaussian_model as component
        elif name == "categoricalmixin":
            from skrl.utils.model_instantiators.torch import categorical_model as component
        elif name == "deterministicmixin":
            from skrl.utils.model_instantiators.torch import deterministic_model as component
        elif name == "multivariategaussianmixin":
            from skrl.utils.model_instantiators.torch import multivariate_gaussian_model as component
        elif name == "shared":
            from skrl.utils.model_instantiators.torch import shared_model as component
        # memory
        elif name == "randommemory":
            from skrl.memories.torch import RandomMemory as component
        # agent
        elif name in ["a2c", "a2c_default_config"]:
            from skrl.agents.torch.a2c import A2C, A2C_DEFAULT_CONFIG

            component = A2C_DEFAULT_CONFIG if "default_config" in name else A2C
        elif name in ["amp", "amp_default_config"]:
            from skrl.agents.torch.amp import AMP, AMP_DEFAULT_CONFIG

            component = AMP_DEFAULT_CONFIG if "default_config" in name else AMP
        elif name in ["cem", "cem_default_config"]:
            from skrl.agents.torch.cem import CEM, CEM_DEFAULT_CONFIG

            component = CEM_DEFAULT_CONFIG if "default_config" in name else CEM
        elif name in ["ddpg", "ddpg_default_config"]:
            from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG

            component = DDPG_DEFAULT_CONFIG if "default_config" in name else DDPG
        elif name in ["ddqn", "ddqn_default_config"]:
            from skrl.agents.torch.dqn import DDQN, DDQN_DEFAULT_CONFIG

            component = DDQN_DEFAULT_CONFIG if "default_config" in name else DDQN
        elif name in ["dqn", "dqn_default_config"]:
            from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG

            component = DQN_DEFAULT_CONFIG if "default_config" in name else DQN
        elif name in ["ppo", "ppo_default_config"]:
            from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG

            component = PPO_DEFAULT_CONFIG if "default_config" in name else PPO
        elif name in ["rpo", "rpo_default_config"]:
            from skrl.agents.torch.rpo import RPO, RPO_DEFAULT_CONFIG

            component = RPO_DEFAULT_CONFIG if "default_config" in name else RPO
        elif name in ["sac", "sac_default_config"]:
            from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG

            component = SAC_DEFAULT_CONFIG if "default_config" in name else SAC
        elif name in ["td3", "td3_default_config"]:
            from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG

            component = TD3_DEFAULT_CONFIG if "default_config" in name else TD3
        elif name in ["trpo", "trpo_default_config"]:
            from skrl.agents.torch.trpo import TRPO, TRPO_DEFAULT_CONFIG

            component = TRPO_DEFAULT_CONFIG if "default_config" in name else TRPO
        # multi-agent
        elif name in ["ippo", "ippo_default_config"]:
            from skrl.multi_agents.torch.ippo import IPPO, IPPO_DEFAULT_CONFIG

            component = IPPO_DEFAULT_CONFIG if "default_config" in name else IPPO
        elif name in ["mappo", "mappo_default_config"]:
            from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG

            component = MAPPO_DEFAULT_CONFIG if "default_config" in name else MAPPO
        # trainer
        elif name == "sequentialtrainer":
            from skrl.trainers.torch import SequentialTrainer as component

        if component is None:
            raise ValueError(f"Unknown component '{name}' in runner cfg")
        return component

def _process_cfg(cfg: dict) -> dict:
        """Convert simple types to skrl classes/components

        :param cfg: A configuration dictionary

        :return: Updated dictionary
        """
        _direct_eval = [
            "learning_rate_scheduler",
            "shared_state_preprocessor",
            "state_preprocessor",
            "value_preprocessor",
            "amp_state_preprocessor",
            "noise",
            "smooth_regularization_noise",
        ]

        def reward_shaper_function(scale):
            def reward_shaper(rewards, *args, **kwargs):
                return rewards * scale

            return reward_shaper

        def update_dict(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    update_dict(value)
                else:
                    if key in _direct_eval:
                        if isinstance(value, str):
                            d[key] = eval(value)
                    elif key.endswith("_kwargs"):
                        d[key] = value if value is not None else {}
                    elif key in ["rewards_shaper_scale"]:
                        d["rewards_shaper"] = reward_shaper_function(value)
            return d

        return update_dict(copy.deepcopy(cfg))


def generate_models(cfg: Mapping[str, Any], observation_space: gym.spaces.Space, action_space: gym.spaces.Space
    ) -> Mapping[str, Mapping[str, Model]]:
        """Generate model instances according to the environment specification and the given config

        :param env: Wrapped environment
        :param cfg: A configuration dictionary

        :return: Model instances
        """
        device = "cuda:0"
        possible_agents = ["agent"]
        observation_spaces = {"agent": observation_space}
        action_spaces = {"agent": action_space}

        # instantiate models
        models = {}
        for agent_id in possible_agents:
            _cfg = copy.deepcopy(cfg)
            models[agent_id] = {}
            models_cfg = _cfg.get("models")
            if not models_cfg:
                raise ValueError("No 'models' are defined in cfg")
            # get separate (non-shared) configuration and remove 'separate' key
            try:
                separate = models_cfg["separate"]
                del models_cfg["separate"]
            except KeyError:
                separate = True
                logger.warning("No 'separate' field defined in 'models' cfg. Defining it as True by default")
            # non-shared models
            if separate:
                for role in models_cfg:
                    # get instantiator function and remove 'class' key
                    model_class = models_cfg[role].get("class")
                    if not model_class:
                        raise ValueError(f"No 'class' field defined in 'models:{role}' cfg")
                    del models_cfg[role]["class"]
                    model_class = _component(model_class)
                    # get specific spaces according to agent/model cfg
                    observation_space = observation_spaces[agent_id]
                    # print model source
                    source = model_class(
                        observation_space=observation_space,
                        action_space=action_spaces[agent_id],
                        device=device,
                        **_process_cfg(models_cfg[role]),
                        return_source=True,
                    )
                    print("==================================================")
                    print(f"Model (role): {role}")
                    print("==================================================\n")
                    print(source)
                    print("--------------------------------------------------")
                    # instantiate model
                    models[agent_id][role] = model_class(
                        observation_space=observation_space,
                        action_space=action_spaces[agent_id],
                        device=device,
                        **_process_cfg(models_cfg[role]),
                    )
            # shared models
            else:
                roles = list(models_cfg.keys())
                if len(roles) != 2:
                    raise ValueError(
                        "Runner currently only supports shared models, made up of exactly two models. "
                        "Set 'separate' field to True to create non-shared models for the given cfg"
                    )
                # get shared model structure and parameters
                structure = []
                parameters = []
                for role in roles:
                    # get instantiator function and remove 'class' key
                    model_structure = models_cfg[role].get("class")
                    if not model_structure:
                        raise ValueError(f"No 'class' field defined in 'models:{role}' cfg")
                    del models_cfg[role]["class"]
                    structure.append(model_structure)
                    parameters.append(_process_cfg(models_cfg[role]))
                model_class = _component("Shared")
                # print model source
                source = model_class(
                    observation_space=observation_spaces[agent_id],
                    action_space=action_spaces[agent_id],
                    device=device,
                    structure=structure,
                    roles=roles,
                    parameters=parameters,
                    return_source=True,
                )
                print("==================================================")
                print(f"Shared model (roles): {roles}")
                print("==================================================\n")
                # print(source)
                print("--------------------------------------------------")
                # instantiate model
                models[agent_id][roles[0]] = model_class(
                    observation_space=observation_spaces[agent_id],
                    action_space=action_spaces[agent_id],
                    device=device,
                    structure=structure,
                    roles=roles,
                    parameters=parameters,
                )
                models[agent_id][roles[1]] = models[agent_id][roles[0]]

        # initialize lazy modules' parameters
        for agent_id in possible_agents:
            for role, model in models[agent_id].items():
                model.init_state_dict(role)

        return models