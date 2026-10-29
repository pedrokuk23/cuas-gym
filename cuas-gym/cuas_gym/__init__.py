"""CUAS-Gym: a training and evaluation environment for language models on multi-sensor placement over terrain.

Each instance is a budgeted placement problem on real terrain. The physics module (cuas_gym.physics) computes, once per instance, the
probability of detection of every candidate configuration (site x sensor type) in every cell; the verifier then scores
any placement exactly and cheaply with the fusion rule P(>=1) = 1 - prod(1 - Pd).

    from cuas_gym import generate, Verifier
    inst = generate(seed=0, split="train")
    v = Verifier(inst)
    print(v.score([(3, "radar"), (7, "eo")]))
"""
from .instances import SPLITS, Instance, generate, load_instance, save_instance
from .verifier import Verifier
from .rewards import hybrid_reward, outcome_reward, process_reward
from .prompt import build_prompt, parse_answer

__all__ = ["SPLITS", "Instance", "generate", "load_instance", "save_instance", "Verifier",
           "hybrid_reward", "outcome_reward", "process_reward", "build_prompt", "parse_answer"]
