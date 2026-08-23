"""Tests for evolutionary / swarm optimizers and neuroevolution."""

import numpy as np
import pytest

from decisionrl.envs import CartPole, PointMass
from decisionrl.evolution import (
    ARS,
    CEM,
    CMAES,
    PSO,
    AntColonyTSP,
    ArtificialBeeColony,
    BatAlgorithm,
    DifferentialEvolution,
    FireflyAlgorithm,
    GeneticAlgorithm,
    GreyWolfOptimizer,
    NeuroevolutionAgent,
    OpenAIES,
    SimulatedAnnealing,
    distance_matrix,
    minimize,
    random_cities,
)
from decisionrl.evolution.functions import ackley, rastrigin, rosenbrock, sphere
from decisionrl.training import evaluate_policy

CONTINUOUS_OPTIMIZERS = [
    CEM, CMAES, DifferentialEvolution, GeneticAlgorithm, OpenAIES, ARS,
    SimulatedAnnealing, PSO, FireflyAlgorithm, ArtificialBeeColony,
    GreyWolfOptimizer, BatAlgorithm,
]


def test_benchmark_functions_have_known_optima():
    assert sphere(np.zeros(5)) == pytest.approx(0.0)
    assert rastrigin(np.zeros(5)) == pytest.approx(0.0)
    assert ackley(np.zeros(5)) == pytest.approx(0.0, abs=1e-10)
    assert rosenbrock(np.ones(5)) == pytest.approx(0.0)


@pytest.mark.parametrize("optimizer_cls", CONTINUOUS_OPTIMIZERS)
def test_optimizer_minimizes_sphere(optimizer_cls):
    dim, bounds = 5, (-5.12, 5.12)
    kwargs = {"iters": 200} if optimizer_cls is GreyWolfOptimizer else {}
    opt = optimizer_cls(dim, bounds=bounds, seed=0, **kwargs)
    _, best_f, history = minimize(sphere, opt, 200)
    assert best_f < 1.0, f"{optimizer_cls.__name__}: {best_f}"
    assert history[-1] <= history[0]  # best-so-far is monotone non-increasing


def test_batched_fitness_matches_per_row():
    # Vectorized (batched) fitness must give identical results to per-row eval.
    from decisionrl.evolution.functions import sphere

    x1, f1, h1 = minimize(sphere, CEM(6, bounds=(-5.12, 5.12), seed=0), 50, batched=False)
    x2, f2, h2 = minimize(sphere, CEM(6, bounds=(-5.12, 5.12), seed=0), 50, batched=True)
    np.testing.assert_allclose(h1, h2)
    np.testing.assert_allclose(x1, x2)
    assert f1 == f2


def test_cem_and_cmaes_solve_rastrigin_well():
    for cls in (CEM, CMAES):
        _, best_f, _ = minimize(rastrigin, cls(6, bounds=(-5.12, 5.12), seed=0), 300)
        assert best_f < 10.0


def test_ant_colony_improves_tsp_tour():
    cities = random_cities(12, seed=1)
    d = distance_matrix(cities)
    aco = AntColonyTSP(d, seed=0)
    tour, length, history = aco.solve(80)
    naive = float(d[np.arange(12), np.roll(np.arange(12), -1)].sum())  # tour 0->1->...->0
    assert set(tour.tolist()) == set(range(12))  # a valid permutation
    assert length < naive
    assert history[-1] <= history[0]


def test_neuroevolution_predicts_and_round_trips(tmp_path, quiet_logger):
    agent = NeuroevolutionAgent(CartPole(), optimizer="cem", hidden_sizes=(8,), popsize=16,
                                seed=0, logger=quiet_logger)
    agent.learn(3_000)
    obs, _ = CartPole().reset(seed=0)
    assert CartPole().action_space.contains(int(agent.predict(obs)))

    path = str(tmp_path / "evo.npz")
    agent.save(path)
    loaded = NeuroevolutionAgent.load(path, env=CartPole())
    for s in range(10):
        o, _ = CartPole().reset(seed=s)
        assert agent.predict(o) == loaded.predict(o)


def test_neuroevolution_continuous_within_bounds(quiet_logger):
    agent = NeuroevolutionAgent(PointMass(), optimizer="pso", hidden_sizes=(8,), popsize=16,
                                seed=0, logger=quiet_logger)
    agent.learn(2_000)
    obs, _ = PointMass().reset(seed=0)
    action = np.asarray(agent.predict(obs))
    assert action.shape == (2,)
    assert np.all(action >= PointMass().action_space.low - 1e-6)
    assert np.all(action <= PointMass().action_space.high + 1e-6)


def _random_policy_return(n_episodes: int = 10, seed: int = 0) -> float:
    """Mean return of a uniformly random policy - the floor any learner has to clear."""
    rng = np.random.default_rng(seed)
    returns = []
    for ep in range(n_episodes):
        env = CartPole()
        env.reset(seed=1_000 + ep)
        done, total = False, 0.0
        while not done:
            _, reward, terminated, truncated, _ = env.step(int(rng.integers(env.action_space.n)))
            total += reward
            done = terminated or truncated
        returns.append(total)
    return float(np.mean(returns))


def test_neuroevolution_is_reproducible_given_a_seed(quiet_logger):
    """Same seed, same policy; different seed, different policy.

    The seed handed to the agent reaches the optimizer's search directly, but the rollout
    start states are the fitness signal itself and those come from the environment. Until
    `learn` seeded the environment too, an identical seed still produced a different policy
    on every run - which is what made the CartPole test below fail at random.
    """
    def train(seed):
        agent = NeuroevolutionAgent(CartPole(), optimizer="cem", hidden_sizes=(8,), popsize=12,
                                    seed=seed, logger=quiet_logger)
        return agent.learn(3_000).params.copy()

    assert np.array_equal(train(0), train(0))
    assert not np.array_equal(train(0), train(1))


@pytest.mark.slow
def test_neuroevolution_cem_beats_random_on_cartpole(quiet_logger):
    """CEM must learn a policy far better than random - not "solve" CartPole.

    At this budget the method is high-variance: across seeds 0-3 it returns roughly
    283 / 105 / 241 / 97 against a 500-step ceiling. A single-seed threshold near that
    ceiling is a threshold on luck, and asserting one is how this test came to fail on
    unrelated pull requests. The median of three seeds, measured against the random-policy
    floor rather than against a magic number, is the claim that actually holds.
    """
    returns = []
    for seed in (0, 1, 2):
        agent = NeuroevolutionAgent(CartPole(), optimizer="cem", hidden_sizes=(16,), popsize=24,
                                    seed=seed, logger=quiet_logger)
        agent.learn(60_000)
        mean_return, _ = evaluate_policy(agent, CartPole(), n_episodes=10, seed=100)
        returns.append(mean_return)

    assert float(np.median(returns)) > 4.0 * _random_policy_return()
