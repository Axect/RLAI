# For type hinting and abstract classes
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Generic,
    TypeVar,
    Sequence,
    Any,
    Callable,
    Iterable,
)

# For statistics
import random
import statistics
import numpy as np

#  ┌──────────────────────────────────────────────────────────┐
#   Distribution
#  └──────────────────────────────────────────────────────────┘
A = TypeVar("A")

class Distribution(ABC, Generic[A]):
    @abstractmethod
    def sample(self) -> A:
        pass

    def sample_n(self, n: int) -> Sequence[A] | np.ndarray[Any, np.dtype[np.float64]]:
        return [self.sample() for _ in range(n)]


def expected_value(d: Distribution[A], f: Callable[[A], float], n: int = 100) -> float:
    return statistics.mean(f(d.sample()) for _ in range(n))


# Die roll
@dataclass(frozen=True)
class Die(Distribution[int]):
    sides: int

    def sample(self) -> int:
        return random.randint(1, self.sides)


# Gaussian
@dataclass
class Gaussian(Distribution[float]):
    mu: float
    sigma: float

    def sample(self) -> float:
        return np.random.normal(self.mu, self.sigma)

    def sample_n(self, n: int) -> np.ndarray[Any, np.dtype[np.float64]]:
        return np.random.normal(self.mu, self.sigma, n)


#  ┌──────────────────────────────────────────────────────────┐
#   State
#  └──────────────────────────────────────────────────────────┘
S = TypeVar("S")
X = TypeVar("X")

class State(ABC, Generic[S]):
    state: S

    def on_non_terminal(self, f: Callable[[NonTerminal[S]], X], default: X) -> X:
        if isinstance(self, NonTerminal):
            return f(self)
        else:
            return default


@dataclass(frozen=True)
class Terminal(State[S]):
    state: S


@dataclass(frozen=True)
class NonTerminal(State[S]):
    state: S


#  ┌──────────────────────────────────────────────────────────┐
#   Markov Decision Process
#  └──────────────────────────────────────────────────────────┘
class MarkovProcess(ABC, Generic[S]):
    @abstractmethod
    def transition(self, state: NonTerminal[S]) -> Distribution[State[S]]:
        pass

    def simulate(
        self, init_state_dist: Distribution[NonTerminal[S]]
    ) -> Iterable[State[S]]:
        state: State[S] = init_state_dist.sample()
        yield state

        while isinstance(state, NonTerminal):
            state = self.transition(state).sample()
            yield state
