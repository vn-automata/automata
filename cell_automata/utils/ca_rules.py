import cellpylib as cpl
from abc import ABC, abstractmethod


class CARule(ABC):
    """Abstract class for application of cellular automata rules"""

    @abstractmethod
    def rule_function(self, n, c, t):
        pass


class ConwayRule(CARule):
    """Implementation of Conway's Game of Life:
    a cellular automaton where a cell is "born" if it has exactly three neighbors,
    and a cell "survives" if it has exactly two or three neighbors. Otherwise,
    the cell dies or remains dead."""

    def rule_function(self, n, c, t):
        sum_n = sum(n)
        return c and 2 <= sum_n <= 3 or sum_n == 3


class HighLifeRule(CARule):
    """Implementation of Game of Life HighLife:
    a variant of Conway's Game of Life that also gives birth to a cell if there are 6 neighbors.
    """

    def rule_function(self, n, c, t):
        sum_n = sum(n)
        return c and 2 <= sum_n <= 3 or sum_n == 6


class DayAndNightRule(CARule):
    """Implementation of Day & Night: a variant of Conway's Game of Life
    that also gives birth to a cell if there are 3, 6, 7, or 8 neighbors."""

    def rule_function(self, n, c, t):
        sum_n = sum(n)
        return sum_n in (3, 6, 7, 8) or c and sum_n in (4, 6, 7, 8)


class Rule30(CARule):
    """Implementation of a one-dimensional cellular automaton rule introduced by Stephen Wolfram,
    known for its chaotic behavior."""

    def rule_function(self, n, c, t):
        return cpl.nks_rule(n, 30)


class Rule110(CARule):
    """Implementation of Rule 110: It's another one-dimensional cellular automaton rule,
    introduced by Stephen Wolfram. It's known for being Turing complete."""

    def rule_function(self, n, c, t):
        return cpl.nks_rule(n, 110)


class FredkinRule(CARule):
    """Implementation of Fredkin's is a cellular automaton rule where a cell is "born" if it has exactly one neighbor,
    and a cell "survives" if it has exactly two neighbors. Otherwise, the cell dies or remains dead.
    """

    def rule_function(self, n, c, t):
        sum_n = sum(n)
        return sum_n == 1 or c and sum_n == 2


class BriansBrainRule(CARule):
    """Implementation of Brian's Brain: a three-state simulation.
    A cell is "born" if it was dead and has exactly two neighbors.
    A live cell dies in the next generation, and a dead cell remains dead."""

    def rule_function(self, n, c, t):
        sum_n = sum(n)
        if c == 0 and sum_n == 2:
            return 1
        elif c == 1:
            return 2
        elif c == 2:
            return 0


class SeedsRule(CARule):
    """Implementation of Seeds is a cellular automaton where a cell is "born" if it has exactly two neighbors,
    and a cell "dies" otherwise."""

    def rule_function(self, n, c, t):
        sum_n = sum(n)
        return sum_n == 2
