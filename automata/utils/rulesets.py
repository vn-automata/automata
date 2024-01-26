import cellpylib as cpl
import numpy as np
from abc import ABC, abstractmethod


class Rule(ABC):
    """Abstract class for application of cellular automata rules"""
    
    @abstractmethod
    def apply_rule(self, neighbourhood, c, t):
        pass
    
    def __call__(self, neighbourhood, c, t):
        return self.apply_rule(neighbourhood, c, t)


class ConwayRule(Rule):
    """Implementation of Conway's Game of Life:
    a cellular automaton where a cell is "born" if it has exactly three neighbors,
    and a cell "survives" if it has exactly two or three neighbors. Otherwise,
    the cell dies or remains dead."""
    
    def apply_rule(self, neighbourhood, c, t):
        # Implement the logic for Conway's Game of Life
        center_cell = neighbourhood[1][1]
        total = np.sum(neighbourhood)
        if center_cell == 1:
            if total - 1 < 2: # Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
                return 0
            if total - 1 == 2 or total - 1 == 3: # Any live cell with two or three live neighbours lives on to the next generation.
                return 1
            if total - 1 > 3: # Any live cell with more than three live neighbours dies, as if by overpopulation.
                return 0
        else:
            if total == 3: # Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
                return 1
            else:
                return 0
            
class HighLifeRule(Rule):
    """Implementation of Game of Life HighLife:
    a variant of Conway's Game of Life that also gives birth to a cell if there are 6 neighbors.
    """
    
    def apply_rule(self, neighbourhood, c, t):
        # Implement the logic for HighLife
        center_cell = neighbourhood[1][1]
        total = np.sum(neighbourhood)
        if center_cell == 1:
            if total - 1 < 2: # Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
                return 0
            if total - 1 == 2 or total - 1 == 3: # Any live cell with two or three live neighbours lives on to the next generation.
                return 1
            if total - 1 > 3: # Any live cell with more than three live neighbours dies, as if by overpopulation.
                return 0
        else:
            if total == 3 or total == 6: # Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
                return 1
            else:
                return 0
            
            
class DayAndNightRule(Rule):
    """Implementation of Day & Night: a variant of Conway's Game of Life
    that also gives birth to a cell if there are 3, 6, 7, or 8 neighbors."""
    
    def apply_rule(self, neighbourhood, c, t):
        # Implement the logic for Day & Night
        center_cell = neighbourhood[1][1]
        total = np.sum(neighbourhood)
        if center_cell == 1:
            if total - 1 < 2: # Any live cell with fewer than two live neighbours dies, as if caused by underpopulation.
                return 0
            if total - 1 == 2 or total - 1 == 3: # Any live cell with two or three live neighbours lives on to the next generation.
                return 1
            if total - 1 > 3: # Any live cell with more than three live neighbours dies, as if by overpopulation.
                return 0
        else:
            if total == 3 or total == 6 or total == 7 or total == 8: # Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
                return 1
            else:
                return 0