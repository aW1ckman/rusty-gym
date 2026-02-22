use crate::core::spaces::Space;
use std::convert::TryFrom;

pub struct GridWorld {
    grid_size: (usize, usize),
    agent_pos: (usize, usize),
    goal_pos: (usize, usize),
    steps: usize,
    max_steps: usize,
    terminated: bool,
}

impl Default for GridWorld {
    fn default() -> Self {
        GridWorld {
            grid_size: (5, 5),
            agent_pos: (0, 0),
            goal_pos: (4, 4),
            steps: 0,
            max_steps: 100,
            terminated: false,
        }
    }
}

impl GridWorld {
    pub fn new(
        grid_size: (usize, usize),
        agent_pos: (usize, usize),
        goal_pos: (usize, usize),
        max_steps: usize,
    ) -> Self {
        GridWorld {
            grid_size,
            agent_pos,
            goal_pos,
            steps: 0,
            max_steps,
            terminated: false,
        }
    }
}

pub enum GridWorldAction {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
}

impl TryFrom<usize> for GridWorldAction {
    type Error = ();

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(GridWorldAction::Up),
            1 => Ok(GridWorldAction::Down),
            2 => Ok(GridWorldAction::Left),
            3 => Ok(GridWorldAction::Right),
            _ => Err(()),
        }
    }
}
