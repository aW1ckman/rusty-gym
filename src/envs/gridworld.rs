use crate::core::{
    environment::{Environment, StepResult},
    spaces::Space,
};
use std::convert::TryFrom;

pub struct GridWorld {
    grid_size: (usize, usize),
    start_pos: (usize, usize),
    agent_pos: (usize, usize),
    goal_pos: (usize, usize),
    steps: usize,
    max_steps: usize,
    terminated: bool,
    truncated: bool,
}

impl Default for GridWorld {
    fn default() -> Self {
        GridWorld {
            grid_size: (5, 5),
            start_pos: (0, 0),
            agent_pos: (0, 0),
            goal_pos: (4, 4),
            steps: 0,
            max_steps: 100,
            terminated: false,
            truncated: false,
        }
    }
}

impl GridWorld {
    pub fn new(
        grid_size: (usize, usize),
        start_pos: (usize, usize),
        goal_pos: (usize, usize),
        max_steps: usize,
    ) -> Self {
        GridWorld {
            grid_size,
            start_pos,
            agent_pos: start_pos,
            goal_pos,
            steps: 0,
            max_steps,
            terminated: false,
            truncated: false,
        }
    }

    fn get_observation(&self) -> Vec<f32> {
        let norm_x: f32 = self.agent_pos.0 as f32 / (self.grid_size.0 - 1) as f32;
        let norm_y: f32 = self.agent_pos.1 as f32 / (self.grid_size.1 - 1) as f32;
        vec![norm_x, norm_y]
    }
}

impl Environment for GridWorld {
    fn reset(&mut self) -> Vec<f32> {
        self.agent_pos = self.start_pos;
        self.steps = 0;
        self.terminated = false;
        self.truncated = false;

        self.get_observation()
    }

    fn is_terminal(&self) -> bool {
        self.terminated || self.truncated
    }
    fn action_space(&self) -> Space {
        Space::Discrete(4)
    }
    fn name(&self) -> &str {
        "GridWorld"
    }
    fn observation_space(&self) -> Space {
        Space::Box {
            low: vec![0.0, 0.0],
            high: vec![1.0, 1.0],
        }
    }
    fn step(&mut self, action: usize) -> StepResult {
        let reward;
        let grid_action_option = GridWorldAction::try_from(action);
        let grid_action = match grid_action_option {
            Ok(action) => action,
            Err(_) => {
                return StepResult {
                    observation: self.get_observation(),
                    reward: 0.0,
                    terminated: self.terminated,
                    truncated: self.truncated,
                };
            }
        };

        self.steps += 1;
        match grid_action {
            GridWorldAction::Left => self.agent_pos.0 = self.agent_pos.0.saturating_sub(1),
            GridWorldAction::Right => {
                self.agent_pos.0 = (self.agent_pos.0 + 1).min(self.grid_size.0 - 1)
            }
            GridWorldAction::Up => {
                self.agent_pos.1 = (self.agent_pos.1 + 1).min(self.grid_size.1 - 1)
            }
            GridWorldAction::Down => self.agent_pos.1 = self.agent_pos.1.saturating_sub(1),
        }

        if self.agent_pos == self.goal_pos {
            reward = 1.0;
            self.terminated = true;
        } else if self.steps >= self.max_steps {
            reward = -1.0;
            self.truncated = true;
        } else {
            reward = -0.01;
        }

        StepResult {
            observation: self.get_observation(),
            reward,
            terminated: self.terminated,
            truncated: self.truncated,
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
