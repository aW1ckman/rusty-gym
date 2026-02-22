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
        }
    }
}

impl Environment for GridWorld {
    fn reset(&mut self) -> Vec<f32> {
        self.agent_pos = self.start_pos;
        self.steps = 0;
        self.terminated = false;

        let norm_x: f32 = self.agent_pos.0 as f32 / (self.grid_size.0 - 1) as f32;
        let norm_y: f32 = self.agent_pos.1 as f32 / (self.grid_size.1 - 1) as f32;
        vec![norm_x, norm_y]
    }

    fn is_terminal(&self) -> bool {
        self.terminated
    }
    fn action_space(&self) -> &Space {
        unimplemented!()
    }
    fn name(&self) -> &str {
        unimplemented!()
    }
    fn observation_space(&self) -> &Space {
        unimplemented!()
    }
    fn step(&mut self, action: usize) -> StepResult {
        unimplemented!()
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
