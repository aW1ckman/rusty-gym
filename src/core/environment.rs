use crate::core::spaces::Space;

pub struct StepResult {
    pub observation: Vec<f32>,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
}

pub trait Environment: Send {
    fn reset(&mut self) -> Vec<f32>;
    fn step(&mut self, action: usize) -> StepResult;
    fn is_terminal(&self) -> bool;
    fn observation_space(&self) -> &Space;
    fn action_space(&self) -> &Space;
    fn name(&self) -> &str;
}
