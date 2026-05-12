use rand::RngExt;
use crate::core::{
    environment::{Environment, StepResult},
    spaces::Space,
};

const GRAVITY: f32 = 9.8;
const CART_MASS: f32 = 1.0;
const POLE_MASS: f32 = 0.1;
const POLE_HALF_LENGTH: f32 = 0.5;
const FORCE_MAGNITUDE: f32 = 10.0;
const TIMESTEP: f32 = 0.02;

pub struct CartPole {
    cart_pos: f32,
    cart_vel: f32,
    pole_angle: f32,
    pole_vel: f32,
    steps: usize,
    terminated: bool,
    truncated: bool,
    obs_labels: Vec<String>,
}

impl CartPole {
    pub fn new() -> Self {
        let mut cart_pole = CartPole {
            cart_pos: 0.0,
            cart_vel: 0.0,
            pole_angle: 0.0,
            pole_vel: 0.0,
            steps: 0,
            terminated: false,
            truncated: false,
            obs_labels: vec![
                "cart_pos".to_string(),
                "cart_vel".to_string(),
                "pole_angle".to_string(),
                "pole_vel".to_string(),
            ],
        };
        cart_pole.reset();
        cart_pole
    }

    fn get_observation(&self) -> Vec<f32> {
        vec![self.cart_pos, self.cart_vel, self.pole_angle, self.pole_vel]
    }
}

impl Default for CartPole {
    fn default() -> Self {
        Self::new()
    }
}

impl Environment for CartPole {
    fn reset(&mut self) -> Vec<f32> {
        let mut rand = rand::rng();
        self.cart_pos =  rand.random_range(-0.05..0.05);
        self.cart_vel =  rand.random_range(-0.05..0.05);
        self.pole_angle =  rand.random_range(-0.05..0.05);
        self.pole_vel =  rand.random_range(-0.05..0.05);
        self.steps = 0;
        self.terminated = false;
        self.truncated = false;
        self.get_observation()
    }

    fn step(&mut self, action: usize) -> StepResult {
        let reward: f32;
        let cartpole_action_option = CartPoleAction::try_from(action);
        let cartpole_action = match cartpole_action_option {
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

        let force = match cartpole_action {
            CartPoleAction::Left => -FORCE_MAGNITUDE,
            CartPoleAction::Right => FORCE_MAGNITUDE,
        };
        let total_mass = CART_MASS + POLE_MASS;
        // Intermediary accelertation calculation
        let temp: f32 = (force + POLE_MASS * POLE_HALF_LENGTH * self.pole_vel.powi(2) * self.pole_angle.sin()) / total_mass;

        // Angular acceleration
        let theta_ddot = (GRAVITY * self.pole_angle.sin() - self.pole_angle.cos() * temp) / (POLE_HALF_LENGTH * (4.0/3.0 - POLE_MASS * self.pole_angle.cos().powi(2) / total_mass));

        // Cart acceleration 
        let x_ddot = temp - POLE_MASS * POLE_HALF_LENGTH * theta_ddot * self.pole_angle.cos() / total_mass;

        // Euler integration
        self.cart_pos += TIMESTEP * self.cart_vel;
        self.cart_vel += TIMESTEP * x_ddot;
        self.pole_angle += TIMESTEP * self.pole_vel;
        self.pole_vel += TIMESTEP * theta_ddot;

        if self.steps >= 500 {
            self.truncated = true;
        }

        if (self.pole_angle.abs() > 12.0_f32.to_radians()) || (self.cart_pos.abs() > 2.4) {
            self.terminated = true;
            reward = 0.0;
        } else {
            reward = 1.0;
        }

        StepResult {
            observation: self.get_observation(),
            reward,
            terminated: self.terminated,
            truncated: self.truncated,
        }

    }

    fn is_terminal(&self) -> bool {
        self.terminated || self.truncated
    }

    fn observation_space(&self) -> Space {
        Space::Box {
            low: vec![-4.8, -f32::INFINITY, -0.418, -f32::INFINITY],
            high: vec![4.8, f32::INFINITY, 0.418, f32::INFINITY],
            labels: self.obs_labels.clone(),
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete(2)
    }

    fn name(&self) -> &str {
        "CartPole"
    }
}


pub enum CartPoleAction {
    Left = 0,
    Right = 1,
}

impl TryFrom<usize> for CartPoleAction {
    type Error = ();

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(CartPoleAction::Left),
            1 => Ok(CartPoleAction::Right),
            _ => Err(()),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    impl CartPole {
        fn fixed_vals() -> Self {
            // Used for confirming calculations
            CartPole {
                cart_pos: 1.0,
                cart_vel: 0.5,
                pole_angle: 0.0,
                pole_vel: 0.5,
                steps: 0,
                terminated: false,
                truncated: false,
                obs_labels: vec![
                    "cart_pos".to_string(),
                    "cart_vel".to_string(),
                    "pole_angle".to_string(),
                    "pole_vel".to_string(),
                ],
            }
        }
    }

    #[test]
    fn test_reset() {
        let mut env = CartPole::default();
        env.reset();
        assert_eq!(env.steps, 0);
        assert!(!env.terminated);
        assert!(!env.truncated);
        assert!(env.cart_pos < 0.05);
        assert!(env.cart_pos > -0.05);
        assert!(env.cart_vel < 0.05);
        assert!(env.cart_vel > -0.05);
        assert!(env.pole_angle < 0.05);
        assert!(env.pole_angle > -0.05);
        assert!(env.pole_vel < 0.05);
        assert!(env.pole_vel > -0.05);
    }

    #[test]
    fn test_step_result() {
        let mut env = CartPole::fixed_vals();
        // Left
        let step_res = env.step(0);
        assert!(step_res.observation[0] == 1.01);
        assert!(step_res.observation[2] == 0.01);
        assert!((step_res.observation[1] - 0.3049).abs() < 1e-4);
        assert!((step_res.observation[3] - 0.7927).abs() < 1e-4);
        assert!(!step_res.truncated);
        assert!(!step_res.terminated);
        assert_eq!(step_res.reward, 1.0);
    }

    #[test]
    fn test_invalid_action() {
        let mut env = CartPole::fixed_vals();

        let step_res = env.step(10);

        /*
        cart_pos: 1.0,
        cart_vel: 0.5,
        pole_angle: 0.0,
        pole_vel: 0.5, 
        */
        assert!(step_res.observation[0] == 1.0);
        assert!(step_res.observation[1] == 0.5);
        assert!(step_res.observation[2] == 0.0);
        assert!(step_res.observation[3] == 0.5);
        assert!(!step_res.truncated);
        assert!(!step_res.terminated);
        assert_eq!(step_res.reward, 0.0);
    }

    #[test]
    fn test_angle_threshold() {
        let mut env = CartPole::fixed_vals();
        env.pole_angle = 11.9_f32.to_radians();
        let step_res = env.step(1);

        assert!(step_res.terminated);
        assert_eq!(step_res.reward, 0.0);
    }

    #[test]
    fn test_cart_pos_threshold() {
        let mut env = CartPole::fixed_vals();
        env.cart_pos = 2.399;
        let step_res = env.step(1);


        assert!(step_res.terminated);
        assert_eq!(step_res.reward, 0.0);
    }
    
    #[test]
    fn test_step_count_threshold() {
        let mut env = CartPole::fixed_vals();
        env.steps = 499;
        let step_res = env.step(0);

        assert!(step_res.truncated);
        assert_eq!(step_res.reward, 1.0);
    }
}