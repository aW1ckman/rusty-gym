use rand::{RngExt, seq::{IndexedRandom}};

use crate::core::{
    environment::{Environment, StepResult},
    spaces::Space,
};

const BALLSPEED: f32 = 0.025;
const MAX_VY: f32 = BALLSPEED * 0.8;
const BALLSIZE: f32 = 0.0125;

const PADDLESPEED: f32 = 0.02;
// Simulating: 20px paddle height (240px screen)
const PADDLEHEIGHT: f32 = 20.0/240.0;
const PADDLEWIDTH: f32 = 0.025;
const PADDLEMARGIN: f32 = 0.04;

const MAXSTEPS: usize = 1000;
const HITREWARD: f32 = 0.1;
const SCOREREWARD: f32 = 1.0;
const CONCEDEREWARD: f32 = -1.0;

const OPPONENTREACTION: f32 = 0.4;
const OPPONENTERROR: f32 = 0.08;


pub struct Pong {
    ball_x: f32,
    ball_y: f32,
    ball_vx: f32,
    ball_vy: f32,
    agent_paddle: Paddle,
    opp_paddle: Paddle,
    opp_actor: Box<dyn OpponentPolicy + Send>,
    steps: usize,
    terminated: bool,
    truncated: bool,
    obs_labels: Vec<String>,
}

impl Pong {
    pub fn new() -> Self {
        let mut pong_game = Self::default();
        pong_game.reset();
        pong_game
    }

    fn get_observation(&self) -> Vec<f32> {
        vec![self.ball_x, self.ball_y, self.ball_vx, self.ball_vy, self.agent_paddle.paddle_y, self.opp_paddle.paddle_y]
    }
}

impl Default for Pong {
    fn default() -> Self {
        let opp = TrackingOpponent {rand: rand::make_rng()};
        Pong {
            ball_x: 0.5,
            ball_y: 0.5,
            ball_vx: BALLSPEED,
            ball_vy: 0.0,
            agent_paddle: Paddle::new(true),
            opp_paddle: Paddle::new(false),
            opp_actor: Box::new(opp),
            steps: 0,
            terminated: false,
            truncated: false,
            obs_labels: vec![
                "ball_x".to_string(),
                "ball_y".to_string(),
                "ball_vx".to_string(),
                "ball_vy".to_string(),
                "paddle_y".to_string(),
                "opp_paddle_y".to_string(),
            ],
        }
    }
}

pub struct Paddle {
    paddle_y: f32,
    paddle_x: f32,
}

impl Paddle {
    pub fn new(agent: bool) -> Self {
        let mut paddle = Self::default();
        if agent {
            paddle.paddle_x = PADDLEMARGIN;
        } else {
            paddle.paddle_x = 1.0 - PADDLEMARGIN - PADDLEWIDTH;
        }
        paddle
    }

    fn reset(&mut self) {
        self.paddle_y = 0.5;
    }

    fn move_paddle(&mut self, action: PongAction) {
        // Top-left origin 
        match action {
            PongAction::Stay => {},
            PongAction::Up => self.paddle_y -= PADDLESPEED ,
            PongAction::Down => self.paddle_y += PADDLESPEED,
        }

        if self.paddle_y < 0.0 {
            self.paddle_y = 0.0;
        } else if self.paddle_y > 1.0-PADDLEHEIGHT {
            self.paddle_y = 1.0-PADDLEHEIGHT
        }
    }

    fn get_paddle_centre(&self) -> f32 {
        self.paddle_y + PADDLEHEIGHT / 2.0
    }
}

impl Default for Paddle {
    fn default() -> Self {
        Paddle { 
            paddle_y: 0.5,
            paddle_x: 0.0, 
        }
    }
}

pub trait OpponentPolicy {
    fn take_action(&mut self, ball_centre: f32, paddle_centre: f32) -> PongAction;
}

pub struct TrackingOpponent {
    rand: rand::rngs::SmallRng,
}

impl OpponentPolicy for TrackingOpponent {
    fn take_action(&mut self, ball_centre: f32, paddle_centre: f32) -> PongAction {
        if self.rand.random_range(0.0..=1.0) > OPPONENTREACTION {
            return PongAction::Stay
        }
        let diff = ball_centre - paddle_centre;
        if diff < -OPPONENTERROR {
            PongAction::Up
        } else if diff > OPPONENTERROR {
            PongAction::Down
        } else {
            PongAction::Stay
        }
    }
}

impl Pong {
    fn update_ball(&mut self) {
        self.ball_x += self.ball_vx;
        self.ball_y += self.ball_vy;
        
        // Top bottom wall collision
        if self.ball_y <= 0.0 {
            self.ball_y = 0.0;
            self.ball_vy = -self.ball_vy;
        } else if self.ball_y >= 1.0-BALLSIZE {
            self.ball_y = 1.0-BALLSIZE;
            self.ball_vy = -self.ball_vy
        }
    }

    fn clamp_ball_vy(&mut self) {
        // Clamp VY to always be at most MAX_VY
        self.ball_vy = self.ball_vy.clamp(-MAX_VY, MAX_VY);
    }

    fn ball_centre(&self) -> f32 {
        self.ball_y + BALLSIZE / 2.0
    }

    fn paddle_collisions(&mut self) -> PongEvent {
        // Ball moving left (to agent)
        if self.ball_vx < 0.0
            // Check: if ball touching paddle
            && self.ball_x <= self.agent_paddle.paddle_x + PADDLEWIDTH &&
            self.ball_x >= self.agent_paddle.paddle_x &&
            self.ball_y >= self.agent_paddle.paddle_y - BALLSIZE &&
            self.ball_y <= self.agent_paddle.paddle_y + PADDLEHEIGHT {
                
                self.ball_x = self.agent_paddle.paddle_x + PADDLEWIDTH;
                self.ball_vx = -self.ball_vx;

                // Vary angle based on hit position
                let hitpos = self.ball_centre() - self.agent_paddle.get_paddle_centre();
                self.ball_vy = hitpos * 0.1; // Offset -> velocity
                self.clamp_ball_vy();

                // Agent hit ball
                return PongEvent::AgentHit;
        }
        // Ball moving right (to opponent)
        if self.ball_vx > 0.0
            && self.ball_x + BALLSIZE >= self.opp_paddle.paddle_x &&
            self.ball_x + BALLSIZE <= self.opp_paddle.paddle_x + PADDLEWIDTH &&
            self.ball_y >= self.opp_paddle.paddle_y - BALLSIZE &&
            self.ball_y <= self.opp_paddle.paddle_y + PADDLEHEIGHT {

                self.ball_x = self.opp_paddle.paddle_x - BALLSIZE;
                self.ball_vx = -self.ball_vx;

                let hitpos = (self.ball_y + BALLSIZE / 2.0) - self.opp_paddle.get_paddle_centre();
                self.ball_vy = hitpos * 0.1;
                self.clamp_ball_vy();

                return PongEvent::OppHit
            }
        PongEvent::None
    }

    fn scoring(&mut self) -> PongEvent {
        if self.ball_x < 0.0 {
            self.terminated = true;
            return PongEvent::OppScore;
        } else if self.ball_x > 1.0 - BALLSIZE {
            self.terminated = true;
            return PongEvent::AgentScore;
        }
        PongEvent::None
    }

    fn event_step(&mut self) -> PongEvent {
        let agent_hit = self.paddle_collisions();

        if agent_hit == PongEvent::None {
            return self.scoring();
        }

        agent_hit
    }
}

impl Environment for Pong {
    fn reset(&mut self) -> Vec<f32> {
        let mut rand = rand::rng();
        self.ball_x = 0.5;
        self.ball_y = 0.5;
        // Random directions
        self.ball_vx = BALLSPEED * ([-1.0, 1.0].choose(&mut rand).unwrap());
        self.ball_vy = rand.random_range((-BALLSPEED*0.5)..(BALLSPEED*0.5));

        self.agent_paddle.reset();
        self.opp_paddle.reset();

        self.steps = 0;
        self.terminated = false;
        self.truncated = false;

        self.get_observation()
    }

    fn step(&mut self, action: usize) -> StepResult {
        
        let pong_action_option = PongAction::try_from(action);
        let pong_action = match pong_action_option {
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
        // Move padels first
        self.agent_paddle.move_paddle(pong_action);
        let opp_action = self.opp_actor.take_action(self.ball_centre(), self.opp_paddle.get_paddle_centre());
        self.opp_paddle.move_paddle(opp_action);

        
        self.update_ball();

        let event = self.event_step();

        let reward: f32 = match event {
            PongEvent::AgentHit => HITREWARD,
            PongEvent::OppHit => 0.0,
            PongEvent::AgentScore => SCOREREWARD,
            PongEvent::OppScore => CONCEDEREWARD,
            PongEvent::None => 0.0,
        };

        if self.steps >= MAXSTEPS {
            self.truncated = true
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

    fn observation_bounds(&self) -> Space {
        Space::Box { low: vec![0.0, 0.0, -BALLSPEED, -MAX_VY, 0.0, 0.0], 
            high: vec![1.0, 1.0, BALLSPEED, MAX_VY, 1.0, 1.0], 
            labels: self.obs_labels.clone(), 
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete(3)
    }

    fn name(&self) -> &str {
        "Pong"
    }
}


pub enum PongAction {
    Stay = 0,
    Up = 1,
    Down = 2,
}

impl TryFrom<usize> for PongAction {
    type Error = ();

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(PongAction::Stay),
            1 => Ok(PongAction::Up),
            2 => Ok(PongAction::Down),
            _ => Err(()),
        }
    }
}

#[derive(PartialEq)]
pub enum PongEvent {
    AgentHit,
    OppHit,
    AgentScore,
    OppScore,
    None
}
