pub enum Space {
    Discrete(usize),
    Box { low: Vec<f32>, high: Vec<f32>, labels: Vec<String> },
}
