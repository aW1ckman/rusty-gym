pub enum Space {
    Discrete(usize),
    Box { low: Vec<f64>, high: Vec<f64> },
}
