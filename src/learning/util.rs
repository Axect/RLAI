use std::collections::HashMap;

// ┌──────────────────────────────────────────────────────────┐
//  Step-size Scheduler
// └──────────────────────────────────────────────────────────┘
pub trait StepsizeScheduler<S> {
    fn stepsize(&mut self, t: usize, s: &S) -> f64;
}

/// Constant stepsize scheduler
#[derive(Debug, Clone)]
pub struct ConstantStepsize {
    c: f64,
}

impl ConstantStepsize {
    pub fn new(c: f64) -> Self {
        ConstantStepsize { c }
    }
}

impl<S> StepsizeScheduler<S> for ConstantStepsize {
    fn stepsize(&mut self, _t: usize, _s: &S) -> f64 {
        self.c
    }
}

/// Inverse stepsize scheduler
///
/// alpha_t = c / t
#[derive(Debug, Clone)]
pub struct InverseTimeDecay {
    c: f64,
}

impl InverseTimeDecay {
    pub fn new(c: f64) -> Self {
        InverseTimeDecay { c }
    }
}

impl<S> StepsizeScheduler<S> for InverseTimeDecay {
    fn stepsize(&mut self, t: usize, _s: &S) -> f64 {
        self.c / t as f64
    }
}

/// Power Decay stepsize Scheduler
///
/// alpha_t = c / t^eta
#[derive(Debug, Clone)]
pub struct PowerDecay {
    c: f64,
    eta: f64,
}

impl PowerDecay {
    pub fn new(c: f64, eta: f64) -> Self {
        PowerDecay { c, eta }
    }
}

impl<S> StepsizeScheduler<S> for PowerDecay {
    fn stepsize(&mut self, t: usize, _s: &S) -> f64 {
        self.c / (t as f64).powf(self.eta)
    }
}

/// Count-based stepsize Scheduler
///
/// alpha_t = c / N_t where N_t is the number of times state s has been visited (N_t >= 1)
#[derive(Debug, Clone)]
pub struct CountDecay<S: Eq + std::hash::Hash + Clone> {
    c: f64,
    counts: HashMap<S, usize>,
}

impl<S: Eq + std::hash::Hash + Clone> CountDecay<S> {
    pub fn new(c: f64) -> Self {
        CountDecay {
            c,
            counts: HashMap::new(),
        }
    }
}

impl<S: Eq + std::hash::Hash + Clone> StepsizeScheduler<S> for CountDecay<S> {
    fn stepsize(&mut self, _t: usize, s: &S) -> f64 {
        let count = self.counts.entry(s.clone()).or_insert(0);
        *count += 1;
        self.c / *count as f64
    }
}
