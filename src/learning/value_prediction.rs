use super::util::StepsizeScheduler;
use std::collections::HashMap;

pub trait ValuePredictor<S> {
    fn get_value_function(&self) -> &HashMap<S, f64>;
    fn step(&mut self);
}

// ┌──────────────────────────────────────────────────────────┐
//  Every-visit Montecarlo
// └──────────────────────────────────────────────────────────┘
pub struct EveryvisitMC<S: Eq + std::hash::Hash + Clone> {
    value_function: HashMap<S, f64>,
    stepsize_scheduler: Box<dyn StepsizeScheduler<S>>,
    gamma: f64,
    episode: Vec<(S, f64)>,
}

impl<S: Eq + std::hash::Hash + Clone> EveryvisitMC<S> {
    pub fn new(
        value_function: HashMap<S, f64>,
        stepsize_scheduler: Box<dyn StepsizeScheduler<S>>,
        gamma: f64,
    ) -> Self {
        EveryvisitMC {
            value_function,
            stepsize_scheduler,
            gamma,
            episode: Vec::new(),
        }
    }

    pub fn update_episode(&mut self, episode: &[(S, f64)]) {
        self.episode = episode.to_vec()
    }

    pub fn get_value(&self, s: &S) -> Option<f64> {
        self.value_function.get(s).cloned()
    }

    pub fn get_stepsize(&mut self, t: usize, s: &S) -> f64 {
        self.stepsize_scheduler.stepsize(t, s)
    }

    pub fn update_value(&mut self, state: &S, value: f64) {
        self.value_function.insert(state.clone(), value);
    }
}

impl<S: Eq + std::hash::Hash + Clone> ValuePredictor<S> for EveryvisitMC<S> {
    fn get_value_function(&self) -> &HashMap<S, f64> {
        &self.value_function
    }

    #[allow(non_snake_case)]
    fn step(&mut self) {
        let l = self.episode.len();
        if l == 0 {
            panic!("Episode is empty");
        }

        let episode = self.episode.clone();

        // Backward update for cumulative discounted return
        let R: Vec<f64> = episode
            .iter()
            .rev()
            .scan(0.0, |acc, (_, r)| {
                *acc = *acc * self.gamma + r;
                Some(*acc)
            })
            .collect();

        // Forward update for value function
        episode
            .iter()
            .zip(R)
            .enumerate()
            .for_each(|(t, ((s, _), r))| {
                let v = self.get_value(s).unwrap_or(0.0);
                let alpha = self.get_stepsize(t, s);
                self.update_value(s, v + alpha * (r - v))
            })
    }
}

// ┌──────────────────────────────────────────────────────────┐
//  Temporal Difference Learning (TD(0))
// └──────────────────────────────────────────────────────────┘
pub struct TD0<S: Eq + std::hash::Hash + Clone> {
    value_function: HashMap<S, f64>,
    stepsize_scheduler: Box<dyn StepsizeScheduler<S>>,
    gamma: f64,
    one_step: Option<(S, f64, Option<S>)>,
    _count: usize,
}

impl<S: Eq + std::hash::Hash + Clone> TD0<S> {
    pub fn new(
        value_function: HashMap<S, f64>,
        stepsize_scheduler: Box<dyn StepsizeScheduler<S>>,
        gamma: f64,
    ) -> Self {
        TD0 {
            value_function,
            stepsize_scheduler,
            gamma,
            one_step: None,
            _count: 0,
        }
    }

    pub fn get_value(&self, s: &S) -> Option<f64> {
        self.value_function.get(s).cloned()
    }

    pub fn get_stepsize(&mut self, t: usize, s: &S) -> f64 {
        self.stepsize_scheduler.stepsize(t, s)
    }

    pub fn update_value(&mut self, state: &S, value: f64) {
        self.value_function.insert(state.clone(), value);
    }

    pub fn update_one_step(&mut self, s: S, r: f64, s_next: Option<S>) {
        self.one_step = Some((s, r, s_next));
    }

    pub fn increment_count(&mut self) {
        self._count += 1;
    }

    pub fn reset_increment(&mut self) {
        self._count = 1;
    }
}

impl<S: Eq + std::hash::Hash + Clone> ValuePredictor<S> for TD0<S> {
    fn get_value_function(&self) -> &HashMap<S, f64> {
        &self.value_function
    }
    #[allow(non_snake_case)]
    fn step(&mut self) {
        let (s, r, s_next) = self.one_step.take().unwrap();
        let delta = match s_next {
            Some(s_next) => {
                r + self.gamma * self.get_value(&s_next).unwrap_or(0.0)
                    - self.get_value(&s).unwrap_or(0.0)
            }
            None => r - self.get_value(&s).unwrap_or(0.0),
        };
        let alpha = self.get_stepsize(self._count, &s);
        let v = self.get_value(&s).unwrap_or(0.0);
        let new_v = v + alpha * delta;
        self.update_value(&s, new_v);
        self.increment_count();
    }
}
