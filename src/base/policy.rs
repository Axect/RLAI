use crate::base::process::MarkovDecisionProcess;
use peroxide::fuga::*;
use std::collections::HashMap;
use std::marker::PhantomData;

pub trait Policy<S, A> {
    fn gen_action(&self, state: &S) -> Option<A>;
}

// ┌──────────────────────────────────────────────────────────┐
//  Greedy Policy (Value)
// └──────────────────────────────────────────────────────────┘
// Greedy value policy implementation
pub struct GreedyValuePolicy<'a, S, A, M: MarkovDecisionProcess<S, A>> {
    mdp: &'a M,
    value_function: HashMap<S, f64>,
    action_type: PhantomData<A>,
}

impl<'a, S: Eq + std::hash::Hash + Clone, A: Clone, M: MarkovDecisionProcess<S, A>>
    GreedyValuePolicy<'a, S, A, M>
{
    pub fn new(mdp: &'a M, value_function: HashMap<S, f64>) -> Self {
        GreedyValuePolicy {
            mdp,
            value_function,
            action_type: PhantomData,
        }
    }

    pub fn get_mdp(&self) -> &M {
        self.mdp
    }

    pub fn get_value_function(&self) -> &HashMap<S, f64> {
        &self.value_function
    }
}

impl<'a, S: Eq + std::hash::Hash + Clone, A: Clone, M: MarkovDecisionProcess<S, A>> Policy<S, A>
    for GreedyValuePolicy<'a, S, A, M>
{
    fn gen_action(&self, state: &S) -> Option<A> {
        let mdp = self.get_mdp();
        let v = self.get_value_function();
        mdp.actions_at(state).into_iter().max_by(|a, b| {
            let value_a = mdp
                .transition(state, a)
                .and_then(|s| v.get(&s))
                .unwrap_or(&0.0);
            let value_b = mdp
                .transition(state, b)
                .and_then(|s| v.get(&s))
                .unwrap_or(&0.0);
            value_a
                .partial_cmp(value_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

// ┌──────────────────────────────────────────────────────────┐
//  Epsilon Greedy Policy (Value)
// └──────────────────────────────────────────────────────────┘
pub struct EpsilonGreedyValuePolicy<'a, S, A, M: MarkovDecisionProcess<S, A>> {
    mdp: &'a M,
    value_function: HashMap<S, f64>,
    action_type: PhantomData<A>,
    _bernoulli: OPDist<f64>,
    _random: bool,
}

impl<'a, S: Eq + std::hash::Hash + Clone, A: Clone, M: MarkovDecisionProcess<S, A>>
    EpsilonGreedyValuePolicy<'a, S, A, M>
{
    pub fn new(mdp: &'a M, value_function: HashMap<S, f64>, epsilon: f64) -> Self {
        let bernoulli = Bernoulli(epsilon);
        EpsilonGreedyValuePolicy {
            mdp,
            value_function,
            action_type: PhantomData,
            _bernoulli: bernoulli,
            _random: true,
        }
    }
    pub fn get_mdp(&self) -> &M {
        self.mdp
    }
    pub fn get_value_function(&self) -> &HashMap<S, f64> {
        &self.value_function
    }
    pub fn update_value_function(&mut self, value_function: &HashMap<S, f64>) {
        self.value_function = value_function.clone();
    }

    pub fn turn_off_random(&mut self) {
        self._random = false;
    }
}

impl<'a, S: Eq + std::hash::Hash + Clone, A: Clone, M: MarkovDecisionProcess<S, A>> Policy<S, A>
    for EpsilonGreedyValuePolicy<'a, S, A, M>
{
    fn gen_action(&self, state: &S) -> Option<A> {
        let sample = self._bernoulli.sample(1)[0];
        let sample = sample > 0f64;

        let mdp = self.get_mdp();
        let v = self.get_value_function();
        if sample && self._random {
            mdp.actions_at(state).into_iter().choose(&mut thread_rng())
        } else {
            mdp.actions_at(state).into_iter().max_by(|a, b| {
                let value_a = mdp
                    .transition(state, a)
                    .and_then(|s| v.get(&s))
                    .unwrap_or(&0.0);
                let value_b = mdp
                    .transition(state, b)
                    .and_then(|s| v.get(&s))
                    .unwrap_or(&0.0);
                value_a
                    .partial_cmp(value_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        }
    }
}
