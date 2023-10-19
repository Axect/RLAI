use super::policy::Policy;

pub trait MarkovDecisionProcess<S, A> {
    fn states(&self) -> Vec<S>;
    fn actions(&self) -> Vec<A>;
    fn actions_at(&self, state: &S) -> Vec<A>;
    fn reward(&self, state: &S, action: &A) -> f64;
    fn transition(&self, state: &S, action: &A) -> Option<S>;
}

pub trait MarkovRewardProcess<S, A>: MarkovDecisionProcess<S, A> {
    fn get_policy(&self) -> &dyn Policy<S, A>;
}
