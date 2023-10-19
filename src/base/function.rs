pub trait ValueFunction<S> {
    fn value(&self, state: &S) -> f64;
}

pub trait ActionValueFunction<S, A> {
    fn value(&self, state: &S, action: &A) -> f64;
}
