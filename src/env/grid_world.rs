use crate::base::process::MarkovDecisionProcess;
use GridWorldAction as GWA;

// ┌──────────────────────────────────────────────────────────┐
//  Grid World
// └──────────────────────────────────────────────────────────┘
#[derive(Debug, Clone)]
pub struct GridWorld {
    num_x: usize,
    num_y: usize,
    current_state: (usize, usize),
    goal_state: (usize, usize),
    terminal_states: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, Copy)]
pub enum GridWorldAction {
    Up,
    Down,
    Left,
    Right,
}

impl GridWorld {
    pub fn new(
        num_x: usize,
        num_y: usize,
        current_state: (usize, usize),
        goal_state: (usize, usize),
        terminal_states: Vec<(usize, usize)>,
    ) -> Self {
        GridWorld {
            num_x,
            num_y,
            current_state,
            goal_state,
            terminal_states,
        }
    }
    pub fn get_num_x(&self) -> usize {
        self.num_x
    }
    pub fn get_num_y(&self) -> usize {
        self.num_y
    }
    pub fn get_current_state(&self) -> (usize, usize) {
        self.current_state
    }
    pub fn get_goal_state(&self) -> (usize, usize) {
        self.goal_state
    }
    pub fn get_terminal_states(&self) -> Vec<(usize, usize)> {
        self.terminal_states.clone()
    }
}

impl MarkovDecisionProcess<(usize, usize), GridWorldAction> for GridWorld {
    fn states(&self) -> Vec<(usize, usize)> {
        let mut states = Vec::new();

        for x in 0..self.num_x {
            for y in 0..self.num_y {
                states.push((x, y))
            }
        }
        states
    }

    fn actions(&self) -> Vec<GridWorldAction> {
        vec![GWA::Up, GWA::Down, GWA::Left, GWA::Right]
    }

    fn actions_at(&self, state: &(usize, usize)) -> Vec<GridWorldAction> {
        let x_max = self.num_x - 1;
        let y_max = self.num_y - 1;

        match state {
            (0, 0) => vec![GWA::Right, GWA::Up],
            (0, y) if *y == y_max => vec![GWA::Right, GWA::Down],
            (0, _) => vec![GWA::Right, GWA::Up, GWA::Down],
            (x, 0) if *x == x_max => vec![GWA::Left, GWA::Up],
            (x, y) if *x == x_max && *y == y_max => vec![GWA::Left, GWA::Down],
            (x, _) if *x == x_max => vec![GWA::Left, GWA::Up, GWA::Down],
            (_, 0) => vec![GWA::Left, GWA::Right, GWA::Up],
            (_, y) if *y == y_max => vec![GWA::Left, GWA::Right, GWA::Down],
            (_, _) => vec![GWA::Left, GWA::Right, GWA::Up, GWA::Down],
        }
    }

    fn reward(&self, state: &(usize, usize), action: &GridWorldAction) -> f64 {
        if let Some(next_state) = self.transition(state, action) {
            if next_state == self.goal_state {
                1.0
            } else {
                0.0
            }
        } else {
            -1.0
        }
    }

    fn transition(
        &self,
        state: &(usize, usize),
        action: &GridWorldAction,
    ) -> Option<(usize, usize)> {
        if self.terminal_states.contains(state) || self.goal_state.eq(state) {
            return None;
        }

        let &(x, y) = state;

        let x_max = self.num_x - 1;
        let y_max = self.num_y - 1;

        match action {
            GWA::Up => {
                if y == y_max {
                    None
                } else {
                    Some((x, y + 1))
                }
            }
            GWA::Down => {
                if y == 0 {
                    None
                } else {
                    Some((x, y - 1))
                }
            }
            GWA::Left => {
                if x == 0 {
                    None
                } else {
                    Some((x - 1, y))
                }
            }
            GWA::Right => {
                if x == x_max {
                    None
                } else {
                    Some((x + 1, y))
                }
            }
        }
    }
}
