use indicatif::{ProgressBar, ProgressStyle};
use peroxide::fuga::*;
use rlai::{
    base::{
        policy::{EpsilonGreedyValuePolicy, Policy},
        process::MarkovDecisionProcess,
    },
    env::grid_world::GridWorld,
    learning::{
        util::InverseTimeDecay,
        value_prediction::{ValuePredictor, TD0},
    },
};
use std::collections::HashMap;

fn main() {
    let goal_state = (4, 3);
    let terminal_states = vec![(1, 0), (1, 1), (1, 2), (1, 3), (3, 4), (3, 3)];
    let env = GridWorld::new(5, 5, (0, 0), goal_state, terminal_states.clone());
    let stepsize_scheduler = InverseTimeDecay::new(10f64);

    let mut value_function = HashMap::new();
    for state in env.states() {
        value_function.insert(state, 0f64);
    }

    let mut policy = EpsilonGreedyValuePolicy::new(&env, value_function.clone(), 0.1);
    let mut value_predictor: TD0<(usize, usize)> =
        TD0::new(value_function.clone(), Box::new(stepsize_scheduler), 0.95);

    let mut episodes = vec![];
    let n = 500;
    let pb = ProgressBar::new(n);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    let max_step = 1000;
    for _ in 0..n {
        let mut episode = Vec::new();
        let mut current_state = env.get_init_state();
        value_predictor.reset_increment();
        for _ in 0..max_step {
            let action = policy.gen_action(&current_state).unwrap();

            let (s_next, r) = env.step(&current_state, &action);
            episode.push((current_state, r));

            // 1. Update Value Function via TD(0)
            value_predictor.update_one_step(current_state, r, s_next);
            value_predictor.step();
            let new_value_function = value_predictor.get_value_function();

            // 2. Update Policy
            policy.update_value_function(new_value_function);
            episode.push((current_state, r));

            if s_next.is_none() {
                break;
            }
            current_state = s_next.unwrap();
        }

        pb.inc(1);
        pb.set_message(format!("Episode length: {}", episode.len()));

        episodes.push(episode);
    }

    println!("Test!");
    println!(
        "Value Function: {:#?}",
        value_predictor.get_value_function()
    );

    // Test
    // - Turn off random policy
    policy.turn_off_random();
    let mut test_episode = Vec::new();
    let mut current_state = env.get_init_state();
    let mut j = 0usize;
    for i in 0..max_step {
        let action = policy.gen_action(&current_state).unwrap();
        match env.step(&current_state, &action) {
            (None, r) => {
                test_episode.push((current_state, r));
                break;
            }
            (Some(s), r) => {
                test_episode.push((current_state, r));
                current_state = s;
                j += 1;
            }
        }
    }
    println!("j = {j}");

    // Store first episodes
    let mut df = DataFrame::new(vec![]);
    let first_episode = episodes[0].clone();
    let ((episode_x, episode_y), rewards): ((Vec<usize>, Vec<usize>), Vec<f64>) =
        first_episode.into_iter().unzip();
    df.push(
        "episode_x",
        Series::new(episode_x.into_iter().map(|x| x as u64).collect()),
    );
    df.push(
        "episode_y",
        Series::new(episode_y.into_iter().map(|x| x as u64).collect()),
    );
    df.push("reward", Series::new(rewards));
    df.write_parquet(
        "./data/grid_world/td0-epsilon_greedy-first.parquet",
        CompressionOptions::Uncompressed,
    )
    .expect("Can't write parquet file");

    // Store test episode
    let mut df = DataFrame::new(vec![]);
    let ((episode_x, episode_y), rewards): ((Vec<usize>, Vec<usize>), Vec<f64>) =
        test_episode.into_iter().unzip();
    df.push(
        "episode_x",
        Series::new(episode_x.into_iter().map(|x| x as u64).collect()),
    );
    df.push(
        "episode_y",
        Series::new(episode_y.into_iter().map(|x| x as u64).collect()),
    );
    df.push("reward", Series::new(rewards));
    df.write_parquet(
        "./data/grid_world/td0-epsilon_greedy-test.parquet",
        CompressionOptions::Uncompressed,
    )
    .expect("Can't write parquet file");

    // Store all episodes' length
    let mut df = DataFrame::new(vec![]);
    df.push(
        "length",
        Series::new(
            episodes
                .iter()
                .map(|e| e.len() as u64)
                .collect::<Vec<u64>>(),
        ),
    );
    df.write_parquet(
        "./data/grid_world/td0-epsilon_greedy-length.parquet",
        CompressionOptions::Uncompressed,
    )
    .expect("Can't write parquet file");
    df.print();

    // Store Goal State
    let mut df = DataFrame::new(vec![]);
    df.push(
        "goal_x",
        Series::new(vec![goal_state.0 as u64; episodes.len()]),
    );
    df.push(
        "goal_y",
        Series::new(vec![goal_state.1 as u64; episodes.len()]),
    );
    df.write_parquet(
        "./data/grid_world/td0-epsilon_greedy-goal.parquet",
        CompressionOptions::Uncompressed,
    )
    .expect("Can't write parquet file");

    // Store Terminal States
    let mut terminal_x = vec![];
    let mut terminal_y = vec![];
    for (x, y) in terminal_states {
        terminal_x.push(x as u64);
        terminal_y.push(y as u64);
    }
    let mut df = DataFrame::new(vec![]);
    df.push("terminal_x", Series::new(terminal_x));
    df.push("terminal_y", Series::new(terminal_y));
    df.write_parquet(
        "./data/grid_world/td0-epsilon_greedy-terminal.parquet",
        CompressionOptions::Uncompressed,
    )
    .expect("Can't write parquet file");
}
