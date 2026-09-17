//! Monte Carlo stock price simulator using Geometric Brownian Motion (GBM).
//!
//! Model:
//!   S_{t+dt} = S_t * exp( (mu - sigma^2 / 2) * dt + sigma * sqrt(dt) * Z )
//! where Z ~ N(0, 1), mu is annualized drift, sigma is annualized volatility,
//! and dt = 1 / trading_days_per_year.

use clap::Parser;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};

const TRADING_DAYS_PER_YEAR: f64 = 252.0;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "mc-sim",
    about = "Monte Carlo stock price simulator (Geometric Brownian Motion)"
)]
struct Args {
    /// Initial stock price (S0)
    #[arg(long, default_value_t = 100.0)]
    initial_price: f64,

    /// Annualized expected return / drift (e.g. 0.08 for 8%)
    #[arg(long, default_value_t = 0.08)]
    drift: f64,

    /// Annualized volatility (e.g. 0.25 for 25%)
    #[arg(long, default_value_t = 0.25)]
    volatility: f64,

    /// Number of trading days to simulate forward
    #[arg(long, default_value_t = 252)]
    days: usize,

    /// Number of Monte Carlo paths to simulate
    #[arg(long, default_value_t = 100_000)]
    simulations: usize,

    /// Optional RNG seed for reproducible runs
    #[arg(long)]
    seed: Option<u64>,

    /// Confidence level for VaR / CVaR, e.g. 0.95
    #[arg(long, default_value_t = 0.95)]
    confidence: f64,

    /// Optional path to write final-price distribution as CSV (one price per line)
    #[arg(long)]
    output: Option<String>,

    /// Optional path to write a sample of full price paths as CSV (rows = days, cols = paths)
    #[arg(long)]
    paths_output: Option<String>,

    /// How many sample paths to record when --paths-output is set
    #[arg(long, default_value_t = 100)]
    sample_paths: usize,
}

struct SimOutcome {
    final_price: f64,
    path: Option<Vec<f64>>,
}

fn simulate_one_path(
    args: &Args,
    dt: f64,
    normal: &Normal<f64>,
    rng: &mut StdRng,
    record_path: bool,
) -> SimOutcome {
    let drift_term = (args.drift - 0.5 * args.volatility * args.volatility) * dt;
    let vol_term = args.volatility * dt.sqrt();

    let mut price = args.initial_price;
    let mut path = if record_path {
        let mut p = Vec::with_capacity(args.days + 1);
        p.push(price);
        Some(p)
    } else {
        None
    };

    for _ in 0..args.days {
        let z: f64 = normal.sample(rng);
        price *= (drift_term + vol_term * z).exp();
        if let Some(p) = path.as_mut() {
            p.push(price);
        }
    }

    SimOutcome {
        final_price: price,
        path,
    }
}

struct Stats {
    mean: f64,
    std_dev: f64,
    min: f64,
    max: f64,
    median: f64,
    p05: f64,
    p25: f64,
    p75: f64,
    p95: f64,
    prob_loss: f64,
    var: f64,
    cvar: f64,
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    let idx = ((sorted.len() - 1) as f64 * pct).round() as usize;
    sorted[idx]
}

fn compute_stats(final_prices: &[f64], initial_price: f64, confidence: f64) -> Stats {
    let n = final_prices.len() as f64;
    let mean = final_prices.iter().sum::<f64>() / n;
    let variance = final_prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    let mut sorted = final_prices.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let median = percentile(&sorted, 0.50);
    let p05 = percentile(&sorted, 0.05);
    let p25 = percentile(&sorted, 0.25);
    let p75 = percentile(&sorted, 0.75);
    let p95 = percentile(&sorted, 0.95);

    let prob_loss = final_prices.iter().filter(|&&p| p < initial_price).count() as f64 / n;

    // VaR / CVaR on simple return, at the given confidence level (loss side).
    let mut returns: Vec<f64> = final_prices
        .iter()
        .map(|p| (p - initial_price) / initial_price)
        .collect();
    returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let tail_cutoff_idx = ((1.0 - confidence) * returns.len() as f64).ceil() as usize;
    let tail_cutoff_idx = tail_cutoff_idx.max(1).min(returns.len());
    let var = -returns[tail_cutoff_idx - 1]; // loss as a positive number
    let cvar = -(returns[..tail_cutoff_idx].iter().sum::<f64>() / tail_cutoff_idx as f64);

    Stats {
        mean,
        std_dev,
        min,
        max,
        median,
        p05,
        p25,
        p75,
        p95,
        prob_loss,
        var,
        cvar,
    }
}

fn write_final_prices_csv(path: &str, final_prices: &[f64]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    writeln!(w, "final_price")?;
    for p in final_prices {
        writeln!(w, "{:.6}", p)?;
    }
    Ok(())
}

fn write_paths_csv(path: &str, paths: &[Vec<f64>]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    let header: Vec<String> = (0..paths.len()).map(|i| format!("path_{i}")).collect();
    writeln!(w, "day,{}", header.join(","))?;

    let n_days = paths[0].len();
    for day in 0..n_days {
        let row: Vec<String> = paths.iter().map(|p| format!("{:.6}", p[day])).collect();
        writeln!(w, "{},{}", day, row.join(","))?;
    }
    Ok(())
}

fn main() {
    let args = Args::parse();

    if args.simulations == 0 {
        eprintln!("error: --simulations must be > 0");
        std::process::exit(1);
    }
    if args.sample_paths > args.simulations {
        eprintln!("error: --sample-paths cannot exceed --simulations");
        std::process::exit(1);
    }

    let dt = 1.0 / TRADING_DAYS_PER_YEAR;
    let normal = Normal::new(0.0, 1.0).unwrap();
    let base_seed = args.seed.unwrap_or_else(rand::random);
    let record_flags: Vec<bool> = (0..args.simulations)
        .map(|i| args.paths_output.is_some() && i < args.sample_paths)
        .collect();

    let start = std::time::Instant::now();

    let outcomes: Vec<SimOutcome> = (0..args.simulations)
        .into_par_iter()
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(i as u64));
            simulate_one_path(&args, dt, &normal, &mut rng, record_flags[i])
        })
        .collect();

    let elapsed = start.elapsed();

    let final_prices: Vec<f64> = outcomes.iter().map(|o| o.final_price).collect();
    let stats = compute_stats(&final_prices, args.initial_price, args.confidence);

    println!("Monte Carlo Stock Simulation (Geometric Brownian Motion)");
    println!("----------------------------------------------------------");
    println!("Initial price:      {:.2}", args.initial_price);
    println!(
        "Drift / Volatility:  {:.2}% / {:.2}% (annualized)",
        args.drift * 100.0,
        args.volatility * 100.0
    );
    println!(
        "Horizon:             {} trading days (~{:.2} years)",
        args.days,
        args.days as f64 / TRADING_DAYS_PER_YEAR
    );
    println!("Simulations:         {}", args.simulations);
    println!("Seed:                {}", base_seed);
    println!("Compute time:        {:.3}s", elapsed.as_secs_f64());
    println!();
    println!("Final price distribution");
    println!("----------------------------------------------------------");
    println!("Mean:                {:.4}", stats.mean);
    println!("Std dev:             {:.4}", stats.std_dev);
    println!("Min / Max:           {:.4} / {:.4}", stats.min, stats.max);
    println!(
        "P05 / P25 / Median / P75 / P95:  {:.4} / {:.4} / {:.4} / {:.4} / {:.4}",
        stats.p05, stats.p25, stats.median, stats.p75, stats.p95
    );
    println!("P(loss):             {:.2}%", stats.prob_loss * 100.0);
    println!(
        "VaR ({:.0}%):          {:.2}%  (loss on initial capital)",
        args.confidence * 100.0,
        stats.var * 100.0
    );
    println!(
        "CVaR / Expected Shortfall ({:.0}%): {:.2}%",
        args.confidence * 100.0,
        stats.cvar * 100.0
    );

    if let Some(out_path) = &args.output {
        match write_final_prices_csv(out_path, &final_prices) {
            Ok(()) => println!("\nFinal prices written to {out_path}"),
            Err(e) => eprintln!("\nfailed to write {out_path}: {e}"),
        }
    }

    if let Some(paths_path) = &args.paths_output {
        let sample_paths: Vec<Vec<f64>> = outcomes
            .iter()
            .filter_map(|o| o.path.clone())
            .take(args.sample_paths)
            .collect();
        if sample_paths.is_empty() {
            eprintln!("\nno sample paths recorded (check --sample-paths)");
        } else {
            match write_paths_csv(paths_path, &sample_paths) {
                Ok(()) => println!("Sample paths written to {paths_path}"),
                Err(e) => eprintln!("failed to write {paths_path}: {e}"),
            }
        }
    }
}