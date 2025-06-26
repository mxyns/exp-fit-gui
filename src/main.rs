use argmin::core::{CostFunction, Error, Executor, State};
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use rand::{Rng, RngCore, SeedableRng};
use std::collections::HashMap;

/// Model reconstruction
fn reconstruct_function(xs: &[f64], c: f64, x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<f64> {
    let dx = x1 - x0;
    let denom = (c * dx).exp() - 1.0;
    if !denom.is_finite() || denom.abs() < 1e-12 {
        return vec![y0; xs.len()];
    }
    let b = (y1 - y0) / denom;
    let a = -b;
    xs.iter()
        .map(|&x| a + b * (c * (x - x0)).exp() + y0)
        .collect()
}

/// Adds uniform noise to a vector
fn add_noise(y: &[f64], noise_level: f64, noise_seed: Option<u64>) -> Vec<f64> {
    let mut rng = match noise_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_os_rng(),
    };

    y.iter()
        .map(|&yi| yi + noise_level * (2.0 * rng.random::<f64>() - 1.0))
        .collect()
}

/// Compute RMSE
fn compute_rmse(a: &[f64], b: &[f64]) -> f64 {
    let mse = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64;
    mse.sqrt()
}

// ----------------- Fitters -----------------

/// Safe bounded grid search
fn fit_c_grid(xs: &[f64], ys: &[f64], x0: f64, y0: f64, x1: f64, y1: f64) -> Option<f64> {
    let mut best = None;
    let mut best_err = f64::INFINITY;
    let start = -100.0;
    let end = 100.0;
    let n = 1000;

    for i in 0..(n + 1) {
        let c = start + (end - start) * (i as f64 / n as f64);
        let y_fit = reconstruct_function(xs, c, x0, y0, x1, y1);
        let err = compute_rmse(&y_fit, ys);
        if err < best_err {
            best_err = err;
            best = Some(c);
        }
    }
    best
}

/// Safe bounded grid search
fn fit_c_grid_trace(xs: &[f64], ys: &[f64], x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<[f64; 2]> {
    let start = -10.0;
    let end = 10.0;
    let n = 5000;
    let mut trace = Vec::new();

    for i in 0..(n + 1) {
        let c = start + (end - start) * (i as f64 / n as f64);
        let y_fit = reconstruct_function(xs, c, x0, y0, x1, y1);
        let err = compute_rmse(&y_fit, ys);
        trace.push([c, err])
    }

    trace
}

/// Nelder-Mead via argmin
use argmin::solver::neldermead::NelderMead;
use rand::prelude::StdRng;

struct FitCOp {
    xs: Vec<f64>,
    ys: Vec<f64>,
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
}

impl CostFunction for FitCOp {
    type Param = f64;
    type Output = f64;

    fn cost(&self, c: &Self::Param) -> Result<Self::Output, Error> {
        let pred = reconstruct_function(&self.xs, *c, self.x0, self.y0, self.x1, self.y1);
        let err = compute_rmse(&pred, &self.ys).powi(2);
        Ok(err)
    }
}

fn fit_c_nelder(xs: &[f64], ys: &[f64], x0: f64, y0: f64, x1: f64, y1: f64) -> Option<f64> {
    let operator = FitCOp {
        xs: xs.to_vec(),
        ys: ys.to_vec(),
        x0,
        y0,
        x1,
        y1,
    };
    let solver = NelderMead::new(vec![0.5, 1.0]);
    let res = Executor::new(operator, solver).run();
    res.ok()
        .map(|r| *r.state().get_param().expect("Do not fail"))
}

/// Simple particle swarm optimizer (1D)
/// TODO use argmin swarm optimizer
fn fit_c_swarm(xs: &[f64], ys: &[f64], x0: f64, y0: f64, x1: f64, y1: f64) -> Option<f64> {
    let mut rng = rand::rng();
    let swarm_size = 20;
    let mut particles =
        vec![(rng.random_range(-3.0..3.0), rng.random_range(-3.0..3.0)); swarm_size];
    let mut best = None;
    let mut best_err = f64::INFINITY;
    for _ in 0..50 {
        for p in &mut particles {
            let c = p.0;
            let fit = reconstruct_function(xs, c, x0, y0, x1, y1);
            let err: f64 = ys.iter().zip(&fit).map(|(y, p)| (y - p).powi(2)).sum();
            if err < best_err {
                best_err = err;
                best = Some(c);
            }
            let vel = p.1 + 0.5 * rng.random_range(0.0..1.0) * (best.unwrap() - p.0);
            p.0 += vel;
            p.1 = vel;
            p.0 = p.0.clamp(-5.0, 5.0);
        }
    }
    best
}

// ----------------- GUI -----------------

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
enum OptimizerType {
    NelderMead,
    Grid,
    Swarm,
}

impl OptimizerType {
    fn all() -> &'static [OptimizerType] {
        &[Self::NelderMead, Self::Grid, Self::Swarm]
    }
    fn name(&self) -> &'static str {
        match self {
            Self::NelderMead => "Nelder-Mead",
            Self::Grid => "Grid Search",
            Self::Swarm => "Swarm",
        }
    }
}

struct MyApp {
    // model
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    true_c: f64,
    noise: f64,
    noise_seed: u64,
    n: usize,
    auto_fit: bool,
    plot_cost_function: bool,
    recompute_requested: bool,
    sel: HashMap<OptimizerType, bool>,
    xs: Vec<f64>,
    ys_true: Vec<f64>,
    ys_noisy: Vec<f64>,
    cs: HashMap<OptimizerType, f64>,
    fits: HashMap<OptimizerType, Vec<f64>>,
    rmses: HashMap<OptimizerType, f64>,
    cost_function_trace: Vec<[f64; 2]>,
}

impl Default for MyApp {
    fn default() -> Self {
        let mut sel = HashMap::new();
        for &o in OptimizerType::all() {
            sel.insert(o, false);
        }
        Self {
            x0: 0.0,
            y0: 0.0,
            x1: 5.0,
            y1: 10.0,
            true_c: 0.7,
            noise: 0.1,
            noise_seed: 42,
            n: 100,
            auto_fit: false,
            plot_cost_function: false,
            recompute_requested: true,
            sel,
            xs: vec![],
            ys_true: vec![],
            ys_noisy: vec![],
            cs: HashMap::new(),
            fits: HashMap::new(),
            rmses: HashMap::new(),
            cost_function_trace: vec![],
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        egui::SidePanel::left("ctl").show(ctx, |ui| {
            ui.heading("Params");
            ui.add(egui::Slider::new(&mut self.x0, -10.0..=10.0).text("x₀"));
            ui.add(egui::Slider::new(&mut self.y0, -10.0..=10.0).text("y₀"));
            ui.add(egui::Slider::new(&mut self.x1, self.x0 + 0.1..=20.0).text("x₁"));
            ui.add(egui::Slider::new(&mut self.y1, -10.0..=20.0).text("y₁"));
            ui.separator();
            ui.add(
                egui::Slider::new(&mut self.true_c, -100.0..=100.0)
                    .text("True C")
                    .step_by(0.001)
                    .drag_value_speed(0.005)
                    .fixed_decimals(3),
            );
            ui.add(egui::Slider::new(&mut self.noise, 0.0..=2.0).text("Noise"));
            ui.add(egui::Slider::new(&mut self.n, 10..=2000).text("n pts"));
            ui.separator();
            ui.checkbox(&mut self.auto_fit, "Auto Fit");
            ui.checkbox(&mut self.plot_cost_function, "Plot Cost Function");
            ui.label("Optimizers:");
            for &o in OptimizerType::all() {
                ui.checkbox(self.sel.get_mut(&o).unwrap(), o.name());
            }
            ui.separator();
            if ui.button("Recompute").clicked() {
                self.recompute_requested = true;
            }
            if ui.button("New noise").clicked() {
                self.noise_seed = rand::rng().next_u64();
                self.recompute_requested = true;
            }

            // After plotting, display a table with results
            ui.separator();
            ui.heading("Fit Results");

            egui::Grid::new("results_table")
                .striped(true)
                .min_col_width(100.0)
                .show(ui, |ui| {
                    ui.label("Optimizer");
                    ui.label("Estimated C");
                    ui.label("RMSE");
                    ui.end_row();

                    for &opt in OptimizerType::all() {
                        if !self.sel[&opt] {
                            continue;
                        }

                        let name = opt.name().to_string();
                        let c = self.cs.get(&opt).copied().unwrap_or_default();
                        let rmse = self.rmses.get(&opt).copied().unwrap_or_default();

                        ui.label(name);
                        ui.label(format!("{:.6}", c));
                        ui.label(format!("{:.6}", rmse));
                        ui.end_row();
                    }
                });
        });

        if self.auto_fit {
            self.recompute_requested = true;
        }

        if self.recompute_requested {
            self.recompute_requested = false;
            self.xs = (0..self.n)
                .map(|i| self.x0 + i as f64 * (self.x1 - self.x0) / (self.n - 1) as f64)
                .collect();
            self.ys_true =
                reconstruct_function(&self.xs, self.true_c, self.x0, self.y0, self.x1, self.y1);
            self.ys_noisy = add_noise(&self.ys_true, self.noise, Some(self.noise_seed));
            self.cs.clear();
            self.fits.clear();
            self.rmses.clear();
            for &o in OptimizerType::all() {
                if !self.sel[&o] {
                    continue;
                }
                let c_opt = match o {
                    OptimizerType::NelderMead => {
                        fit_c_nelder(&self.xs, &self.ys_noisy, self.x0, self.y0, self.x1, self.y1)
                            .unwrap_or(f64::NAN)
                    }
                    OptimizerType::Grid => {
                        self.cost_function_trace = fit_c_grid_trace(
                            &self.xs,
                            &self.ys_noisy,
                            self.x0,
                            self.y0,
                            self.x1,
                            self.y1,
                        );
                        let [c, _cost] = self
                            .cost_function_trace
                            .iter()
                            .reduce(|min, curr| if curr[1] < min[1] { curr } else { min })
                            .unwrap_or(&[f64::NAN, f64::NAN]);
                        *c
                    }
                    OptimizerType::Swarm => {
                        fit_c_swarm(&self.xs, &self.ys_noisy, self.x0, self.y0, self.x1, self.y1)
                            .unwrap_or(f64::NAN)
                    }
                };
                let f = reconstruct_function(&self.xs, c_opt, self.x0, self.y0, self.x1, self.y1);
                let rmse = compute_rmse(&f, &self.ys_noisy);
                self.cs.insert(o, c_opt);
                self.fits.insert(o, f);
                self.rmses.insert(o, rmse);
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Fits");
            Plot::new("Exponential Fit Plot")
                .legend(egui_plot::Legend::default())
                .show(ui, |pu| {
                    let to_points = |ys: &[f64]| {
                        self.xs
                            .iter()
                            .zip(ys)
                            .map(|(x, y)| [*x, *y])
                            .collect::<PlotPoints>()
                    };

                    let pts_true = to_points(&self.ys_true);
                    let pts_noisy = to_points(&self.ys_noisy);
                    pu.line(Line::new("", pts_true).name("True"));
                    pu.line(Line::new("", pts_noisy).name("Noisy"));

                    if self.plot_cost_function {
                        let pts_cost_trace = self
                            .cost_function_trace
                            .iter()
                            .copied()
                            .collect::<PlotPoints>();

                        pu.line(Line::new("", pts_cost_trace).name("Cost Trace"));
                    }

                    for (opt, fit) in self.fits.iter() {
                        let pts_fit: PlotPoints =
                            self.xs.iter().zip(fit).map(|(x, y)| [*x, *y]).collect();
                        pu.line(Line::new("", pts_fit).name(format!("Fit with {}", opt.name())));
                    }
                });
        });
    }
}

pub(crate) fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Curve Fit",
        options,
        Box::new(|_cc| Ok(Box::new(MyApp::default()))),
    )
}
