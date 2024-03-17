use std::time::Instant;

use nalgebra::{Vector2, Vector3};
use sqpnp::types::SolverParameters;

fn main() {
    let pts3 = [
        Vector3::new(-0.429857595273321, -0.441798127281825, 0.714342354521372),
        Vector3::new(-2.1568268264648, 0.113521604867983, -0.148634122716948),
        Vector3::new(0.694636908485644, -0.737067927134015, -1.38877746946909),
        Vector3::new(-1.07051455287146, -1.2122304801284, -0.841002964233812),
        Vector3::new(0.509844073252947, -1.07097319594739, 0.675410167109412),
        Vector3::new(0.40951585099, 2.2300713816052, 0.365229861025625),
        Vector3::new(2.04320214188098, 1.11847674401846, 0.623432173763436),
    ];

    let pts2 = [
        // no noise
        Vector2::new(0.139024436737141, -0.00108631784422283),
        Vector2::new(0.149897105048989, 0.270584578309815),
        Vector2::new(-0.118448642309468, -0.0844116551810971),
        Vector2::new(0.0917181969674735, 0.0435196877212059),
        Vector2::new(0.100243308685939, -0.178506520365217),
        Vector2::new(-0.296312157121094, 0.220675975198136),
        Vector2::new(-0.331509880499455, -0.213091587841007),
    ];
    let start = Instant::now();
    let solver = sqpnp::PnpSolver::new(&pts3, &pts2, None, SolverParameters::default());

    let mut stop = Instant::now();
    if let Some(mut solver) = solver {
        solver.solve();
        stop = Instant::now();
        println!("SQPnP found {} solution(s)", solver.number_of_solutions());
        for i in 0..solver.number_of_solutions() {
            println!("Solution {i}");
            solver.solution_ptr(i).unwrap().print();
            println!(" Average squared projection error : {:e}", solver.average_squared_projection_errors()[i]);
        }
    }
    let duration = stop - start;
    println!("Time taken by SQPnP: {} microseconds", duration.as_micros());
}
