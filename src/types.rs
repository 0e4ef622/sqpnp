use nalgebra::{Matrix2x1, Matrix3x1, SMatrix};

pub type _Projection = Matrix2x1<f64>;
pub type _Point = Matrix3x1<f64>;

pub enum OmegaNullspaceMethod {
    Rrqr,
    Svd,
}

pub enum NearestRotationMethod {
    Foam,
    Svd,
}

pub struct SolverParameters {
    pub rank_tolerance: f64,
    pub sqp_squared_tolerance: f64,
    pub sqp_det_threshold: f64,
    pub sqp_max_iteration: i32,
    pub omega_nullspace_method: OmegaNullspaceMethod,
    pub nearest_rotation_method: NearestRotationMethod,
    pub orthogonality_squared_error_threshold: f64,
    pub equal_vectors_squared_diff: f64,
    pub equal_squared_errors_diff: f64,
    pub point_variance_threshold: f64,
}

impl SolverParameters {
    pub const DEFAULT_RANK_TOLERANCE: f64 = 1e-7;
    pub const DEFAULT_SQP_SQUARED_TOLERANCE: f64 = 1e-10;
    pub const DEFAULT_SQP_DET_THRESHOLD: f64 = 1.001;
    pub const DEFAULT_SQP_MAX_ITERATION: i32 = 15;
    pub const DEFAULT_OMEGA_NULLSPACE_METHOD: OmegaNullspaceMethod = OmegaNullspaceMethod::Svd; // Originally Rrqr which isn't implemented
    pub const DEFAULT_NEAREST_ROTATION_METHOD: NearestRotationMethod = NearestRotationMethod::Foam;
    pub const DEFAULT_ORTHOGONALITY_SQUARED_ERROR_THRESHOLD: f64 = 1e-8;
    pub const DEFAULT_EQUAL_VECTORS_SQUARED_DIFF: f64 = 1e-10;
    pub const DEFAULT_EQUAL_SQUARED_ERRORS_DIFF: f64 = 1e-6;
    pub const DEFAULT_POINT_VARIANCE_THRESHOLD: f64 = 1e-5;
}

#[derive(Default)]
pub struct SQPSolution {
    /// Actual matrix upon convergence
    pub r: SMatrix<f64, 9, 1>,
    /// "Clean" (nearest) rotation matrix
    pub r_hat: SMatrix<f64, 9, 1>,
    pub t: Matrix3x1<f64>,
    pub num_iterations: i32,
    pub sq_error: f64,
}

impl SQPSolution {
    pub fn print(&self) {
        println!("r_hat: {:?}", self.r_hat);
        println!("t: {:?}", self.t);
        println!("squared error: {:.5}", self.sq_error);
        println!("number of SQP iterations: {}", self.num_iterations);
    }
}
