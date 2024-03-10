#![allow(non_snake_case)]

use nalgebra::{SMatrix, Vector2, Point3, Matrix3, Matrix3x1, Const};
use types::{_Projection, _Point, SolverParameters, SQPSolution, OmegaNullspaceMethod};

use crate::types::NearestRotationMethod;

pub mod types;
mod sqpnp;

pub const SQRT3: f64 = 1.732050807568877293527446341505872367_f64;

pub struct PnpSolver {
    projections_: Vec<_Projection>,
    points_: Vec<_Point>,
    weights_: Vec<f64>,
    parameters_: SolverParameters,

    Omega_: SMatrix<f64, 9, 9>,

    s_: SMatrix<f64, 9, 1>,
    U_: SMatrix<f64, 9, 9>,
    P_: SMatrix<f64, 3, 9>,
    point_mean_: Matrix3x1<f64>,

    num_null_vectors_: i32,
    solutions_: [SQPSolution; 18],
    num_solutions_: usize,
    NearestRotationMatrix: fn(&SMatrix<f64, 9, 1>, &mut SMatrix<f64, 9, 1>),
}

impl PnpSolver {
    pub fn Omega(&self) -> &SMatrix<f64, 9, 9> {
        &self.Omega_
    }
    pub fn EigenVectors(&self) -> &SMatrix<f64, 9, 9> {
        &self.U_
    }
    pub fn EigenValues(&self) -> &SMatrix<f64, 9, 1> {
        &self.s_
    }
    pub fn NullSpaceDimension(&self) -> i32 {
        self.num_null_vectors_
    }
    pub fn NumberOfSolutions(&self) -> usize {
        self.num_solutions_
    }
    pub fn SolutionPtr(&self, index: usize) -> Option<&SQPSolution> {
        if index >= self.num_solutions_ {
            None
        } else {
            Some(&self.solutions_[index])
        }
    }

    /// Return average reprojection errors
    pub fn AverageSquaredProjectionErrors(&self) -> Vec<f64> {
        (0..self.num_solutions_).map(|i| {
            self.AverageSquaredProjectionError(i)
        }).collect()
    }

    /// Constructor (initializes Omega, P and U, s, i.e. the decomposition of Omega)
    pub fn new(
        _3dpoints: &[Point3<f64>],
        _projections: &[Vector2<f64>],
        _weights: Vec<f64>,
        _parameters: SolverParameters,
    ) -> Option<Self> {
        let n = _3dpoints.len();
        if n != _projections.len() || n < 3 {
            return None;
        }

        let weights_ = if !_weights.is_empty() {
            if n != _weights.len() {
                return None;
            }
            _weights
        } else {
            vec![1.0; n]
        };

        let mut points_ = Vec::<_Point>::with_capacity(n);
        let mut projections_ = Vec::<_Projection>::with_capacity(n);
        let mut num_null_vectors_ = -1;
        let mut Omega_ = SMatrix::<f64, 9, 9>::zeros();
        let mut sum_wx @ mut sum_wy @ mut sum_wx2_plus_wy2 @ mut sum_w = 0.0;

        let mut sum_wX @ mut sum_wY @ mut sum_wZ = 0.0;

        let mut QA = SMatrix::<f64, 3, 9>::zeros();

        for i in 0..n {
            let w = weights_[i];
            points_.push(_3dpoints[i].coords);
            projections_.push(_projections[i]);

            if w == 0.0 {
                continue;
            }

            let proj_ = &projections_[i];
            let wx = proj_[0] * w;
            let wy = proj_[1] * w;
            let wsq_norm_m = w*proj_.norm_squared();
            sum_wx += wx;
            sum_wy += wy;
            sum_wx2_plus_wy2 += wsq_norm_m;
            sum_w += w;

            let pt_ = &points_[i];
            let X = pt_[0];
            let Y = pt_[1];
            let Z = pt_[2];
            let wX = w * X;
            let wY = w * Y;
            let wZ = w * Z;
            sum_wX += wX;
            sum_wY += wY;
            sum_wZ += wZ;

            // Accumulate Omega by kronecker( Qi, Mi*Mi' ) = A'*Qi*Ai. NOTE: Skipping block (3:5, 3:5) because it's same as (0:2, 0:2)
            let X2 = X*X;
            let XY = X*Y;
            let XZ = X*Z;
            let Y2 = Y*Y;
            let YZ = Y*Z;
            let Z2 = Z*Z;

            // a. Block (0:2, 0:2) populated by Mi*Mi'. NOTE: Only upper triangle
            Omega_[(0, 0)] += w*X2;
            Omega_[(0, 1)] += w*XY;
            Omega_[(0, 2)] += w*XZ;
            Omega_[(1, 1)] += w*Y2;
            Omega_[(1, 2)] += w*YZ;
            Omega_[(2, 2)] += w*Z2;

            // b. Block (0:2, 6:8) populated by -x*Mi*Mi'. NOTE: Only upper triangle
            Omega_[(0, 6)] += -wx*X2; Omega_[(0, 7)] += -wx*XY; Omega_[(0, 8)] += -wx*XZ;
                                      Omega_[(1, 7)] += -wx*Y2; Omega_[(1, 8)] += -wx*YZ;
                                                                Omega_[(2, 8)] += -wx*Z2;

            // c. Block (3:5, 6:8) populated by -y*Mi*Mi'. NOTE: Only upper triangle
            Omega_[(3, 6)] += -wy*X2; Omega_[(3, 7)] += -wy*XY; Omega_[(3, 8)] += -wy*XZ;
                                      Omega_[(4, 7)] += -wy*Y2; Omega_[(4, 8)] += -wy*YZ;
                                                                Omega_[(5, 8)] += -wy*Z2;
                                                            
            // d. Block (6:8, 6:8) populated by (x^2+y^2)*Mi*Mi'. NOTE: Only upper triangle
            Omega_[(6, 6)] += wsq_norm_m*X2; Omega_[(6, 7)] += wsq_norm_m*XY; Omega_[(6, 8)] += wsq_norm_m*XZ;
                                             Omega_[(7, 7)] += wsq_norm_m*Y2; Omega_[(7, 8)] += wsq_norm_m*YZ;
                                                                              Omega_[(8, 8)] += wsq_norm_m*Z2;

            // Accumulating Qi*Ai in QA.
            // Note that certain pairs of elements are equal, so we save some operations by filling them outside the loop
            QA[(0, 0)] += wX; QA[(0, 1)] += wY; QA[(0, 2)] += wZ;   QA[(0, 6)] += -wx*X; QA[(0, 7)] += -wx*Y; QA[(0, 8)] += -wx*Z;
            //QA[(1, 3)] += wX; QA[(1, 4)] += wY; QA[(1, 5)] += wZ;
                                                                    QA[(1, 6)] += -wy*X; QA[(1, 7)] += -wy*Y; QA[(1, 8)] += -wy*Z;
            
            //QA[(2, 0)] += -wx*X; QA[(2, 1)] += -wx*Y; QA[(2, 2)] += -wx*Z;     QA[(2, 3)] += -wy*X; QA[(2, 4)] += -wy*Y; QA[(2, 5)] += -wy*Z;
            QA[(2, 6)] += wsq_norm_m*X; QA[(2, 7)] += wsq_norm_m*Y; QA[(2, 8)] += wsq_norm_m*Z;
        }


        // Complete QA
        QA[(1, 3)] = QA[(0, 0)]; QA[(1, 4)] = QA[(0, 1)]; QA[(1, 5)] = QA[(0, 2)];
        QA[(2, 0)] = QA[(0, 6)]; QA[(2, 1)] = QA[(0, 7)]; QA[(2, 2)] = QA[(0, 8)];
        QA[(2, 3)] = QA[(1, 6)]; QA[(2, 4)] = QA[(1, 7)]; QA[(2, 5)] = QA[(1, 8)];

        // Fill-in lower triangles of off-diagonal blocks (0:2, 6:8), (3:5, 6:8) and (6:8, 6:8)
        Omega_[(1, 6)] = Omega_[(0, 7)]; Omega_[(2, 6)] = Omega_[(0, 8)]; Omega_[(2, 7)] = Omega_[(1, 8)];
        Omega_[(4, 6)] = Omega_[(3, 7)]; Omega_[(5, 6)] = Omega_[(3, 8)]; Omega_[(5, 7)] = Omega_[(4, 8)];
        Omega_[(7, 6)] = Omega_[(6, 7)]; Omega_[(8, 6)] = Omega_[(6, 8)]; Omega_[(8, 7)] = Omega_[(7, 8)];

        // Fill-in upper triangle of block (3:5, 3:5)
        Omega_[(3, 3)] = Omega_[(0, 0)]; Omega_[(3, 4)] = Omega_[(0, 1)]; Omega_[(3, 5)] = Omega_[(0, 2)];
        Omega_[(4, 4)] = Omega_[(1, 1)]; Omega_[(4, 5)] = Omega_[(1, 2)];
        Omega_[(5, 5)] = Omega_[(2, 2)];

        // Fill lower triangle of Omega; elements (7, 6), (8, 6) & (8, 7) have already been assigned above
        Omega_[(1, 0)] = Omega_[(0, 1)];
        Omega_[(2, 0)] = Omega_[(0, 2)]; Omega_[(2, 1)] = Omega_[(1, 2)];
        Omega_[(3, 0)] = Omega_[(0, 3)]; Omega_[(3, 1)] = Omega_[(1, 3)]; Omega_[(3, 2)] = Omega_[(2, 3)];
        Omega_[(4, 0)] = Omega_[(0, 4)]; Omega_[(4, 1)] = Omega_[(1, 4)]; Omega_[(4, 2)] = Omega_[(2, 4)]; Omega_[(4, 3)] = Omega_[(3, 4)];
        Omega_[(5, 0)] = Omega_[(0, 5)]; Omega_[(5, 1)] = Omega_[(1, 5)]; Omega_[(5, 2)] = Omega_[(2, 5)]; Omega_[(5, 3)] = Omega_[(3, 5)]; Omega_[(5, 4)] = Omega_[(4, 5)];
        Omega_[(6, 0)] = Omega_[(0, 6)]; Omega_[(6, 1)] = Omega_[(1, 6)]; Omega_[(6, 2)] = Omega_[(2, 6)]; Omega_[(6, 3)] = Omega_[(3, 6)]; Omega_[(6, 4)] = Omega_[(4, 6)]; Omega_[(6, 5)] = Omega_[(5, 6)];
        Omega_[(7, 0)] = Omega_[(0, 7)]; Omega_[(7, 1)] = Omega_[(1, 7)]; Omega_[(7, 2)] = Omega_[(2, 7)]; Omega_[(7, 3)] = Omega_[(3, 7)]; Omega_[(7, 4)] = Omega_[(4, 7)]; Omega_[(7, 5)] = Omega_[(5, 7)];
        Omega_[(8, 0)] = Omega_[(0, 8)]; Omega_[(8, 1)] = Omega_[(1, 8)]; Omega_[(8, 2)] = Omega_[(2, 8)]; Omega_[(8, 3)] = Omega_[(3, 8)]; Omega_[(8, 4)] = Omega_[(4, 8)]; Omega_[(8, 5)] = Omega_[(5, 8)];


        // Q = Sum( wi*Qi ) = Sum( [ wi, 0, -wi*xi; 0, 1, -wi*yi; -wi*xi, -wi*yi, wi*(xi^2 + yi^2)] )
        let Q = Matrix3::new(
            sum_w,     0.0,       -sum_wx,
            0.0,       sum_w,     -sum_wy,
            -sum_wx,   -sum_wy,   sum_wx2_plus_wy2
        );
      
        // Qinv = inv( Q ) = inv( Sum( Qi) )
        // let Qinv = Q.try_inverse().unwrap();
        let mut Qinv = SMatrix::<f64, 3, 3>::zeros();
        InvertSymmetric3x3(Q, &mut Qinv);

        // Compute P = -inv( Sum(wi*Qi) ) * Sum( wi*Qi*Ai ) = -Qinv * QA
        let P_ = -Qinv * QA;
        // Complete Omega (i.e., Omega = Sum(A'*Qi*A') + Sum(Qi*Ai)'*P = Sum(A'*Qi*A') + Sum(Qi*Ai)'*inv(Sum(Qi))*Sum( Qi*Ai) 
        Omega_ +=  QA.transpose()*P_;

        // Finally, decompose Omega with the chosen method
        let U_;
        let s_;
        match _parameters.omega_nullspace_method {
            OmegaNullspaceMethod::Rrqr => {
                // Rank revealing QR nullspace computation. This is slightly less accurate compared to SVD but x2 faster
                // Eigen::FullPivHouseholderQR<Eigen::Matrix<double, 9, 9> > rrqr(Omega_);
                // U_ = rrqr.matrixQ();
                //
                // Eigen::Matrix<double, 9, 9> R = rrqr.matrixQR().template triangularView<Eigen::Upper>();
                // s_ = R.diagonal().array().abs();
                unimplemented!();
            }
            OmegaNullspaceMethod::Svd => {
                // SVD-based nullspace computation. This is the most accurate but slowest option
                // Eigen::JacobiSVD<Eigen::Matrix<double, 9, 9>> svd(Omega_, Eigen::ComputeFullU);
                let svd = Omega_.svd(true, false);
                U_ = svd.u.unwrap();
                s_ = svd.singular_values;
            }
        }

        // Find dimension of null space; the check guards against overly large rank_tolerance
        while 7 - num_null_vectors_ >= 0 && s_[(7 - num_null_vectors_) as usize] < _parameters.rank_tolerance {
            num_null_vectors_ += 1;
        }

        // Dimension of null space of Omega must be <= 6
        if num_null_vectors_ > 6 {
            return None;
        }
        num_null_vectors_ += 1;

        // 3D point weighted mean (for quick cheirality checks)
        let inv_sum_w = 1.0 / sum_w;
        let point_mean_ = Matrix3x1::new(sum_wX*inv_sum_w, sum_wY*inv_sum_w, sum_wZ*inv_sum_w);

        // Assign nearest rotation method
        let NearestRotationMatrix = match _parameters.nearest_rotation_method {
            NearestRotationMethod::Foam => NearestRotationMatrix_FOAM,
            NearestRotationMethod::Svd => NearestRotationMatrix_SVD,
        };

        Some(Self {
            projections_,
            points_,
            weights_,
            parameters_: _parameters,
            Omega_,
            s_,
            U_,
            P_,
            point_mean_,
            num_null_vectors_,
            solutions_: Default::default(),
            num_solutions_: 0,
            NearestRotationMatrix,
        })
    }
}

impl PnpSolver {
    fn AverageSquaredProjectionError(&self, index: usize) -> f64 {
        let mut avg = 0.0;
        let r = &self.solutions_[index].r_hat;
        let t = &self.solutions_[index].t;

        for i in 0..self.points_.len() {
            let M = &self.points_[i];
            let Xc = r[0]*M[0] + r[1]*M[1] + r[2]*M[2] + t[0];
            let Yc = r[3]*M[0] + r[4]*M[1] + r[5]*M[2] + t[1];
            let inv_Zc = 1.0 / ( r[6]*M[0] + r[7]*M[1] + r[8]*M[2] + t[2] );

            let m = &self.projections_[i];
            let dx = Xc*inv_Zc - m[0];
            let dy = Yc*inv_Zc - m[1];
            avg += dx*dx + dy*dy;
        }

        return avg / self.points_.len() as f64;
    }

    /// Test cheirality on the mean point for a given solution
    fn TestPositiveDepth(&self, solution: &SQPSolution) -> bool {
        let r = &solution.r_hat;
        let t = &solution.t;
        let M = &self.point_mean_;
        return r[6]*M[0] + r[7]*M[1] + r[8]*M[2] + t[2] > 0.;
    }

    /// Test cheirality on the majority of points for a given solution
    fn TestPositiveMajorityDepths(&self, solution: &SQPSolution) -> bool {
        let r = &solution.r_hat;
        let t = &solution.t;
        let mut npos = 0;
        let mut nneg = 0;

        for i in 0..self.points_.len() {
            if self.weights_[i] == 0.0 { continue; }
            let M = &self.points_[i];
            if r[6]*M[0] + r[7]*M[1] + r[8]*M[2] + t[2] > 0. {
                npos += 1;
            } else {
                nneg += 1;
            }
        }

        return npos >= nneg;
    }
}


/// Determinant of 3x3 matrix stored as a 9x1 vector in *row-major* order
fn Determinant9x1(r: &SMatrix<f64, 9, 1>) -> f64 {
    r[0]*r[4]*r[8] + r[1]*r[5]*r[6] + r[2]*r[3]*r[7] - r[6]*r[4]*r[2] - r[7]*r[5]*r[0] - r[8]*r[3]*r[1]
}


/// Invert a 3x3 symmetric matrix (using low triangle values only)
fn InvertSymmetric3x3(Q: SMatrix<f64, 3, 3>,
    Qinv: &mut SMatrix<f64, 3, 3>,
) -> bool {
    let det_threshold = 1e-10;
    // 1. Get the elements of the matrix
    let (a, b, c, d, e, f);
    a = Q[(0, 0)];
    b = Q[(1, 0)]; d = Q[(1, 1)];
    c = Q[(2, 0)]; e = Q[(2, 1)]; f = Q[(2, 2)];

    // 2. Determinant
    let (t2, t4, t7, t9, t12);
    t2 = e*e;
    t4 = a*d;
    t7 = b*b;
    t9 = b*c;
    t12 = c*c;
    let det = -t4*f+a*t2+t7*f-2.0*t9*e+t12*d;
  
    // TODO nalgebra doesn't have complete orthogonal decomposition
    // if det.abs() < det_threshold { *Qinv=Q.completeOrthogonalDecomposition().pseudoInverse(); return false; } // fall back to pseudoinverse
    if det.abs() < det_threshold { *Qinv=Q.pseudo_inverse(1e-10).unwrap(); return false; } // fall back to pseudoinverse

    // 3. Inverse
    let (t15, t20, t24, t30);
    t15 = 1.0/det;
    t20 = (-b*f+c*e)*t15;
    t24 = (b*e-c*d)*t15;
    t30 = (a*e-t9)*t15;
    Qinv[(0, 0)] = (-d*f+t2)*t15;
    Qinv[(0, 1)] = -t20;
    Qinv[(1, 0)] = -t20;
    Qinv[(0, 2)] = -t24;
    Qinv[(2, 0)] = -t24;
    Qinv[(1, 1)] = -(a*f-t12)*t15;
    Qinv[(1, 2)] = t30;
    Qinv[(2, 1)] = t30;
    Qinv[(2, 2)] = -(t4-t7)*t15;

    true
}

/// Simple SVD - based nearest rotation matrix. Argument should be a *row-major* matrix representation.
/// Returns a row-major vector representation of the nearest rotation matrix.
fn NearestRotationMatrix_SVD(e: &SMatrix<f64, 9, 1>, r: &mut SMatrix<f64, 9, 1>) {
    let E = e.reshape_generic(Const::<3>, Const::<3>);
    let svd = E.svd(true, true);
    let detUV = svd.u.unwrap().determinant() * svd.v_t.unwrap().determinant();
    // so we return back a row-major vector representation of the orthogonal matrix
    let R = svd.u.unwrap() * Matrix3::from_partial_diagonal(&[1., 1., detUV]) * svd.v_t.unwrap().transpose();
    *r = R.reshape_generic(Const, Const);
}


/// Faster nearest rotation computation based on FOAM. See M. Lourakis: "An Efficient Solution to Absolute Orientation", ICPR 2016
/// and M. Lourakis, G. Terzakis: "Efficient Absolute Orientation Revisited", IROS 2018.
///
/// Solve the nearest orthogonal approximation problem
/// i.e., given B, find R minimizing ||R-B||_F
/// 
/// The computation borrows from Markley's FOAM algorithm
/// "Attitude Determination Using Vector Observations: A Fast Optimal Matrix Algorithm", J. Astronaut. Sci. 1993.
/// 
///  Copyright (C) 2019 Manolis Lourakis (lourakis **at** ics forth gr)
///  Institute of Computer Science, Foundation for Research & Technology - Hellas
///  Heraklion, Crete, Greece.
/// 
fn NearestRotationMatrix_FOAM(e: &SMatrix<f64, 9, 1>, r: &mut SMatrix<f64, 9, 1>) {
    let mut i;
    let B = e;
    let (mut l, mut lprev, detB, Bsq, adjBsq);
    let mut adjB = [0.0; 9];

    // det(B)
    detB=B[0]*B[4]*B[8] - B[0]*B[5]*B[7] - B[1]*B[3]*B[8] + B[2]*B[3]*B[7] + B[1]*B[6]*B[5] - B[2]*B[6]*B[4];
    if detB.abs() < 1E-04 { // singular, let SVD handle it
        NearestRotationMatrix_SVD(e, r);
        return;
    }

    // B's adjoint
    adjB[0]=B[4]*B[8] - B[5]*B[7]; adjB[1]=B[2]*B[7] - B[1]*B[8]; adjB[2]=B[1]*B[5] - B[2]*B[4];
    adjB[3]=B[5]*B[6] - B[3]*B[8]; adjB[4]=B[0]*B[8] - B[2]*B[6]; adjB[5]=B[2]*B[3] - B[0]*B[5];
    adjB[6]=B[3]*B[7] - B[4]*B[6]; adjB[7]=B[1]*B[6] - B[0]*B[7]; adjB[8]=B[0]*B[4] - B[1]*B[3];

    // ||B||^2, ||adj(B)||^2
    Bsq=B[0]*B[0]+B[1]*B[1]+B[2]*B[2] + B[3]*B[3]+B[4]*B[4]+B[5]*B[5] + B[6]*B[6]+B[7]*B[7]+B[8]*B[8];
    adjBsq=adjB[0]*adjB[0]+adjB[1]*adjB[1]+adjB[2]*adjB[2] + adjB[3]*adjB[3]+adjB[4]*adjB[4]+adjB[5]*adjB[5] + adjB[6]*adjB[6]+adjB[7]*adjB[7]+adjB[8]*adjB[8];

    // compute l_max with Newton-Raphson from FOAM's characteristic polynomial, i.e. eq.(23) - (26)
    l=0.5*(Bsq + 3.0); // 1/2*(trace(B*B') + trace(eye(3)))
    if detB<0.0 { l=-l; } // trB & detB have opposite signs!
    // for(i=15, lprev=0.0; fabs(l-lprev)>1E-12*fabs(lprev) && i>0; --i){
    i = 15;
    lprev = 0.0;
    while (l-lprev).abs() > 1E-12*lprev.abs() && i>0 {
        let (tmp, p, pp);

        tmp=l*l-Bsq;
        p=tmp*tmp - 8.0*l*detB - 4.0*adjBsq;
        pp=8.0*(0.5*tmp*l - detB);

        lprev=l;
        l-=p/pp;

        i -= 1;
    }

    // the rotation matrix equals ((l^2 + Bsq)*B + 2*l*adj(B') - 2*B*B'*B) / (l*(l*l-Bsq) - 2*det(B)), i.e. eq.(14) using (18), (19)
    {
        // compute (l^2 + Bsq)*B
        let mut tmp = [0.0; 9];
        let mut BBt = [0.0; 9];
        let mut denom;
        let a=l*l + Bsq;

        // BBt=B*B'
        BBt[0]=B[0]*B[0] + B[1]*B[1] + B[2]*B[2];
        BBt[1]=B[0]*B[3] + B[1]*B[4] + B[2]*B[5];
        BBt[2]=B[0]*B[6] + B[1]*B[7] + B[2]*B[8];

        BBt[3]=BBt[1];
        BBt[4]=B[3]*B[3] + B[4]*B[4] + B[5]*B[5];
        BBt[5]=B[3]*B[6] + B[4]*B[7] + B[5]*B[8];

        BBt[6]=BBt[2];
        BBt[7]=BBt[5];
        BBt[8]=B[6]*B[6] + B[7]*B[7] + B[8]*B[8];

        // tmp=BBt*B
        tmp[0]=BBt[0]*B[0] + BBt[1]*B[3] + BBt[2]*B[6];
        tmp[1]=BBt[0]*B[1] + BBt[1]*B[4] + BBt[2]*B[7];
        tmp[2]=BBt[0]*B[2] + BBt[1]*B[5] + BBt[2]*B[8];

        tmp[3]=BBt[3]*B[0] + BBt[4]*B[3] + BBt[5]*B[6];
        tmp[4]=BBt[3]*B[1] + BBt[4]*B[4] + BBt[5]*B[7];
        tmp[5]=BBt[3]*B[2] + BBt[4]*B[5] + BBt[5]*B[8];

        tmp[6]=BBt[6]*B[0] + BBt[7]*B[3] + BBt[8]*B[6];
        tmp[7]=BBt[6]*B[1] + BBt[7]*B[4] + BBt[8]*B[7];
        tmp[8]=BBt[6]*B[2] + BBt[7]*B[5] + BBt[8]*B[8];

        // compute R as (a*B + 2*(l*adj(B)' - tmp))*denom; note that adj(B')=adj(B)'
        denom=l*(l*l-Bsq) - 2.0*detB;
        denom=1.0/denom;
        r[0]=(a*B[0] + 2.0*(l*adjB[0] - tmp[0]))*denom;
        r[1]=(a*B[1] + 2.0*(l*adjB[3] - tmp[1]))*denom;
        r[2]=(a*B[2] + 2.0*(l*adjB[6] - tmp[2]))*denom;

        r[3]=(a*B[3] + 2.0*(l*adjB[1] - tmp[3]))*denom;
        r[4]=(a*B[4] + 2.0*(l*adjB[4] - tmp[4]))*denom;
        r[5]=(a*B[5] + 2.0*(l*adjB[7] - tmp[5]))*denom;

        r[6]=(a*B[6] + 2.0*(l*adjB[2] - tmp[6]))*denom;
        r[7]=(a*B[7] + 2.0*(l*adjB[5] - tmp[7]))*denom;
        r[8]=(a*B[8] + 2.0*(l*adjB[8] - tmp[8]))*denom;
    }

    //double R[9];
    //r=Eigen::Map<Eigen::Matrix<double, 9, 1>>(R);
}


/// Produce a distance from being orthogonal for a random 3x3 matrix
/// Matrix is provided as a vector
fn OrthogonalityError(a: &SMatrix<f64, 9, 1>) -> f64 {
    let sq_norm_a1 = a[0]*a[0] + a[1]*a[1] + a[2]*a[2];
    let sq_norm_a2 = a[3]*a[3] + a[4]*a[4] + a[5]*a[5];
    let sq_norm_a3 = a[6]*a[6] + a[7]*a[7] + a[8]*a[8];
    let dot_a1a2 = a[0]*a[3] + a[1]*a[4] + a[2]*a[5];
    let dot_a1a3 = a[0]*a[6] + a[1]*a[7] + a[2]*a[8];
    let dot_a2a3 = a[3]*a[6] + a[4]*a[7] + a[5]*a[8];

    return (sq_norm_a1 - 1.)*(sq_norm_a1 - 1.) + (sq_norm_a2 - 1.)*(sq_norm_a2 - 1.) + (sq_norm_a3 - 1.)*(sq_norm_a3 - 1.) +
    2.*( dot_a1a2*dot_a1a2 + dot_a1a3*dot_a1a3 + dot_a2a3*dot_a2a3 );
}
