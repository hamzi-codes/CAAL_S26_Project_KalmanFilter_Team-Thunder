// arctan2_approx.cpp
// Team Thunder — Kalman Filter Milestone 2

#include "arctan2_approx.h"
#include "kalman_matrices.h"
#include <stdexcept>

// CORDIC lookup table: CORDIC_TABLE[i] = atan(2^{-i}) in radians
// These are the micro-rotation angles used by CORDIC algorithm.
static const double CORDIC_TABLE[CORDIC_ITERATIONS] = {
    0.7853981633974483,        // atan(2^0)
    0.4636476090008257,        // atan(2^-1)
    0.24497866312686414,       // atan(2^-2)
    0.12435499454676144,       // atan(2^-3)
    0.06241880999595735,       // atan(2^-4)
    0.031239833430268277,      // atan(2^-5)
    0.015623728620476831,      // atan(2^-6)
    0.007812341060101111,      // atan(2^-7)
    0.003906230131966972,      // atan(2^-8)
    0.0019531225164788188,     // atan(2^-9)
    0.0009765621895593195,     // atan(2^-10)
    0.0004882812111948983,     // atan(2^-11)
    0.00024414062014936177,    // atan(2^-12)
    0.00012207031189367021,    // atan(2^-13)
    0.000061035156174208773,   // atan(2^-14)
    0.000030517578115526096,   // atan(2^-15)
    0.000015258789062277982,   // atan(2^-16)
    0.0000076293945311369800,  // atan(2^-17)
    0.0000038146972656429850,  // atan(2^-18)
    0.0000019073486329385070,  // atan(2^-19)
    0.00000095367431640625,    // atan(2^-20)
    0.000000476837158203125,   // atan(2^-21)
    0.0000002384185791015625,  // atan(2^-22)
    0.00000011920928955078125  // atan(2^-23)
};

static const double PI_VALUE = 3.141592653589793;

double manual_pi() {
    
    // 4 * atan(1) is pi
    
    return 4.0 * CORDIC_TABLE[0];

}

// For manually squaring we use Newton-Raphson method
// Starting guess: x/2 (easily divisible and multiplicable), iterate 20 times for double precision
double manual_sqrt(double s) {
   
    if (s < 0.0)
       
        throw std::invalid_argument("manual_sqrt: negative input");

    if (s == 0.0)
        
        return 0.0;

    double x = s / 2.0;
    static const int MAX_ITER = 20;

    for (int i = 0; i < MAX_ITER; ++i) {
        
        double x_new = 0.5 * (x + s / x);
        double diff = x_new - x;
        
        if (diff < 0.0)
            
            diff = -diff;
        
        x = x_new;
        if (diff < 1e-15 * x) break;
    }
    return x;
}

// for cordic_atan we use the CORDIC vectoring mode
// 
// Input: vector (x, y) with x > 0
// Output: angle θ = atan(y/x) in (-π/2, π/2)
//
// Algorithm rotates vector onto x-axis, accumulating rotation angle.
// Sign chosen to drive y → 0: if y>0 rotate clockwise (subtract angle), else y<0 rotate counter-clockwise (add angle)
double cordic_atan(double y, double x) {
    
    double angle = 0.0;
    double power_of_2 = 1.0;   // 2^{-i}

    for (int i = 0; i < CORDIC_ITERATIONS; ++i) {
        
        double x_new, y_new;
        
        if (y > 0.0) {
            
            x_new = x + y * power_of_2;
            y_new = y - x * power_of_2;
            angle += CORDIC_TABLE[i];
        }
        else {
            
            x_new = x - y * power_of_2;
            y_new = y + x * power_of_2;
            angle -= CORDIC_TABLE[i];
        }
        x = x_new;
        y = y_new;
        power_of_2 *= 0.5;
    }
    return angle;
}

// for manual_atan2  full four-quadrant arctangent
// Uses CORDIC with quadrant correction.
// Special cases: 
// x=0,y>0 → +π/2; 
// x=0,y<0 → -π/2; 
// x=0,y=0 → 0; 
// x>0,y=0 → 0; 
// x<0,y=0 → +π
double manual_atan2(double y, double x) {
    
    static const double PI = PI_VALUE;
    static const double PI_2 = PI_VALUE / 2.0;
    static const double EPS = 1e-15;

    if (x * x + y * y < EPS * EPS) 
        
        return 0.0;

    if (x > -EPS && x < EPS)
        
        return (y >= 0.0) ? PI_2 : -PI_2;

    if (y > -EPS && y < EPS)
        
        return (x >= 0.0) ? 0.0 : PI;

    if (x > 0.0) {
        
        return cordic_atan(y, x);
    }
    else {
        
        double angle = cordic_atan(-y, -x);
        return (y >= 0.0) ? PI + angle : -PI + angle;
    }
}

// h_joint is the nonlinear measurement function for one joint
// Input: 12×1 state matrix [px vx ax jx | py vy ay jy | pz vz az jz]
// Output: 3×1 spherical [r, θ, φ] where:
//   
//   ρ = sqrt(px²+py²)
//   r = sqrt(px²+py²+pz²)  (Milestone 1, Eq.65)
//   θ = atan2(py, px)      (Milestone 1, Eq.66)
//   φ = atan2(pz, ρ)       (Milestone 1, Eq.67)
Matrix h_joint(const Matrix& x_joint) {
    
    if (x_joint.rows != N_JOINT || x_joint.cols != 1)
        
        throw std::invalid_argument("h_joint: input must be 12×1");

    double px = x_joint(0, 0);
    double py = x_joint(4, 0);
    double pz = x_joint(8, 0);

    double rho = manual_sqrt(px * px + py * py);
    double r = manual_sqrt(px * px + py * py + pz * pz);

    static const double EPS = 1e-9;
    
    if (r < EPS)
        
        return Matrix(3, 1);

    double theta = manual_atan2(py, px);
    double phi = manual_atan2(pz, rho);

    Matrix z(3, 1);
    z(0, 0) = r;
    z(1, 0) = theta;
    z(2, 0) = phi;
    
    return z;
}

// H_jacobian_joint is the Jacobian dh/dx for one joint (3×12)
// Derived in Milestone 1, Section VIII-D.
// Nonzero partials only for position components (cols 0,4,8).
// Singularities: ρ=0 (elevation) zeros θ,φ rows; r=0 zeros all rows.
Matrix H_jacobian_joint(const Matrix& x_joint) {
    
    if (x_joint.rows != N_JOINT || x_joint.cols != 1)
        
        throw std::invalid_argument("H_jacobian_joint: input must be 12×1");

    double px = x_joint(0, 0);
    double py = x_joint(4, 0);
    double pz = x_joint(8, 0);

    double rho2 = px * px + py * py;
    double r2 = rho2 + pz * pz;
    double rho = manual_sqrt(rho2);
    double r = manual_sqrt(r2);

    static const double EPS = 1e-9;

    Matrix Hj(3, N_JOINT);

    // Row 0: ∂r/∂px = px/r, ∂r/∂py = py/r, ∂r/∂pz = pz/r
    
    if (r > EPS) {
        
        Hj(0, 0) = px / r;
        Hj(0, 4) = py / r;
        Hj(0, 8) = pz / r;
    }

    // Row 1: ∂θ/∂px = -py/ρ², ∂θ/∂py = px/ρ², ∂θ/∂pz = 0
    
    if (rho > EPS) {
        
        Hj(1, 0) = -py / rho2;
        Hj(1, 4) = px / rho2;
        Hj(1, 8) = 0.0;
    }

    // Row 2: ∂φ/∂px = -(px·pz)/(ρ·r²), ∂φ/∂py = -(py·pz)/(ρ·r²), ∂φ/∂pz = ρ/r²
    
    if (rho > EPS && r > EPS) {
        
        double denom_xz = rho * r2;
        double denom_z = r2;

        Hj(2, 0) = -(px * pz) / denom_xz;
        Hj(2, 4) = -(py * pz) / denom_xz;
        Hj(2, 8) = rho / denom_z;
    }

    return Hj;
}

// h_full — nonlinear measurement for full body (69×1)
// Stacks h_joint results for all 23 joints.
// Joint j: state rows [j*12, j*12+12) → measurement rows [j*3, j*3+3)
Matrix h_full(const Matrix& x_full) {
   
    if (x_full.rows != N_STATE || x_full.cols != 1)
        
        throw std::invalid_argument("h_full: input must be 276×1");

    Matrix z(N_MEAS, 1);

    for (int j = 0; j < N_JOINTS; ++j) {
        
        Matrix x_j = x_full.get_block(j * N_JOINT, 0, N_JOINT, 1);
        Matrix z_j = h_joint(x_j);
        z.set_block(j * 3, 0, z_j);
    }
    return z;
}

// H_jacobian_full is the full-body Jacobian (69×276)
// Block diagonal: H_jacobian_joint for each joint.
// Joint j → rows [j*3, j*3+3), cols [j*12, j*12+12)
Matrix H_jacobian_full(const Matrix& x_full) {
    
    if (x_full.rows != N_STATE || x_full.cols != 1)
        
        throw std::invalid_argument("H_jacobian_full: input must be 276×1");

    Matrix Hk(N_MEAS, N_STATE);

    for (int j = 0; j < N_JOINTS; ++j) {
        
        Matrix x_j = x_full.get_block(j * N_JOINT, 0, N_JOINT, 1);
        Matrix Hj = H_jacobian_joint(x_j);
        Hk.set_block(j * 3, j * N_JOINT, Hj);
    }
    return Hk;
}
