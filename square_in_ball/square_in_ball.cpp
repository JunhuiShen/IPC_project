#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>

using namespace Eigen;
using namespace std;

// Constants
const int numPoints = 4;
const double dt = 0.033;
const double gravity = 9.81;
//const double gravity = 0.0;
const double totalMass = 0.05;
const double mass = totalMass / numPoints; // Mass per vertex
const int maxNewtonIters = 300;
const double newtonTol = 1e-10;

const Vector2d circleCenter(0, 1.5);
const double circleRadius = 1.5;

// ---------- Utility Functions ----------

// 2D rotation matrix
Matrix2d rotationMatrix(double theta) {
    double c = cos(theta), s = sin(theta);
    Matrix2d R;
    R << c, -s,
            s,  c;
    return R;
}

// First derivative of rotation matrix
Matrix2d rotationMatrixDeriv(double theta) {
    double c = cos(theta), s = sin(theta);
    Matrix2d Rprime;
    Rprime << -s, -c,
            c, -s;
    return Rprime;
}

// Second derivative of rotation matrix
Matrix2d rotationMatrixSecondDeriv(double theta) {
    double c = cos(theta), s = sin(theta);
    Matrix2d Rpp;
    Rpp << -c,  s,
            -s, -c;
    return Rpp;
}

// ---------- Barrier Energy Functions ----------

// Taylor-based barrier function b(d)
double barrier(double d, double hat_d, int barrier_taylor_N) {
    if (d < hat_d) {
        double x = d - hat_d;
        double result = 0.0;
        for (int n = 1; n <= barrier_taylor_N; ++n) {
            double sign = (n % 2 == 0) ? 1.0 : -1.0;
            double coeff = sign / (n * std::pow(hat_d, n));
            result += coeff * std::pow(x, n + 2);
        }
        return result;
    }
    return 0.0;
}

// Taylor-based barrier gradient b'(d)
double db_dd(double d, double hat_d, int barrier_taylor_N) {
    if (d < hat_d) {
        double x = d - hat_d;
        double result = 0.0;
        for (int n = 1; n <= barrier_taylor_N; ++n) {
            double sign = (n % 2 == 0) ? 1.0 : -1.0;
            double coeff = sign * (n + 2) / (n * std::pow(hat_d, n));
            result += coeff * std::pow(x, n + 1);
        }
        return result;
    }
    return 0.0;
}

// Taylor-based barrier Hessian b''(d)
double d2b_dd2(double d, double hat_d, int barrier_taylor_N) {
    if (d < hat_d) {
        double x = d - hat_d;
        double result = 0.0;
        for (int n = 1; n <= barrier_taylor_N; ++n) {
            double sign = (n % 2 == 0) ? 1.0 : -1.0;
            double coeff = sign * (n + 2) * (n + 1) / (n * std::pow(hat_d, n));
            result += coeff * std::pow(x, n);
        }
        return result;
    }
    return 0.0;
}

// ---------- Energy Evaluation ----------

// Potential energy: barrier function + gravity  where b(d) = sum_{i=1}^{4} b(y_{i2} (theta, c)
double potentialEnergy(const MatrixXd &y, double hat_d, int barrier_taylor_N) {
    double PE = 0;
    for (int i = 0; i < numPoints; ++i) {
        Vector2d yi = y.row(i).transpose();                       // 2D position of vertex i
        double dist_to_center = (yi - circleCenter).norm();       // ||x - center||
        double distance = circleRadius - dist_to_center;                 // d = R - ||x - center||
        PE += barrier(distance, hat_d, barrier_taylor_N); // Barrier energy
        PE += mass * gravity * y(i, 1); // Gravitational potential
    }
    return PE;
}

// Gradient of potential energy
MatrixXd potentialEnergyGrad(const MatrixXd &y, double hat_d, int barrier_taylor_N) {
    MatrixXd grad = MatrixXd::Zero(numPoints, 2);
    for (int i = 0; i < numPoints; ++i) {
        Vector2d yi = y.row(i).transpose();
        Vector2d r = yi - circleCenter;
        double norm_r = r.norm();
        double d = circleRadius - norm_r;

        if (d < hat_d ) {
            Vector2d barrier_grad = -db_dd(d, hat_d, barrier_taylor_N) * (r / norm_r);
            grad.row(i) += barrier_grad.transpose();
        }

        grad(i, 1) += mass * gravity;  // Add gravity only to y-direction
    }
    return grad;
}

// Hessian of the potential energy
MatrixXd potentialEnergyHessian(const MatrixXd &y, double hat_d, int barrier_taylor_N) {
    MatrixXd H = MatrixXd::Zero(2 * numPoints, 2 * numPoints);

    for (int i = 0; i < numPoints; ++i) {
        Vector2d yi = y.row(i).transpose();
        Vector2d r = yi - circleCenter;
        double norm_r = r.norm();
        double d = circleRadius - norm_r;

        if (d < hat_d && norm_r > 1e-12) {
            Matrix2d I = Matrix2d::Identity();
            Matrix2d outer = r * r.transpose();

            Matrix2d H_i =
                    d2b_dd2(d, hat_d, barrier_taylor_N) * (outer / (norm_r * norm_r)) +
                    db_dd(d, hat_d, barrier_taylor_N) * ((I / norm_r) - (outer / std::pow(norm_r, 3)));

            int row = 2 * i;
            H.block<2, 2>(row, row) += H_i;
        }
    }

    return H;
}

// ---------- Reduced Energy and Derivatives ----------

// Compute y(theta, c): y_i = R(theta) (x_i - x_hat) + c
MatrixXd computeY(const MatrixXd &x_ref, const Vector2d &x_hat_ref, double theta, const Vector2d &c) {
    Matrix2d R = rotationMatrix(theta);
    MatrixXd y(numPoints, 2);
    for (int i = 0; i < numPoints; ++i) {
        Vector2d dx = x_ref.row(i).transpose() - x_hat_ref; // x_q - x_hat
        Vector2d y_i = R * dx + c;  // y_q = R * dx + c
        y.row(i) = y_i.transpose();
    }
    return y;
}

// Energy functional
double computePsi(const MatrixXd &y, const MatrixXd &x_hat, double hat_d, int N) {
    return 0.5 * mass * ((y - x_hat).squaredNorm())  + dt * dt * potentialEnergy(y, hat_d, N);
}

// Gradient of energy functional
Vector3d computeGradient(const MatrixXd &x_ref, const Vector2d &x_hat_ref, const MatrixXd &x_hat,
                         double theta, const Vector2d &c, double hat_d, int barrier_taylor_N) {
    MatrixXd y = computeY(x_ref, x_hat_ref, theta, c);
    MatrixXd grad_PE = potentialEnergyGrad(y, hat_d, barrier_taylor_N);

    Vector3d grad = Vector3d::Zero();
    Matrix2d Rprime = rotationMatrixDeriv(theta);

    // --- partial hat(Psi) / partial theta ---
    for (int i = 0; i < numPoints; ++i) {
        Vector2d dx = x_ref.row(i).transpose() - x_hat_ref;
        Vector2d dy = y.row(i).transpose() - x_hat.row(i).transpose();
        Vector2d dE = mass * dy + dt * dt * grad_PE.row(i).transpose();

        grad(0) += dE.dot(Rprime * dx);
    }

    // --- partial hat(Psi) / partial c ---
    Vector2d sum_xhat = Vector2d::Zero();
    Vector2d sum_gradPE = Vector2d::Zero();
    for (int i = 0; i < numPoints; ++i) {
        sum_xhat += x_hat.row(i).transpose();
        sum_gradPE += grad_PE.row(i).transpose();
    }

    grad.segment<2>(1) = totalMass * c - sum_xhat * mass + dt * dt * sum_gradPE;

    return grad;
}

// Computes the 3x3 Hessian matrix of the reduced energy function.
// The matrix has the following block structure:
//
// [ d^2Psi/dtheta^2         d^2Psi/dtheta_dc1       d^2Psi/dtheta_dc2      ]
// [ d^2Psi/dc1_dtheta       d^2Psi/dc1^2            d^2Psi/dc1_dc2         ]
// [ d^2Psi/dc2_dtheta       d^2Psi/dc2_dc1          d^2Psi/dc2^2           ]
Matrix3d computeHessian(const MatrixXd &x_ref, const Vector2d &x_hat_ref, const MatrixXd &x_hat,
                        double theta, const Vector2d &c, double hat_d, int barrier_taylor_N) {
    Matrix2d R = rotationMatrix(theta);
    Matrix2d Rprime = rotationMatrixDeriv(theta);
    Matrix2d Rpp = rotationMatrixSecondDeriv(theta);

    Matrix3d H = Matrix3d::Zero();
    MatrixXd y = computeY(x_ref, x_hat_ref, theta, c);

    for (int i = 0; i < numPoints; ++i) {
        Vector2d dx = x_ref.row(i).transpose() - x_hat_ref;
        Vector2d dy = y.row(i).transpose() - x_hat.row(i).transpose();

        Vector2d dR_dx = Rprime * dx; // partial y/ partial theta
        Vector2d ddR_dx = Rpp * dx; // partial^2 y/ partial theta^2

        double d2b = d2b_dd2(y(i, 1), hat_d, barrier_taylor_N);  // second derivative of barrier in y

        // --- d^2Psi/dtheta^2  ---
        double h_theta_theta = mass * dR_dx.dot(dR_dx) + mass * dy.dot(ddR_dx);
        h_theta_theta += dt * dt * d2b * std::pow(dR_dx(1), 2);  // only y contributes to PE
        H(0, 0) += h_theta_theta;

        // --- d^2Psi/dtheta_dc1 ---
        H(0, 1) += mass * dR_dx(0);

        // --- d^2Psi/dtheta_dc2 ---
        H(0, 2) += mass * dR_dx(1) + dt * dt * d2b * dR_dx(1);

        // --- d^2Psi/dc1^2 ---
        H(1, 1) += 0;

        // d^2Psi/dc2^2
        H(2, 2) += dt * dt * d2b;      // y-y gets PE second derivative
    }

    // Add mass to H_c_c terms
    H(1, 1) += totalMass;
    H(2, 2) += totalMass;

    // Fill symmetry
    H(1, 0) = H(0, 1);
    H(2, 0) = H(0, 2);

    return H;
}

// ---------- Mesh Export ----------

// Output mesh to .obj file for visualization
void writeObj(int frame, const MatrixXd& vertices) {
    std::ostringstream filename;
    filename << "frame_" << std::setfill('0') << std::setw(4) << frame << ".obj";
    std::ofstream out(filename.str());

    if (!out) {
        std::cerr << "Failed to open " << filename.str() << " for writing.\n";
        return;
    }

    // 1. Write square vertices
    for (int i = 0; i < vertices.rows(); ++i) {
        out << "v " << vertices(i, 0) << " " << vertices(i, 1) << " 0.0\n";
    }

    // 2. Write circular boundary as a polygon
    const int circleSegments = 64;
    for (int i = 0; i < circleSegments; ++i) {
        double angle = 2.0 * M_PI * i / circleSegments;
        double x = circleCenter(0) + circleRadius * cos(angle);
        double y = circleCenter(1) + circleRadius * sin(angle);
        out << "v " << x << " " << y << " 0.0\n";
    }

    // 3. Write square faces (1-indexed)
    out << "f 1 2 3\n";
    out << "f 1 3 4\n";

    // Optional: write the circle as a closed loop (line)
    // Square = 4 vertices, so circle vertices start at index 5
    out << "l";
    for (int i = 0; i < circleSegments; ++i) {
        out << " " << (5 + i);
    }
    out << " " << 5 << "\n"; // close the loop

    out.close();
}

// ---------- Simulation  ----------

int main() {
    MatrixXd x_ref(numPoints, 2);
    x_ref << 0.15, 0.15,
            0.3,  0.15,
            0.3,  0.3,
            0.15, 0.3;

    Vector2d x_hat_ref = x_ref.colwise().mean();
    Vector2d c(0.0, 2.0);       // Initial translation
    double theta = M_PI / 3.0;  // Initial rotation
    Vector2d v_cm(-1.0, 0.0);    // Initial center-of-mass velocity
    double omega = 2.5;         // Initial angular velocity

    double hat_d = 0.4;          // Barrier soft threshold
    int barrier_taylor_N = 25;   // Taylor expansion order

    MatrixXd x0 = computeY(x_ref, x_hat_ref, theta, c);
    cout << "Frame 0: theta = " << theta << ", c = (" << c(0) << ", " << c(1) << ")\n";
    for (int i = 0; i < numPoints; ++i) {
        cout << "  Vertex " << i << ": (" << x0(i, 0) << ", " << x0(i, 1) << ")\n";
    }

    for (int frame = 1; frame <= 200; ++frame) {
        double theta_old = theta;

        MatrixXd x = computeY(x_ref, x_hat_ref, theta, c);

        // Use current omega and v_cm to define x_hat
        MatrixXd x_hat(numPoints, 2);
        for (int i = 0; i < numPoints; ++i) {
            Vector2d xi = x.row(i).transpose();
            Vector2d r = xi - c;
            Vector2d omega_cross_r;
            omega_cross_r << -r.y(), r.x();
            Vector2d v_i = omega * omega_cross_r + v_cm;
            x_hat.row(i) = (xi + dt * v_i).transpose();
        }

        int iter = 0;
        double residual = 0.0;
        for (; iter < maxNewtonIters; ++iter) {
            Vector3d grad = computeGradient(x_ref, x_hat_ref, x_hat, theta, c, hat_d, barrier_taylor_N);
            residual = grad.norm();
            if (residual < newtonTol) break;

            Matrix3d H = computeHessian(x_ref, x_hat_ref, x_hat, theta, c, hat_d, barrier_taylor_N);
            Vector3d delta = H.ldlt().solve(-grad);

            theta += delta[0];
            c[0] += delta[1];
            c[1] += delta[2];
        }

        cout << "Total Newton steps used: " << iter << ", with final gradient residual = " << residual;
        if (residual >= newtonTol) cout << " [ABOVE TOLERANCE]";
        cout << endl;

        MatrixXd x_new = computeY(x_ref, x_hat_ref, theta, c);

        v_cm = (x_new.colwise().mean() - x.colwise().mean()).transpose() / dt;
        omega = (theta - theta_old) / dt;


        cout << "Frame " << frame << ": theta = " << theta
             << ", c = (" << c(0) << ", " << c(1) << ")"
             << ", omega = " << omega << endl;

        for (int i = 0; i < numPoints; ++i) {
            Vector2d xi = x_new.row(i).transpose();
            double dist = (xi - circleCenter).norm();

            if (dist > circleRadius)
                cout << "  Vertex " << i << ": (" << xi(0) << ", " << xi(1) << ") [OUTSIDE CIRCLE!]\n";
            else
                cout << "  Vertex " << i << ": (" << xi(0) << ", " << xi(1) << ")\n";
        }

        writeObj(frame, x_new);
    }

    return 0;
}






