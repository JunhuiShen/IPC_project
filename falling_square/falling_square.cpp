#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
#include <string>

using namespace Eigen;
using namespace std;

// Constants
const int numPoints = 4;
const double dt = 0.03;
const double gravity = 9.81;
const double totalMass = 0.05;
const double mass = totalMass / numPoints;
const int maxNewtonIters = 300;
const double newtonTol = 1e-10;

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

// Potential energy: barrier function + gravity b(d) + mg * y where b(d) = sum_{i=1}^{4} b(y_{i2} (theta, c)
double potentialEnergy(const MatrixXd &y, double hat_d, int barrier_taylor_N) {
    double PE = 0;
    for (int i = 0; i < numPoints; ++i) {
        double yi = y(i, 1); // y_{i2}
        PE += barrier(yi, hat_d, barrier_taylor_N); // barrier
        PE += mass * gravity * y(i, 1); // gravity
    }
    return PE;
}

// Gradient of potential energy: b'(y_{i2}) + mg
MatrixXd potentialEnergyGrad(const MatrixXd &y, double hat_d, int barrier_taylor_N) {
    MatrixXd grad = MatrixXd::Zero(numPoints, 2);
    for (int i = 0; i < numPoints; ++i) {
        double yi = y(i, 1); // y_{i2}
        double barrier_grad = db_dd(yi, hat_d, barrier_taylor_N); // barrier gradient
        double gravity_grad = mass * gravity; // gravity gradient
        grad(i, 1) = barrier_grad + gravity_grad;
    }
    return grad;
}

// Hessian of potential energy: b''(y_{i2})
MatrixXd potentialEnergyHessian(const MatrixXd &y, double hat_d, int barrier_taylor_N) {
    MatrixXd H = MatrixXd::Zero(2 * numPoints, 2 * numPoints);
    for (int i = 0; i < numPoints; ++i) {
        double yi = y(i, 1);
        double hess_yy = d2b_dd2(yi, hat_d, barrier_taylor_N);  // gravity term is 0
        int row = 2 * i + 1; // y-index in flattened Hessian
        H(row, row) = hess_yy;
    }
    return H;
}

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

double computePsi(const MatrixXd &y, const MatrixXd &x_hat, double hat_d, int N) {
    return 0.5 * mass * ((y - x_hat).squaredNorm())  + dt * dt * potentialEnergy(y, hat_d, N);
}

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

void writeObj(int frame, const MatrixXd& vertices) {
    std::ostringstream filename;
    filename << "frame_" << std::setfill('0') << std::setw(4) << frame << ".obj";
    std::ofstream out(filename.str());

    if (!out) {
        std::cerr << "Failed to open " << filename.str() << " for writing.\n";
        return;
    }

    // Write vertices
    for (int i = 0; i < vertices.rows(); ++i) {
        out << "v " << vertices(i, 0) << " " << vertices(i, 1) << " 0.0\n";
    }

    // Write faces (1-indexed for OBJ)
    out << "f 1 2 3\n";
    out << "f 1 3 4\n";

    out.close();
}

int main() {
    MatrixXd x_ref(numPoints, 2);
    x_ref << 0.15, 0.15,
            0.3,  0.15,
            0.3,  0.3,
            0.15, 0.3;

    Vector2d x_hat_ref = x_ref.colwise().mean();
    Vector2d c(0, 2.0);       // Initial translation
    double theta = M_PI / 3.0;  // Initial rotation
    Vector2d v_cm(0.0, 0.0);    // Initial center-of-mass velocity
    double omega = 2;         // Initial angular velocity

    double hat_d = 0.2;
    int barrier_taylor_N = 25;

    MatrixXd x0 = computeY(x_ref, x_hat_ref, theta, c);
    cout << "Frame 0: theta = " << theta << ", c = (" << c(0) << ", " << c(1) << ")\n";
    for (int i = 0; i < numPoints; ++i) {
        cout << "  Vertex " << i << ": (" << x0(i, 0) << ", " << x0(i, 1) << ")\n";
    }

    for (int frame = 1; frame <= 150; ++frame) {
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
            double yi = x_new(i, 1);
            if (yi < 0)
                cout << "  Vertex " << i << ": (" << x_new(i, 0) << ", " << yi << ") [BELOW GROUND!]\n";
            else
                cout << "  Vertex " << i << ": (" << x_new(i, 0) << ", " << yi << ")\n";
        }

        writeObj(frame, x_new);
    }


    return 0;
}
