#include <Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>

using namespace Eigen;
using namespace std;

// Constants
const int numPoints = 4;
const double dt = 0.1;
const double gravity = 9.81;
const double mass = 1.0;
const double totalMass = numPoints * mass;
const Vector2d g(0, gravity);  // Gravity as a vector

// Rotation matrix and derivatives
Matrix2d rotationMatrix(double theta) {
    double c = cos(theta), s = sin(theta);
    Matrix2d R;
    R << c, -s, s, c;
    return R;
}

Matrix2d rotationMatrixDeriv(double theta) {
    double c = cos(theta), s = sin(theta);
    Matrix2d Rprime;
    Rprime << -s, -c, c, -s;
    return Rprime;
}

// Barrier function and its derivatives
double barrier(double d, double hat_d, int N) {
    if (d < hat_d) {
        double x = d - hat_d;
        double result = 0.0;
        for (int n = 1; n <= N; ++n) {
            double sign = (n % 2 == 0) ? 1.0 : -1.0;
            double coeff = sign / (n * pow(hat_d, n));
            result += coeff * pow(x, n + 2);
        }
        return result;
    }
    return 0.0;
}

double db_dd(double d, double hat_d, int N) {
    if (d < hat_d) {
        double x = d - hat_d;
        double result = 0.0;
        for (int n = 1; n <= N; ++n) {
            double sign = (n % 2 == 0) ? 1.0 : -1.0;
            double coeff = sign * (n + 2) / (n * pow(hat_d, n));
            result += coeff * pow(x, n + 1);
        }
        return result;
    }
    return 0.0;
}



// Energy functions
//double potentialEnergy(const MatrixXd &y, double hat_d, int N) {
//    double PE = 0.0;
//    for (int i = 0; i < numPoints; ++i) {
//        double yi = y(i, 1);
//        PE += barrier(yi, hat_d, N);
//        PE += mass * g.dot(y.row(i));  // Gravity as dot product
//    }
//    return PE;
//}

double potentialEnergy(const MatrixXd &y, double hat_d, int N) {
    double PE = 0.0;
    for (int i = 0; i < numPoints; ++i) {
        double yi = y(i, 1);
        PE += mass * gravity * yi;
    }
    return PE;
}

//MatrixXd potentialEnergyGrad(const MatrixXd &y, double hat_d, int N) {
//    MatrixXd grad = MatrixXd::Zero(numPoints, 2);
//    for (int i = 0; i < numPoints; ++i) {
//        double yi = y(i, 1);
//        grad.row(i) = Vector2d(0, db_dd(yi, hat_d, N)) + mass * g;  // Vector-based gravity
//    }
//    return grad;
//}

MatrixXd potentialEnergyGrad(const MatrixXd &y, double hat_d, int N) {
    MatrixXd grad = MatrixXd::Zero(numPoints, 2);
    for (int i = 0; i < numPoints; ++i) {
        grad.row(i) = mass * g;
    }
    return grad;
}



// Geometry
MatrixXd computeY(const MatrixXd &x_ref, const Vector2d &x_hat_ref, double theta, const Vector2d &c) {
    Matrix2d R = rotationMatrix(theta);
    MatrixXd y(numPoints, 2);
    for (int i = 0; i < numPoints; ++i) {
        Vector2d dx = x_ref.row(i).transpose() - x_hat_ref;
        y.row(i) = (R * dx + c).transpose();
    }
    return y;
}

double computePsi(const MatrixXd &y, const MatrixXd &x_hat, double hat_d, int N) {
    double KE = 0.5 * mass * (y - x_hat).squaredNorm();
    double PE = potentialEnergy(y, hat_d, N);
    return KE + dt * dt * PE;
}

Vector3d computeGradient(const MatrixXd &x_ref, const Vector2d &x_hat_ref, const MatrixXd &x_hat,
                         double theta, const Vector2d &c, double hat_d, int N) {
    MatrixXd y = computeY(x_ref, x_hat_ref, theta, c);
    MatrixXd grad_PE = potentialEnergyGrad(y, hat_d, N);
    Vector3d grad = Vector3d::Zero();
    Matrix2d Rprime = rotationMatrixDeriv(theta);

    // partial Psi/ partial theta
    for (int i = 0; i < numPoints; ++i) {
        Vector2d dx = x_ref.row(i).transpose() - x_hat_ref;
        Vector2d dy = y.row(i).transpose() - x_hat.row(i).transpose();
        Vector2d dE = mass * dy + dt * dt * grad_PE.row(i).transpose();
        grad(0) += dE.dot(Rprime * dx);
    }

    // partial Psi/ partial c_alpha
    Vector2d sum_xhat = Vector2d::Zero(), sum_gradPE = Vector2d::Zero();
    for (int i = 0; i < numPoints; ++i) {
        sum_xhat +=  mass * x_hat.row(i).transpose();
        sum_gradPE += grad_PE.row(i).transpose();
    }

    grad.segment<2>(1) = totalMass * c - sum_xhat + dt * dt * sum_gradPE;
    return grad;
}

void testGradientConvergence(const MatrixXd &x_ref,
                             const Vector2d &x_hat_ref,
                             const MatrixXd &x_hat,
                             double theta,
                             const Vector2d &c,
                             double hat_d,
                             int barrier_taylor_N) {
    Vector3d grad_analytic = computeGradient(x_ref, x_hat_ref, x_hat, theta, c, hat_d, barrier_taylor_N);

    // Use practical range for convergence analysis
    std::vector<double> hs = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
    std::vector<double> errors;

    cout << "h\t\tGradient Error\t\tRate" << endl;
    cout << "------------------------------------------" << endl;

    for (size_t i = 0; i < hs.size(); ++i) {
        double h = hs[i];
        Vector3d grad_numeric;

        for (int j = 0; j < 3; ++j) {
            double t_plus = theta, t_minus = theta;
            Vector2d c_plus = c, c_minus = c;

            if (j == 0) { t_plus += h; t_minus -= h; }
            else        { c_plus[j - 1] += h; c_minus[j - 1] -= h; }

            double psi_plus = computePsi(computeY(x_ref, x_hat_ref, t_plus, c_plus), x_hat, hat_d, barrier_taylor_N);
            double psi_minus = computePsi(computeY(x_ref, x_hat_ref, t_minus, c_minus), x_hat, hat_d, barrier_taylor_N);

            grad_numeric[j] = (psi_plus - psi_minus) / (2 * h);
        }

        double error = (grad_analytic - grad_numeric).norm();
        errors.push_back(error);

        if (i == 0) {
            cout << std::scientific << h << "\t" << error << "\t\t--" << endl;
        } else {
            double rate = log10(errors[i - 1] / error) / log10(hs[i - 1] / h);
            cout << std::scientific << h << "\t" << error << "\t\t" << rate << endl;
        }
    }
}


int main() {
    MatrixXd x_ref(numPoints, 2);
    x_ref << -0.2, -0.2,
            0.2, -0.2,
            0.2,  0.2,
            -0.2,  0.2;

    Vector2d x_hat_ref = x_ref.colwise().mean();
    double theta = 0.0;
    Vector2d c(0, 2.0);
    Vector2d v_cm(0, 0.0);
    double hat_d = 0.2;
    int N = 10;

    MatrixXd x = computeY(x_ref, x_hat_ref, theta, c);
    MatrixXd x_hat = x + dt * MatrixXd::Ones(numPoints, 1) * v_cm.transpose();

    testGradientConvergence(x_ref, x_hat_ref, x_hat, theta, c, hat_d, N);

    return 0;
}