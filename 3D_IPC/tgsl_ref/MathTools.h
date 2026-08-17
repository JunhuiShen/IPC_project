#include <Eigen/Dense>

#include <core/algebra/ImplicitQRSVD.h>
#include <core/Definitions.h>
#include <stack>
#include <queue>
#include <io/BinaryIO.h>
#include <io/IO.h>
#ifndef MATHTOOLS_INCLUDED
#define MATHTOOLS_INCLUDED

namespace TGSL {

inline void PolarDecomposition(const T A00, const T A10, const T A01, const T A11, T& c, T& s, T& s00, T& s10, T& s11) {

  //A = [A00,A01]
  //    [A10,A11]
  //
  //A = RS
  //
  //R = [c,-s]
  //    [s, c]
  //
  //S = [s00,s10]
  //    [s10,s11]

  T v0 = A00 + A11, v1 = A01 - A10;
  T denom = sqrt(v0 * v0 + v1 * v1);
  if (denom == T(0)) {
    c = T(1);
    s = T(0);
    s00 = A00;
    s10 = A10;
    s11 = A11;
  } else {
    c = v0 / denom;
    s = -v1 / denom;
    s10 = -A00 * s + A10 * c;
    s00 = A00 * c + A10 * s;
    s11 = -A01 * s + A11 * c;
  }
}

inline void SymSchur2D(const T Aqq, const T App, const T Apq, T& c, T& s) {
  // A is an n x n matrix
  // (p, q) is an index pair satisfying 1 <= p < q <= n
  //
  // This function computes a (c, s) pair such that
  //
  // B = [ c, s]^T [a_pp, a_pq] [ c, s]
  //     [-s, c]   [a_pq, a_qq] [-s, c]
  //
  // is diagonal

  if (Apq != 0) {
    T tau = T(0.5) * (Aqq - App) / Apq;
    T t;
    if (tau >= 0)
      t = T(1) / (tau + std::sqrt(1 + tau * tau));
    else
      t = T(-1) / (-tau + std::sqrt(1 + tau * tau));
    c = T(1) / std::sqrt(1 + t * t);
    s = t * c;
  } else {
    c = 1;
    s = 0;
  }
}

inline T GetMaxvalue(const Eigen::Matrix<T, 3, 3>& PMat, sz& p, sz& q) {
  p = 0;
  q = 1;
  T temp;
  T max = PMat(0, 1);
  for (sz i = 0; i < 3; i++) {
    for (sz j = i + 1; j < 3; j++) {
      temp = abs(PMat(i, j));
      if (temp > max) {
        p = i;
        q = j;
        max = temp;
      }
    }
  }
  return max;
}

inline void Jacobi(const Eigen::Matrix<T, 3, 3>& B, Eigen::Matrix<T, 3, 1>& D, Eigen::Matrix<T, 3, 3>& V) {
  Eigen::Matrix<T, 3, 3> I, U;
  I.setIdentity();
  T max = T(1000000);
  Eigen::Matrix<T, 3, 3> tempB;
  tempB = B;
  sz p, q;
  int te = 0;
  while (!(max < 0.0001) || te < 30) {
    max = GetMaxvalue(tempB, p, q);
    T app = tempB(p, p);
    T aqq = tempB(q, q);
    T apq = tempB(p, q);

    T alpha = (app - aqq) / T(2);
    T beta = -apq;
    T gamma = abs(alpha) / sqrt(alpha * alpha + beta * beta);
    T s = sqrt((T(1) - gamma) / T(2));
    T c = sqrt((T(1) + gamma) / T(2));
    if (alpha * beta < T(0))
      s = -s;
    for (sz i = 0; i < 3; i++) {
      T temp = c * tempB(p, i) - s * tempB(q, i);
      tempB(q, i) = s * tempB(p, i) + c * tempB(q, i);
      tempB(p, i) = temp;
    }

    for (sz i = 0; i < 3; i++) {
      tempB(i, p) = tempB(p, i);
      tempB(i, q) = tempB(q, i);
    }

    tempB(p, p) = c * c * app + s * s * aqq - T(2) * s * c * apq;
    tempB(p, q) = s * c * (app - aqq) + (c * c - s * s) * apq;
    tempB(q, p) = s * c * (app - aqq) + (c * c - s * s) * apq;
    tempB(q, q) = s * s * app + c * c * aqq + T(2) * s * c * apq;

    for (sz i = 0; i < 3; i++) {
      T temp = c * I(i, p) - s * I(i, q);
      I(i, q) = s * I(i, p) + c * I(i, q);
      I(i, p) = temp;
    }
    te++;
  }
  V = I;
  D(0, 0) = tempB(0, 0);
  D(1, 0) = tempB(1, 1);
  D(2, 0) = tempB(2, 2);
}

inline void SVD(const T A00, const T A10, const T A01, const T A11, T& cu, T& su, T& cv, T& sv, T& s0, T& s1) {

  //A =     [A00,A01]
  //        [A10,A11]
  //
  //A =     U Sigma V^T
  //
  //U =     [cu,-su]
  //        [su, cu]
  //
  //V =     [cv,-sv]
  //        [sv, cv]
  //
  //Sigma = [s0, 0]
  //        [0, s1]

  T c, s, s00, s10, s11;
  PolarDecomposition(A00, A10, A01, A11, c, s, s00, s10, s11);
  SymSchur2D(s11, s00, s10, cv, sv);
  sv = -sv;
  s0 = cv * (cv * s00 + sv * s10) + sv * (cv * s10 + sv * s11);
  s1 = -sv * (-sv * s00 + cv * s10) + cv * (-sv * s10 + cv * s11);
  cu = c * cv - s * sv;
  su = s * cv + c * sv;
}

inline void SVD(const Eigen::Matrix<T, 3, 3>& F, Eigen::Matrix<T, 3, 3>& U, Eigen::Matrix<T, 3, 3>& V, Eigen::Matrix<T, 3, 1>& s) {
  JIXIE::singularValueDecomposition(F, U, s, V);
}

inline T Givens(const Vector2T& x,Vector2T& givens, const T tol=T(1e-13)){
  T norm=std::sqrt(x[0]*x[0] + x[1]*x[1]);
  if(norm>tol)
    givens={x[0]/norm,-x[1]/norm};
  else
    givens={T(1),0};
  return norm;
}

inline void GivensRotate(const Vector2T& givens,Vector2T& rotated){
  T x=rotated[0],y=rotated[1];
  T c=givens[0],s=givens[1];
  rotated={c*x - s*y, s*x + c*y};
}

inline void Jacobi(const Eigen::Matrix<T, 2, 2>& B, Eigen::Matrix<T, 2, 2>& D, Eigen::Matrix<T, 2, 2>& V) {
  T c, s;
  SymSchur2D(B(1, 1), B(0, 0), B(0, 1), c, s);

  // Build Givens
  V(0, 0) = c;
  V(0, 1) = s;
  V(1, 0) = -s;
  V(1, 1) = c;

  D = V.transpose() * B * V;
}

inline void Jacobi(const ALGEBRA::DenseMatrix& B, ALGEBRA::DenseMatrix& D, ALGEBRA::DenseMatrix& V) {
  if (B.m == 2 and B.n == 2) {
    T c, s;
    SymSchur2D(B(1, 1), B(0, 0), B(0, 1), c, s);
    V.Resize(2,2);
    // Build Givens
    V(0, 0) = c;
    V(0, 1) = s;
    V(1, 0) = -s;
    V(1, 1) = c;

    D = V.Transpose() * B * V;
  }
  else {
    TGSLAssert(false, "MathTools::Jacobi: Does not support non 2*2 DenseMatrix.");
  }
}

template <typename F>
T GaussianQuadrature2(F function, T a, T b) {
  std::array<T, 2> weights = {{T(1), T(1)}};
  std::array<T, 2> quad_points = {{T(1) / std::sqrt(T(3)), T(-1) / std::sqrt(T(3))}};

  T result = T(0);
  for (size_t i = 0; i < 2; ++i)
    result += weights[i] * function((b - a) / T(2) * quad_points[i] + (b + a) / 2);
  result *= (b - a) / T(2);

  return result;
}

template <typename F>
T GaussianQuadrature4(F function, T a, T b) {
  std::array<T, 4> weights = {{(T(18) + std::sqrt(T(30))) / T(36), (T(18) + std::sqrt(T(30))) / T(36), (T(18) - std::sqrt(T(30))) / T(36), (T(18) - std::sqrt(T(30))) / T(36)}};
  std::array<T, 4> quad_points = {{std::sqrt(T(3) / T(7) - T(2) / T(7) * std::sqrt(T(6) / T(5))),
                                   T(-1) * std::sqrt(T(3) / T(7) - T(2) / T(7) * std::sqrt(T(6) / T(5))),
                                   std::sqrt(T(3) / T(7) + T(2) / T(7) * std::sqrt(T(6) / T(5))),
                                   T(-1) * std::sqrt(T(3) / T(7) + T(2) / T(7) * std::sqrt(T(6) / T(5)))}};

  T result = T(0);
  for (size_t i = 0; i < 4; ++i)
    result += weights[i] * function((b - a) / T(2) * quad_points[i] + (b + a) / 2);
  result *= (b - a) / T(2);

  return result;
}

template <typename F>
T GaussianQuadrature2_2DSquare(const F& function, T a = T(0), T b = T(1)) {
  std::array<T, 2> weights = {{T(1), T(1)}};
  std::array<T, 2> quad_points = {{T(1) / std::sqrt(T(3)), T(-1) / std::sqrt(T(3))}};

  T result = T(0);
  for (size_t k = 0; k < 2; ++k)
    for (size_t l = 0; l < 2; ++l) {
      result += weights[k] * weights[l] * function(quad_points[k] * (b - a) / T(2) + (b + a) / T(2), quad_points[l] * (b - a) / T(2) + (b + a) / T(2));
    }
  result *= T(0.25);
  result *= (b - a) * (b - a);

  return result;
}

template <typename F>
T GaussianQuadrature4_2DSquare(const F& function, T a = T(0), T b = T(1)) {
  // for integrating over the square [a, b] x [a, b]
  std::array<T, 4> weights = {{(T(18) + std::sqrt(T(30))) / T(36), (T(18) + std::sqrt(T(30))) / T(36), (T(18) - std::sqrt(T(30))) / T(36), (T(18) - std::sqrt(T(30))) / T(36)}};
  std::array<T, 4> quad_points = {{std::sqrt(T(3) / T(7) - T(2) / T(7) * std::sqrt(T(6) / T(5))),
                                   T(-1) * std::sqrt(T(3) / T(7) - T(2) / T(7) * std::sqrt(T(6) / T(5))),
                                   std::sqrt(T(3) / T(7) + T(2) / T(7) * std::sqrt(T(6) / T(5))),
                                   T(-1) * std::sqrt(T(3) / T(7) + T(2) / T(7) * std::sqrt(T(6) / T(5)))}};

  T result = T(0);
  for (size_t k = 0; k < 4; ++k)
    for (size_t l = 0; l < 4; ++l)
      result += weights[k] * weights[l] * function(quad_points[k] * (b - a) / T(2) + (b + a) / T(2), quad_points[l] * (b - a) / T(2) + (b + a) / T(2));
  result *= T(0.25);
  result *= (b - a) * (b - a);

  return result;
}

template <typename F>
T GaussianQuadrature5_2DSquare(const F& function, T a = T(0), T b = T(1)) {
  // for integrating over the square [a, b] x [a, b]
  std::array<T, 5> weights = {{T(128) / T(225),
                               (T(322) + T(13) * std::sqrt(T(70))) / T(900),
                               (T(322) + T(13) * std::sqrt(T(70))) / T(900),
                               (T(322) - T(13) * std::sqrt(T(70))) / T(900),
                               (T(322) - T(13) * std::sqrt(T(70))) / T(900)}};
  std::array<T, 5> quad_points = {{T(0),
                                   T(1) / T(3) * std::sqrt(T(5) - T(2) * std::sqrt(T(10) / T(7))),
                                   T(-1) / T(3) * std::sqrt(T(5) - T(2) * std::sqrt(T(10) / T(7))),
                                   T(1) / T(3) * std::sqrt(T(5) + T(2) * std::sqrt(T(10) / T(7))),
                                   T(-1) / T(3) * std::sqrt(T(5) + T(2) * std::sqrt(T(10) / T(7)))}};

  T result = T(0);
  for (size_t k = 0; k < 5; ++k)
    for (size_t l = 0; l < 5; ++l)
      result += weights[k] * weights[l] * function(quad_points[k] * (b - a) / T(2) + (b + a) / T(2), quad_points[l] * (b - a) / T(2) + (b + a) / T(2));
  result *= T(0.25);
  result *= (b - a) * (b - a);

  return result;
}

template <typename F>
T GaussianQuadrature6_2DSquare(const F& function, T a = T(0), T b = T(1)) {
  // for integrating over the square [a, b] x [a, b]
  std::array<T, 6> weights = {{0.3607615730481386, 0.3607615730481386, 0.4679139345726910, 0.4679139345726910, 0.1713244923791704, 0.1713244923791704}};
  std::array<T, 6> quad_points = {{0.6612093864662645, -0.6612093864662645, -0.2386191860831969, 0.2386191860831969, -0.9324695142031521, 0.9324695142031521}};

  T result = T(0);
  for (size_t k = 0; k < 6; ++k)
    for (size_t l = 0; l < 6; ++l)
      result += weights[k] * weights[l] * function(quad_points[k] * (b - a) / T(2) + (b + a) / T(2), quad_points[l] * (b - a) / T(2) + (b + a) / T(2));
  result *= T(0.25);
  result *= (b - a) * (b - a);

  return result;
}

template <typename F>
T GaussianQuadrature6_3DCube(const F& function, T a = T(0), T b = T(1)) {
  // for integrating over the cube [a, b] x [a, b] x [a, b]
  std::array<T, 6> weights = {{0.3607615730481386, 0.3607615730481386, 0.4679139345726910, 0.4679139345726910, 0.1713244923791704, 0.1713244923791704}};
  std::array<T, 6> quad_points = {{0.6612093864662645, -0.6612093864662645, -0.2386191860831969, 0.2386191860831969, -0.9324695142031521, 0.9324695142031521}};

  T result = T(0);
  for (size_t k = 0; k < 6; ++k)
    for (size_t l = 0; l < 6; ++l)
      for (size_t m = 0; m < 6; ++m)
        result += weights[k] * weights[l] * weights[m] *
                  function(quad_points[k] * (b - a) / T(2) + (b + a) / T(2), quad_points[l] * (b - a) / T(2) + (b + a) / T(2), quad_points[m] * (b - a) / T(2) + (b + a) / T(2));
  result *= T(0.125);
  result *= (b - a) * (b - a) * (b - a);

  return result;
}

inline int GCD(int a, int b) {
  while (a != b) {
    if (a > b)
      a = a - b;
    else
      b = b - a;
  }

  return a;
}

inline int LeviCivita(int alpha, int beta) {
  TGSLAssert((0 <= alpha) && (alpha <= 1), "MathTools: Invalid first argument for LeviCivita 2D.");
  TGSLAssert((0 <= beta) && (beta <= 1), "MathTools: Invalid second argument for LeviCivita 2D.");
  if (alpha == 0 && beta == 1)
    return 1;
  else if (alpha == 1 && beta == 0)
    return -1;
  else
    return 0;
}

inline int LeviCivita(int alpha, int beta, int gamma) {
  TGSLAssert((0 <= alpha) && (alpha <= 2), "MathTools: Invalid first argument for LeviCivita 3D.");
  TGSLAssert((0 <= beta) && (beta <= 2), "MathTools: Invalid second argument for LeviCivita 3D.");
  TGSLAssert((0 <= gamma) && (gamma <= 2), "MathTools: Invalid second argument for LeviCivita 3D.");
  if (alpha == 0 && beta == 1 && gamma == 2)
    return 1;
  else if (alpha == 1 && beta == 2 && gamma == 0)
    return 1;
  else if (alpha == 2 && beta == 0 && gamma == 1)
    return 1;
  else if (alpha == 2 && beta == 1 && gamma == 0)
    return -1;
  else if (alpha == 1 && beta == 0 && gamma == 2)
    return -1;
  else if (alpha == 0 && beta == 2 && gamma == 1)
    return -1;
  else
    return 0;
}

// Quaternion Product Tensor
inline int QPT(int alpha, int beta, int gamma) {
  TGSLAssert((0 <= alpha) && (alpha <= 3), "MathTools: Invalid first argument for Quaternion Product Tensor.");
  TGSLAssert((0 <= beta) && (beta <= 3), "MathTools: Invalid second argument for Quaternion Product Tensor.");
  TGSLAssert((0 <= gamma) && (gamma <= 3), "MathTools: Invalid second argument for Quaternion Product Tensor.");
  if (alpha == 0 && beta == 0 && gamma == 0)
      return 1;
  else if (alpha == 0 && beta == gamma && beta != 0)
      return -1;
  else if (alpha != 0 && beta == 0 && alpha == gamma)
      return 1;
  else if (alpha != 0 && gamma == 0 && alpha == beta)
      return 1;
  else if (alpha != 0 && beta != 0 && gamma != 0)
      return LeviCivita(alpha - 1, beta - 1, gamma - 1);
  else
      return 0;
}

inline int QPT_QPT(int alpha, int beta, int delta, int epsilon) {
  TGSLAssert((0 <= alpha) && (alpha <= 3), "MathTools: Invalid first argument for Quaternion Product Tensor.");
  TGSLAssert((0 <= beta) && (beta <= 3), "MathTools: Invalid second argument for Quaternion Product Tensor.");
  TGSLAssert((0 <= delta) && (delta <= 3), "MathTools: Invalid second argument for Quaternion Product Tensor.");
  TGSLAssert((0 <= epsilon) && (epsilon <= 3), "MathTools: Invalid second argument for Quaternion Product Tensor.");
  T result = 0;
  for (int gamma = 0; gamma < 4; gamma++) {
    result += QPT(alpha, beta, gamma) * QPT(gamma, delta, epsilon);
  }
  return result;
}

inline void DFS(const IVV& L, const nm v, BV& visited, IV& component){
  visited[v] = 1_uc;
  component.emplace_back(v);
  for (nm neighbor:L[v]){
    if (not visited[neighbor]){
      DFS(L,neighbor,visited,component);
    }
  }
}

inline void ConnectedComponents_DFS(const IVV& L, IVV& C){
  //Input: L, the given adjacency list
  //Output: C, connected components
  BV visited(L.size(), 0_uc);
  for (sz v=0; v<L.size(); ++v){
    if (not visited[v]){
      IV component;
      DFS(L,v,visited,component);
      C.emplace_back(component);
    }
  }
}

inline void DFS_iterative(const IVV& L, nm v, BV& visited, IV& component){
  std::stack<nm> stack;
  stack.push(v);
  while (!stack.empty()){
    v = stack.top();
    stack.pop();
    if (!visited[v]){
      visited[v] = 1_uc;
      component.emplace_back(v);
      for (auto i = L[v].begin(); i != L[v].end(); ++i){
        if (!visited[*i]){
          stack.push(*i);
        }
      }
    }
  }
}

inline void ConnectedComponents_DFS_iterative(const IVV& L, IVV& C){
  //Input: L, the given adjacency list
  //Output: C, connected components
  BV visited(L.size(), 0_uc);
  for (sz v=0; v<L.size(); ++v){
    if (not visited[v]){
      IV component;
      DFS_iterative(L,v,visited,component);
      C.emplace_back(component);
    }
  }
}

inline void BFS(const IVV& L, const nm v, BV& visited, IV& component){
  IV to_visit = {v};
  while (not to_visit.empty()){
    nm current = to_visit[0];
    to_visit.erase(to_visit.begin());
    if (!visited[current]){
      visited[current] = 1_uc;
      component.emplace_back(current);
      for (nm neighbor:L[current]){
        if (not visited[neighbor]){
          to_visit.emplace_back(neighbor);
        }
      }
    }
  }
}

inline void ConnectedComponents_BFS(const IVV& L, IVV& C){
  //Input: L, the given adjacency list
  //Output: C, connected components
  BV visited(L.size(), 0_uc);
  for (sz v=0; v<L.size(); ++v){
    if (not visited[v]){
      IV component;
      BFS(L,v,visited,component);
      C.emplace_back(component);
    }
  }
}

//Grow the visited nodes to include nodes that are within dist in topology
inline IV Grow_BFS(const IVV& L, const IV& input_nodes, sz dist){
  //Input: L, the given adjacency list
  //Output: component, the nodes
  BV visited(L.size(), 0_uc);
  for (sz i = 0; i < input_nodes.size(); ++i){
    visited[input_nodes[0]] = 1_uc;
  }
  IV to_visit = input_nodes;
  IV component = input_nodes;
  for (sz iter = 0; iter < dist; ++iter){
    IV to_visit_new;
    for (sz i = 0; i < to_visit.size(); ++i){
      for (nm neighbor:L[to_visit[i]]){
        if (not visited[neighbor]){
          visited[neighbor] = 1_uc;
          to_visit_new.emplace_back(neighbor);
          component.emplace_back(neighbor);
        }
      }
    }
    to_visit = to_visit_new;
  }
  return component;
}

inline void ShortestPathBFS(const IVV& L, const nm start, const nm end, IV& shortest_path) {
  /*
  Returns a sequence of vertices beginning with 'start' and ending with 'end'
  representing one of the possible shortest paths from 'start' to 'end'.
  If no path exists, shortest_path is resized to 0.

  Uses the standard BFS approach: each visited vertex tracks the vertex who
  found it; when 'end' is reached we trace back the chain of predecessors.
  */
  TGSLAssert(0 <= start && sz(start) < L.size() && 0 <= end && sz(end) < L.size(), "ShortestPathBFS: Invalid start and end vertices specified.");
  TGSLAssert(start != end, "ShortestPathBFS: start and end cannot be the same.");

  // Build predecessors map using BFS
  std::unordered_map<nm, nm> predecessors;
  {
    BV visited(L.size(), 0_uc);
    std::queue<nm> bfs_queue;
    bfs_queue.push(start);
    visited[start] = 1_uc;

    while (!bfs_queue.empty()) {
      nm current_vert = bfs_queue.front();
      bfs_queue.pop();
      for (nm v : L[current_vert]) {
        if (!visited[v]) {
          predecessors[v] = current_vert;
          if (v == end)
            goto found;
          bfs_queue.push(v);
          visited[v] = 1_uc;
        }
      }
    }
    found:;
  }
  // Build shortest_path by recursively pulling predecessors
  if (predecessors.find(end) == predecessors.end())
    shortest_path.resize(0);
  else {
    std::stack<nm> path_stack;
    nm current = end;
    do {
      path_stack.push(current);
      current = predecessors.at(current);
    }
    while (current != start);
    path_stack.push(start);

    shortest_path.resize(path_stack.size());
    for (sz i = 0; !path_stack.empty(); i++) {
      shortest_path[i] = path_stack.top();
      path_stack.pop();
    }
  }
}

inline void FindCubicSpline(const TV& X, const TV& Y, std::vector<TV>& parameters) {
  // X, Y: size N+1
  // parameters: cubic spline parameters, size (N-1)*4, a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3
  // satisfying f_i(x_i) = y_i, f_i(x_{i+1}) = y_{i+1}, f_i'(x_{i+1}) = f_{i+1}'(x_{i+1}), f_i''(x_{i+1}) = f_{i+1}''(x_{i+1})
  // BC: f_0''(x_0) = f_{N-1}''(x_{N-1}) = 0 
  // Reference: Numerical Analysis by Burden, page 150
  sz N = sz(X.size() - 1);
  TGSLAssert(X.size() > 1, "MathTools::FindCubicSpline: X size must be greater than 1");
  TGSLAssert(X.size() == Y.size(), "MathTools::FindCubicSpline: X, Y variable sizes do not match");
  parameters.resize(N+1);
  for (sz i = 0; i < parameters.size(); i++) {
    parameters[i].resize(4);
  }
  TV H(N);
  TV Alpha(N+1);
  //solve for c_i
  for (sz i = 0; i < N; i++) {
    H[i] = X[i+1]-X[i];
  }
  for (sz i = 1; i < N; i++) {
    Alpha[i] = 3/H[i]*(Y[i+1]-Y[i])-3/H[i-1]*(Y[i]-Y[i-1]);
  }
  TV L(N+1);
  TV Mu(N+1);
  TV Z(N+1);
  L[0] = T(1);
  Mu[0] = T(0);
  Z[0] = T(0);
  for (sz i = 1; i < N; i++) {
    L[i] = 2*(X[i+1]-X[i-1]) - H[i-1]*Mu[i-1];
    Mu[i] = H[i]/L[i];
    Z[i] = (Alpha[i]-H[i-1]*Z[i-1])/L[i];
  }
  L[N] = T(1);
  Z[N] = T(0);
  parameters[N][2] = T(0); //C[N] = 0
  parameters[N][0] = Y[N];
  for (nm j = N-1; j > -1; j--) {
    parameters[j][0] = Y[j]; // a_j
    parameters[j][2] = Z[j] - Mu[j]*parameters[j+1][2]; // c_j
    parameters[j][1] = (parameters[j+1][0]-parameters[j][0])/H[j]-H[j]*(parameters[j+1][2] + 2*parameters[j][2])/T(3);
    parameters[j][3] = (parameters[j+1][2] - parameters[j][2])/(T(3)*H[j]);
  }
  parameters.resize(N);
}

inline void GenerateCubicSpline(const std::string& filename, const std::string& extension, const sz& T_end, std::vector<std::vector<std::vector<TV>>>& parameters, const nm& file_offset = 1) {
  TVP X;
  IV mesh;
  TVP Y;
  size_t N;
  if (extension == ".bin") {
    IO::Deserialize(X, filename + std::to_string(file_offset) + extension);
  }
  else if (extension == ".obj") {
    IO::ReadOBJMesh(X, mesh, N, filename + std::to_string(file_offset) + extension);
  }
  else {
    TGSLAssert(false, "MathTools::GenerateCubicSpline: extension is not supported.");
  }
  parameters.resize(X.size()); 
  std::vector<std::vector<TV>> data(X.size());
  // parameters[i][c][t][j]: point i, coordinate c, time interval t, coefficient j 
  // data[i][c][t]: position of point i, coordinate c, time t
  for (sz i = 0; i < parameters.size(); ++i) {
    parameters[i].resize(d);
    data[i].resize(d);
    for (sz c = 0; c < parameters[i].size(); ++c) {
      data[i][c].resize(T_end);
    }
  }
  TV time;
  for (sz t = 0; t < sz(T_end); ++t) {
    time.emplace_back(T(t));
    if (extension == ".bin") {
      IO::Deserialize(Y, filename + std::to_string(t+file_offset) + extension);
    }
    else if (extension == ".obj") {
      IO::ReadOBJMesh(Y, mesh, N, filename + std::to_string(t+file_offset) + extension);
    }
    // IO::Deserialize(Y, dir + "x_" + std::to_string(t+1) + ".bin");
    TGSLAssert(Y.size() == X.size(), "X sizes are not consistent.");
    for (sz i = 0; i < Y.size(); ++i) {
      for (sz c = 0; c < d; ++c) {
        data[i][c][t] = Y[i][c];
      }
    }
  }
  for (sz i = 0; i < data.size(); ++i) {
    for (sz c = 0; c < d; ++c) {
      FindCubicSpline(time, data[i][c], parameters[i][c]);
    }
  }
}

inline void InterpolateCubicSpline(const std::vector<std::vector<std::vector<TV>>>& parameters, const T& t, TVP& X) {
  TGSLAssert(t <= parameters[0][0].size(), "MathTools::InterpolateCubicSpline: time t is out of range.");
  X.resize(parameters.size());
  nm frame = nm(floor(t));
  T h = t - T(frame);
  for (sz i = 0; i < X.size(); ++i) {
    for (sz c = 0; c < d; ++c) {
      X[i][c] = parameters[i][c][frame][0] + parameters[i][c][frame][1]*h + parameters[i][c][frame][2]*h*h + parameters[i][c][frame][3]*h*h*h;
    }
  }
}

inline void ForwardStepCubicSpline(const TV& parameters, const T& i, const T& ks, const T& kd, const T& t_start, const T& t_end, const T& x_start, const T& v_start, T& x_end, T& v_end) {
  //parameters: length 4, cubic spline Y(t) = p[0] + p[1](t-i) + p[2](t-i)^2 + p[3](t-i)^3
  //k_s, k_d: coefficients for spring and damping
  //Solve for ODE X'' = k_s(X-Y) + k_d(X'-Y') subject to BC X(t_start) = x_start, V(t_start) = v_start, return X(t_end), V(t_end)
  // std::cout << "t_start = " << t_start << ", i = " << i << std::endl;
  // TGSLAssert(t_start >= i-1e-10, "MathTools::ForwardStepCubicSpline: start time needs to be after interval start.");
  TGSLAssert(t_end >= t_start, "MathTools::ForwardStepCubicSpline: end time needs to be after start time.");

  //Solve for inhomogenous solution P(t)= M[0] + M[1](t-i) + M[2](t-i)^2 + M[3](t-i)^3
  TV M = parameters;
  M[1] -= T(6)/ks*parameters[3];
  M[0] += T(6)*kd/ks/ks*parameters[3] - T(2)/ks*parameters[2];
  T p_start = M[0] + M[1]*(t_start-i) + M[2]*(t_start-i)*(t_start-i) + M[3]*(t_start-i)*(t_start-i)*(t_start-i); // p'(tn)
  T p_dot_start = M[1]+ T(2)*M[2]*(t_start-i) + T(3)*M[3]*(t_start-i)*(t_start-i);
  T p_end = M[0] + M[1]*(t_end-i) + M[2]*(t_end-i)*(t_end-i) + M[3]*(t_end-i)*(t_end-i)*(t_end-i); 
  T p_dot_end = M[1]+ T(2)*M[2]*(t_end-i) + T(3)*M[3]*(t_end-i)*(t_end-i); // p'(tn+1)
  T c1, c2;
  T dt = t_end - t_start;
  if (kd*kd > T(4)*ks) { //overdamping X(t) = c1e^{rp(t-tn)} + c2e^{rm(t-tn)} + P(t)
    T rp = -kd/T(2) + std::sqrt(kd*kd/T(4)-ks); //r_plus
    T rm = -kd/T(2) - std::sqrt(kd*kd/T(4)-ks); //r_minus
    c1 = (v_start-p_dot_start-rm*(x_start-p_start))/(rp-rm);
    c2 = x_start - p_start - c1;
    x_end = c1*exp(rp*dt) + c2*exp(rm*dt) + p_end;
    v_end = c1*rp*exp(rp*dt) + c2*rm*exp(rm*dt) + p_dot_end;
  }
  else if (kd*kd < T(4)*ks) { //underdamping
    T a = std::sqrt(ks-kd*kd/T(4));
    c1 = x_start - p_start;
    c2 = (v_start-p_dot_start)/a;
    x_end = exp(-kd/T(2)*dt)*(c1*cos(a*dt) + c2*sin(a*dt)) + p_end;
    v_end = -kd/T(2)*(x_end - p_end) + exp(-kd/T(2)*dt)*(-c1*a*sin(a*dt) + c2*a*cos(a*dt)) + p_dot_end;
  }
  else { //critical
    c1 = x_start - p_start;
    c2 = v_start + kd/T(2)*c1 - p_dot_start;
    x_end = c1*exp(-kd/T(2)*dt) + c2*dt*exp(-kd/T(2)*dt) + p_end;
    v_end = -c1*kd/T(2)*exp(-kd/T(2)*dt) + c2*exp(-kd/T(2)*dt) - c2*dt*kd/T(2)*exp(-kd/T(2)*dt) + p_dot_end;
  }
}

template <class T>
void
MinMax(const std::vector<T> &v, T &min, T &max)
{
    min = std::numeric_limits<T>::max();
    max = -std::numeric_limits<T>::max();
    for(size_t i=0; i < v.size(); i++)
    {
        min = std::min(min, v[i]);
        max = std::max(max, v[i]);
    }
}

template <class T>
void
MinAvgMax(const std::vector<T> &v, T &min, T &avg, T &max)
{
    min = std::numeric_limits<T>::max();
    max = -std::numeric_limits<T>::max();
    avg = 0.0;
    for(size_t i=0; i < v.size(); i++)
    {
        min = std::min(min, v[i]);
        max = std::max(max, v[i]);
        avg += v[i];
    }
    if(v.size()) avg /= v.size();
}

/*
  Outputs the pth Eigen dxd matrix from the array of flatten matrices
*/
inline Eigen::Matrix<T, d, d> GetEigenMatrix(sz p, const TV& mat_array)
{
  Eigen::Matrix<T, d, d> mat_eig;
  for (size_t r = 0; r < d; ++r) {
    for (size_t c = 0; c < d; ++c) {
      mat_eig(r, c) = mat_array[d * d * p + d * r + c];
    }
  }
  return mat_eig;
}

/*
  Outputs the pth Eigen dx1 vector from the array of vectors
*/
inline Eigen::Matrix<T, d, 1> GetEigenVector(sz p, const TVV& vec_array)
{
  Eigen::Matrix<T, d, 1> vec_eig;
  for (size_t r = 0; r < d; ++r) {
    vec_eig(r) = vec_array[p][r];
  }
  return vec_eig;
}

/*
  Write the pth Eigen dxd matrix into the array of flatten matrices
*/
inline void SetEigenMatrix(sz p, const Eigen::Matrix<T, d, d>& mat_eig, TV& mat_array)
{
  for (sz i = 0; i < d; ++i) {
    for (sz j = 0; j < d; ++j) {
      mat_array[d * d * p + d * i + j] = mat_eig(i, j);
    }
  }
}


}  // namespace TGSL

#endif
