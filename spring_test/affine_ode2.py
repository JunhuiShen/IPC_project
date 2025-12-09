import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# Data structure
# ======================================================

class Sample:
    def __init__(self, pos, vel, mass=1.0):
        self.x = np.array(pos, dtype=float)      # Material coordinate
        self.v = np.array(vel, dtype=float)      # Initial material velocity
        self.m = mass
        self.x_current = self.x.copy()           # Spatial position x(t)


# ======================================================
# Random sampling
# ======================================================

def make_random_samples(n=10, seed=1, radius=0.5):
    np.random.seed(seed)
    pts = []
    attempts = 0
    max_attempts = 2000

    while len(pts) < n and attempts < max_attempts:
        attempts += 1
        candidate = np.random.uniform(-10, 10, size=2)
        if all(np.linalg.norm(candidate - p.x) > radius for p in pts):
            vel = np.random.normal(scale=1.0, size=2)
            pts.append(Sample(candidate, 0.5 * vel, 1.0))

    return pts


# ======================================================
# Center of mass (reference space)
# ======================================================

def center_of_mass(samples):
    M = np.sum([s.m for s in samples])
    return np.sum([s.m * s.x for s in samples], axis=0) / M


# ======================================================
# Rotational + translational basis functions
# ======================================================
# Basis vector fields:
#   U1 = rotational basis: (-y, x)
#   U2 = translation in x: (1, 0)
#   U3 = translation in y: (0, 1)

def Uk_rot(k, pos, xhat):
    dx, dy = pos - xhat

    if k == 1:    # rotation basis
        return np.array([-dy, dx])

    if k == 2:    # translation x
        return np.array([1.0, 0.0])

    if k == 3:    # translation y
        return np.array([0.0, 1.0])

    return np.zeros(2)


# ======================================================
# Build least-squares system Gc = b
# (restricted to rotation + translation)
# ======================================================

def build_G_b_rot(samples, xhat):
    G = np.zeros((3, 3))
    b = np.zeros(3)

    for s in samples:
        for k in range(3):
            uk = Uk_rot(k+1, s.x, xhat)
            b[k] += s.m * (uk @ s.v)
            for j in range(3):
                uj = Uk_rot(j+1, s.x, xhat)
                G[k, j] += s.m * (uk @ uj)

    return G, b


# ======================================================
# Construct B and v0 from rotational projection
# ======================================================

def assemble_rot_B_v0(c):
    omega = c[0]

    # Pure rotation generator matrix
    B = omega * np.array([[0, -1],
                          [1,  0]])

    # Translational velocity
    v0 = np.array([c[1], c[2]])

    return B, v0


# ======================================================
# Simulation with DV/Dt = 0 and B restricted to rotation
# ======================================================

def simulate_uniform_affine(samples, dt=1/30, n_frames=300):
    xhat_initial = center_of_mass(samples)

    # Build rotational least-squares system
    G, b = build_G_b_rot(samples, xhat_initial)
    c = np.linalg.solve(G, b)        # solve for omega, vx, vy
    B, v0 = assemble_rot_B_v0(c)

    A0 = np.eye(2)
    xhat_0 = xhat_initial.copy()

    history = [[] for _ in samples]

    for s in samples:
        s.x_current = s.x.copy()

    for i, s in enumerate(samples):
        history[i].append(s.x_current.copy())

    time = 0.0

    for _ in range(n_frames):
        time += dt

        # Guaranteed invertible deformation gradient
        # det(I + tB) = 1 + (omega t)^2 > 0
        A_t = A0 + time * B
        xhat_t = xhat_0 + v0 * time

        for i, s in enumerate(samples):
            X = s.x
            s.x_current = A_t @ (X - xhat_initial) + xhat_t
            history[i].append(s.x_current.copy())

    return np.array(history)


# ======================================================
# Plot trajectories
# ======================================================

def plot_static(history):
    plt.figure(figsize=(9, 9))
    cmap = plt.cm.get_cmap("turbo", len(history))

    for i, traj in enumerate(history):
        traj = np.array(traj)
        col = cmap(i)

        label_traj = "Particle trajectory" if i == 0 else None
        plt.plot(traj[:, 0], traj[:, 1], color=col, linewidth=2, label=label_traj)

        label_init = "Initial position" if i == 0 else None
        plt.scatter(traj[0, 0], traj[0, 1], color="black", s=90, label=label_init, zorder=3)

        label_final = "Final position" if i == 0 else None
        plt.scatter(traj[-1, 0], traj[-1, 1], color="red", s=90, label=label_final, zorder=3)

        plt.plot([traj[0,0], traj[-1,0]], [traj[0,1], traj[-1,1]],
                 "--", color=col, alpha=0.4)

    plt.gca().set_aspect("equal", "box")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.title("Collision-Free Affine Motion (Pure Rotation + Translation)")
    plt.legend(loc='upper right')
    plt.show()


# ======================================================
# Run
# ======================================================

if __name__ == "__main__":
    samples = make_random_samples(n=10, seed=900)
    history = simulate_uniform_affine(samples)
    plot_static(history)



