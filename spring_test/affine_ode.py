import numpy as np
from scipy.linalg import schur
import matplotlib.pyplot as plt


# ======================================================
# Data structure
# ======================================================

class Sample:
    def __init__(self, pos, vel, mass=1.0):
        self.x = np.array(pos, dtype=float)
        self.v = np.array(vel, dtype=float)
        self.m = mass


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
            pts.append(Sample(candidate, vel, 1.0))

    return pts

# ======================================================
# Math
# ======================================================
def center_of_mass(samples):
    M = np.sum([s.m for s in samples])
    return np.sum([s.m * s.x for s in samples], axis=0) / M


def Uk(k, x, xhat):
    dx, dy = x - xhat
    if k == 1: return np.array([-dy, dx])
    if k == 2: return np.array([ dy, dx])
    if k == 3: return np.array([ dx,  0])
    if k == 4: return np.array([ 0,  dy])
    if k == 5: return np.array([1, 0])
    if k == 6: return np.array([0, 1])
    return np.zeros(2)


def build_G_b(samples, xhat):
    G = np.zeros((6,6))
    b = np.zeros(6)

    for s in samples:
        for k in range(6):
            uk = Uk(k+1, s.x, xhat)
            b[k] += s.m * (uk @ s.v)
            for j in range(6):
                uj = Uk(j+1, s.x, xhat)
                G[k,j] += s.m * (uk @ uj)

    return G, b


def assemble_B_vhat(c):
    A1 = np.array([[0,-1],[1,0]])
    A2 = np.array([[0,1],[1,0]])
    A3 = np.array([[1,0],[0,0]])
    A4 = np.array([[0,0],[0,1]])
    return c[0]*A1 + c[1]*A2 + c[2]*A3 + c[3]*A4, np.array([c[4], c[5]])


# ======================================================
# Solve the affine ODE
# ======================================================
EPS = 1e-14

def step_exact_upper2x2real(a, b, d, h, vtil, dt):
    if abs(b) < EPS:
        Ea, Ed = np.exp(a*dt), np.exp(d*dt)
        h1 = Ea*h[0] + ((dt if abs(a)<EPS else (Ea-1)/a)*vtil[0])
        h2 = Ed*h[1] + ((dt if abs(d)<EPS else (Ed-1)/d)*vtil[1])
        return np.array([h1,h2])

    Ea, Ed = np.exp(a*dt), np.exp(d*dt)
    h2p = h[1] + dt*vtil[1] if abs(d)<EPS else Ed*h[1] + (Ed-1)/d*vtil[1]
    h1_hom = Ea*h[0] + (b/(d-a))*(Ed-Ea)*h[1] if abs(a-d)>=EPS else Ea*h[0] + b*Ea*(dt*h[1])
    h1_v1 = dt*vtil[0] if abs(a)<EPS else (Ea-1)/a*vtil[0]

    if abs(a)<EPS and abs(d)<EPS: 
        h1_v2 = 0.5*dt*dt*vtil[1]
    elif abs(d)<EPS: 
        h1_v2 = ((dt/a) - (1-Ea)/(a*a))*vtil[1]
    elif abs(a)<EPS: 
        h1_v2 = (((Ed-1)/(d*d)) - dt/d)*vtil[1]
    else: 
        h1_v2 = (((Ed-Ea)/(d-a)) - (Ea-1)/a)/d*vtil[1]

    return np.array([h1_hom + h1_v1 + h1_v2, h2p])


def step_exact_upper2x2complex(a, b, h, vtil, dt):
    E, c, s = np.exp(a*dt), np.cos(b*dt), np.sin(b*dt)
    h_hom = np.array([E*(c*h[0] + s*h[1]), E*(-s*h[0] + c*h[1])])
    A, B = E*c - 1, E*s
    det = a*a + b*b

    if det > EPS:
        m11, m12 = a*A + b*B, a*B - b*A
        F = np.array([[m11/det, m12/det],[-m12/det, m11/det]])
    else:
        dt2, dt3 = dt*dt, dt*dt*dt
        I = dt + 0.5*a*dt2 + (1/6)*(a*a - b*b)*dt3
        J = 0.5*b*dt2 + (1/3)*a*b*dt3
        F = np.array([[I,J],[-J,I]])

    return h_hom + F@vtil


def step_exact_schur2x2(T, h, vtil, dt):
    if np.all(np.abs(T)<EPS): 
        return h + dt*vtil
    a,b,c,d = T[0,0],T[0,1],T[1,0],T[1,1]
    return step_exact_upper2x2complex(a,b,h,vtil,dt) if (abs(a-d)<EPS and abs(c+b)<EPS and abs(c)>EPS) else step_exact_upper2x2real(a,b,d,h,vtil,dt)


# ======================================================
# Simulation
# ======================================================

def simulate_with_history(samples, dt=1/30, n_frames=300):
    xhat = center_of_mass(samples)
    G,b = build_G_b(samples, xhat)
    B,vhat = assemble_B_vhat(np.linalg.solve(G,b))
    T,Q = schur(B, output='real')

    QT = Q.T
    vtil = QT @ vhat
    H = [QT@(s.x-xhat) for s in samples]

    history = [[] for _ in samples]
    for _ in range(n_frames):
        for i in range(len(H)):
            H[i] = step_exact_schur2x2(T,H[i],vtil,dt)
            history[i].append((Q@H[i] + xhat).copy())

    return np.array(history)


# ======================================================
# Plot
# ======================================================

def plot_static(history):
    plt.figure(figsize=(9,9))
    cmap = plt.cm.get_cmap("turbo", len(history))

    for i,traj in enumerate(history):
        traj=np.array(traj)
        col = cmap(i)

        label_traj = "Particle trajectory" if i == 0 else None
        plt.plot(traj[:,0], traj[:,1], color=col, linewidth=2, label=label_traj)

        label_init = "Initial position" if i == 0 else None
        plt.scatter(traj[0,0],traj[0,1],color="black",s=90, label=label_init, zorder=3)

        label_final = "Final position" if i == 0 else None
        plt.scatter(traj[-1,0],traj[-1,1],color="red",s=90, label=label_final, zorder=3)

        plt.plot([traj[0,0],traj[-1,0]],[traj[0,1],traj[-1,1]],"--",color=col,alpha=0.4)

    plt.plot([], [], "--", color="gray", alpha=0.8, label="Linear trajectory")
    plt.gca().set_aspect("equal","box")
    plt.grid(True,linestyle="--",alpha=0.4)
    plt.title("Affine Trajectories")
    plt.legend(loc='upper right')
    plt.show()


# ======================================================
# Run
# ======================================================

if __name__ == "__main__":
    samples = make_random_samples(n=10, seed= 100)
    history = simulate_with_history(samples)
    plot_static(history)
