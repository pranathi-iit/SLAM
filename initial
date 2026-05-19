"""
IMU Dead-Reckoning SLAM Pipeline  v2
=====================================
Handles the specific characteristics of this dataset:
  - Sensor mounted at ~65° roll tilt from vertical
  - High-dynamics motion (gyro peaks ~18 rad/s → aggressive manoeuvres)
  - ~200 Hz, 28 s duration, 5679 samples
  - ZUPT via gyro-magnitude gating (more reliable than accel variance)

Pipeline:
  1. Static gravity alignment → quaternion initialisation
  2. Gyroscope bias estimation (first 200 static samples)
  3. Madgwick AHRS (6-DOF, IMU-only) orientation fusion
  4. World-frame linear acceleration = R·a_body − g
  5. ZUPT via sliding gyro-magnitude window → velocity zeroing
  6. Double integration → 3-D position
  7. Comprehensive diagnostic plots + CSV export
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import sys

GRAVITY = 9.81          # m/s²  (local approx)


# ─────────────────────────────────────────────────────────────────────────────
# Quaternion Utilities  [w, x, y, z]
# ─────────────────────────────────────────────────────────────────────────────

def qmult(q, r):
    w0,x0,y0,z0 = q;  w1,x1,y1,z1 = r
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1])

def qconj(q): return np.array([q[0], -q[1], -q[2], -q[3]])
def qnorm(q): n=np.linalg.norm(q); return q/n if n>1e-12 else q

def qrot(q, v):
    """Rotate 3-vector v by quaternion q."""
    return qmult(qmult(q, [0,v[0],v[1],v[2]]), qconj(q))[1:]

def quat_to_rpy(q):
    w,x,y,z = q
    roll  = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = np.arcsin(np.clip(2*(w*y-z*x), -1, 1))
    yaw   = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return roll, pitch, yaw

def gravity_alignment_quat(ax, ay, az):
    """Return quaternion that rotates body-frame g → world [0,0,g]."""
    a = np.array([ax, ay, az]); a /= np.linalg.norm(a)
    roll  = np.arctan2(a[1], a[2])
    pitch = np.arctan2(-a[0], np.sqrt(a[1]**2 + a[2]**2))
    cr,sr = np.cos(roll/2), np.sin(roll/2)
    cp,sp = np.cos(pitch/2), np.sin(pitch/2)
    return qnorm(np.array([cr*cp, sr*cp, cr*sp, -sr*sp]))


# ─────────────────────────────────────────────────────────────────────────────
# Madgwick AHRS  (IMU-only, no magnetometer)
# ─────────────────────────────────────────────────────────────────────────────

class Madgwick:
    def __init__(self, beta=0.05):
        self.beta = beta
        self.q = np.array([1.,0.,0.,0.])

    def update(self, gx,gy,gz, ax,ay,az, dt):
        q = self.q
        a_n = np.linalg.norm([ax,ay,az])
        if a_n < 1e-6:
            q += 0.5*qmult(q,[0,gx,gy,gz])*dt
            self.q = qnorm(q); return

        ax_,ay_,az_ = ax/a_n, ay/a_n, az/a_n
        w,x,y,z = q
        f = np.array([
            2*(x*z - w*y) - ax_,
            2*(w*x + y*z) - ay_,
            2*(0.5 - x*x - y*y) - az_])
        J = np.array([
            [-2*y,  2*z, -2*w,  2*x],
            [ 2*x,  2*w,  2*z,  2*y],
            [ 0,   -4*x, -4*y,  0  ]])
        grad = J.T @ f
        grad_n = np.linalg.norm(grad)
        if grad_n > 1e-10: grad /= grad_n
        qdot = 0.5*qmult(q,[0,gx,gy,gz]) - self.beta*grad
        self.q = qnorm(q + qdot*dt)


# ─────────────────────────────────────────────────────────────────────────────
# ZUPT (Zero-Velocity Update) — gyro-magnitude gating
# ─────────────────────────────────────────────────────────────────────────────

class ZUPT:
    """
    Declare static when the RMS gyro magnitude over a sliding window
    stays below a threshold.  Gyro is far less noisy than accelerometer
    for motion detection, especially on tilted sensors.
    """
    def __init__(self, window=15, thresh=0.04):
        self.window = window; self.thresh = thresh
        self._buf = []

    def update(self, gx, gy, gz):
        self._buf.append(gx**2+gy**2+gz**2)
        if len(self._buf) > self.window: self._buf.pop(0)
        rms = np.sqrt(np.mean(self._buf))
        return rms < self.thresh


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(data, madgwick_beta=0.05, zupt_thresh=0.04,
                 zupt_win=15, n_static=200):
    N = len(data)
    ts   = data[:,0]
    raw_g= data[:,1:4]   # gyro   rad/s
    raw_a= data[:,4:7]   # accel  m/s²

    # ── Bias estimation (static window) ─────────────────────────────────────
    gyro_bias  = raw_g[:n_static].mean(axis=0)
    accel_bias = raw_a[:n_static].mean(axis=0)
    g_vec_body = accel_bias.copy()
    g_mag      = np.linalg.norm(g_vec_body)

    print(f"[BIAS]  gyro  (rad/s) : {gyro_bias}")
    print(f"[BIAS]  accel (m/s²)  : {accel_bias}  |g|={g_mag:.4f}")

    # ── Gravity alignment ───────────────────────────────────────────────────
    filt = Madgwick(beta=madgwick_beta)
    filt.q = gravity_alignment_quat(*g_vec_body)
    r0,p0,_ = quat_to_rpy(filt.q)
    print(f"[INIT]  Roll={np.degrees(r0):.2f}°  Pitch={np.degrees(p0):.2f}°")

    zupt   = ZUPT(window=zupt_win, thresh=zupt_thresh)
    g_world = np.array([0., 0., g_mag])    # world-frame gravity

    # ── Output buffers ──────────────────────────────────────────────────────
    roll_h  = np.zeros(N); pitch_h = np.zeros(N); yaw_h = np.zeros(N)
    pos_h   = np.zeros((N,3)); vel_h = np.zeros((N,3))
    a_w_h   = np.zeros((N,3)); zupt_h = np.zeros(N, dtype=bool)

    pos = np.zeros(3); vel = np.zeros(3)

    for i in range(N):
        dt = float(ts[i]-ts[i-1]) if i>0 else 0.005
        dt = np.clip(dt, 1e-4, 0.05)          # guard against timestamp gaps

        gx,gy,gz = raw_g[i] - gyro_bias
        ax,ay,az = raw_a[i] - accel_bias       # residual accel (gravity removed)

        # Add back corrected gravity so Madgwick sees proper body-frame accel
        # (accel_bias ≈ gravity; we pass the corrected total reading)
        ax_f,ay_f,az_f = raw_a[i] - gyro_bias*0   # pass raw accel to filter
        filt.update(gx,gy,gz, raw_a[i,0],raw_a[i,1],raw_a[i,2], dt)
        q = filt.q

        roll_h[i], pitch_h[i], yaw_h[i] = quat_to_rpy(q)

        # World-frame linear accel = R·a_body − g_world
        a_world = qrot(q, raw_a[i]) - g_world
        a_w_h[i] = a_world

        # ZUPT
        is_static = zupt.update(gx,gy,gz)
        zupt_h[i] = is_static
        if is_static:
            vel = np.zeros(3)

        vel += a_world * dt
        pos += vel * dt
        pos_h[i] = pos; vel_h[i] = vel

    n_zupt = zupt_h.sum()
    print(f"[DONE]  {N} samples | {ts[-1]-ts[0]:.2f} s")
    print(f"[ZUPT]  static frames : {n_zupt} / {N}  ({100*n_zupt/N:.1f}%)")
    print(f"[TRAJ]  Δpos (m) : X={pos_h[-1,0]:.3f}  Y={pos_h[-1,1]:.3f}  Z={pos_h[-1,2]:.3f}")
    return dict(ts=ts, pos=pos_h, vel=vel_h,
                roll=roll_h, pitch=pitch_h, yaw=yaw_h,
                zupt=zupt_h, accel_w=a_w_h, raw=data,
                gyro_bias=gyro_bias, accel_bias=accel_bias)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(res, save_path=None):
    ts    = res["ts"] - res["ts"][0]
    pos   = res["pos"]; vel = res["vel"]
    roll  = np.degrees(res["roll"])
    pitch = np.degrees(res["pitch"])
    yaw   = np.degrees(res["yaw"])
    raw   = res["raw"]; zupt = res["zupt"]
    vmag  = np.linalg.norm(vel, axis=1)
    amag  = np.linalg.norm(raw[:,4:7], axis=1)

    plt.rcParams.update({
        "figure.facecolor":"#0b0d12","axes.facecolor":"#10131a",
        "axes.edgecolor":"#252a38","axes.labelcolor":"#bcc0cc",
        "xtick.color":"#5a607a","ytick.color":"#5a607a",
        "text.color":"#bcc0cc","grid.color":"#1a1e2a",
        "grid.linewidth":0.5,"lines.linewidth":1.3,
        "font.family":"monospace","font.size":8,
    })

    C = dict(cx="#00d8ff",cy="#39ff14",cz="#ff8c42",
             mag="#ffd700",zupt="#7fff7f",vel="#5b9dff",red="#ff4060")

    fig = plt.figure(figsize=(22,15))
    fig.suptitle("IMU Dead-Reckoning SLAM  —  6-DoF Pipeline",
                 fontsize=16, fontweight="bold", color="#dde2f0", y=0.98)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.38)

    def ax_(row, col, title, **kw):
        a = fig.add_subplot(gs[row, col], **kw)
        a.set_title(title, color="#6e7e9e", fontsize=8, pad=4)
        a.grid(True, alpha=0.4); return a

    # Row 0 — raw signals
    a0 = ax_(0,0,"Raw Gyroscope  (rad/s)")
    a0.plot(ts,raw[:,1],color=C["cx"],label="ωx",alpha=.85)
    a0.plot(ts,raw[:,2],color=C["cy"],label="ωy",alpha=.85)
    a0.plot(ts,raw[:,3],color=C["cz"],label="ωz",alpha=.85)
    a0.legend(fontsize=6); a0.set_xlabel("t (s)")

    a1 = ax_(0,1,"Raw Accelerometer  (m/s²)")
    a1.plot(ts,raw[:,4],color=C["cx"],label="ax",alpha=.85)
    a1.plot(ts,raw[:,5],color=C["cy"],label="ay",alpha=.85)
    a1.plot(ts,raw[:,6],color=C["cz"],label="az",alpha=.85)
    a1.legend(fontsize=6); a1.set_xlabel("t (s)")

    a2 = ax_(0,2,"|Accel| magnitude  (m/s²)")
    a2.plot(ts,amag,color=C["mag"],linewidth=1.2)
    a2.axhline(GRAVITY,color=C["red"],ls="--",lw=0.9,label="9.81 m/s²")
    a2.fill_between(ts,amag,GRAVITY,alpha=.12,color=C["mag"])
    a2.legend(fontsize=6); a2.set_xlabel("t (s)")

    a3 = ax_(0,3,"|Gyro| magnitude  (rad/s)")
    gm = np.linalg.norm(raw[:,1:4],axis=1)
    a3.plot(ts,gm,color=C["cz"],linewidth=1.1)
    a3.axhline(0.04,color=C["zupt"],ls="--",lw=0.8,label="ZUPT thresh")
    a3.fill_between(ts,0,gm*zupt,alpha=.25,color=C["zupt"],label="ZUPT")
    a3.legend(fontsize=6); a3.set_xlabel("t (s)")

    # Row 1 — orientation & velocity
    a4 = ax_(1,0,"Roll / Pitch / Yaw  (°)")
    a4.plot(ts,roll, color=C["cx"],label="Roll")
    a4.plot(ts,pitch,color=C["cy"],label="Pitch")
    a4.plot(ts,yaw,  color=C["cz"],label="Yaw")
    a4.legend(fontsize=6); a4.set_xlabel("t (s)")

    a5 = ax_(1,1,"World-frame Velocity  (m/s)")
    a5.plot(ts,vel[:,0],color=C["cx"],label="Vx",alpha=.8)
    a5.plot(ts,vel[:,1],color=C["cy"],label="Vy",alpha=.8)
    a5.plot(ts,vel[:,2],color=C["cz"],label="Vz",alpha=.8)
    a5.plot(ts,vmag,    color=C["mag"],label="|V|",lw=1.8)
    a5.fill_between(ts,0,vmag.max()*zupt,alpha=.15,color=C["zupt"])
    a5.legend(fontsize=6); a5.set_xlabel("t (s)")

    a6 = ax_(1,2,"Dead-Reckoning Position  (m)")
    a6.plot(ts,pos[:,0],color=C["cx"],label="X")
    a6.plot(ts,pos[:,1],color=C["cy"],label="Y")
    a6.plot(ts,pos[:,2],color=C["cz"],label="Z")
    a6.legend(fontsize=6); a6.set_xlabel("t (s)")

    a7 = ax_(1,3,"Linear Accel (world-frame, m/s²)")
    aw = res["accel_w"]
    a7.plot(ts,aw[:,0],color=C["cx"],label="Ax_w",alpha=.8)
    a7.plot(ts,aw[:,1],color=C["cy"],label="Ay_w",alpha=.8)
    a7.plot(ts,aw[:,2],color=C["cz"],label="Az_w",alpha=.8)
    a7.legend(fontsize=6); a7.set_xlabel("t (s)")

    # Row 2 — trajectories
    a8 = ax_(2,0,"2D Trajectory  X–Y")
    sc = a8.scatter(pos[:,0],pos[:,1],c=ts,cmap="plasma",s=2,alpha=.8)
    a8.plot(*pos[0,:2],"o",color=C["cy"],ms=8,label="Start",zorder=5)
    a8.plot(*pos[-1,:2],"*",color=C["cz"],ms=12,label="End",zorder=5)
    plt.colorbar(sc,ax=a8,label="t(s)",pad=0.02)
    a8.set_xlabel("X (m)"); a8.set_ylabel("Y (m)")
    a8.set_aspect("equal","datalim"); a8.legend(fontsize=6)

    a9 = ax_(2,1,"2D Trajectory  X–Z")
    sc2 = a9.scatter(pos[:,0],pos[:,2],c=ts,cmap="viridis",s=2,alpha=.8)
    a9.plot(*[pos[0,0],pos[0,2]],"o",color=C["cy"],ms=8,zorder=5)
    a9.plot(*[pos[-1,0],pos[-1,2]],"*",color=C["cz"],ms=12,zorder=5)
    plt.colorbar(sc2,ax=a9,label="t(s)",pad=0.02)
    a9.set_xlabel("X (m)"); a9.set_ylabel("Z (m)")
    a9.set_aspect("equal","datalim")

    a10 = fig.add_subplot(gs[2,2], projection="3d")
    a10.set_facecolor("#10131a")
    n = len(pos); step = max(1,n//300)
    for i in range(0,n-step,step):
        c = plt.cm.plasma(i/n)
        a10.plot(pos[i:i+step+1,0],pos[i:i+step+1,1],pos[i:i+step+1,2],
                 color=c,lw=1.1,alpha=.85)
    a10.scatter(*pos[0], color=C["cy"],s=50,zorder=6,label="Start")
    a10.scatter(*pos[-1],color=C["cz"],s=80,marker="*",zorder=6,label="End")
    a10.set_title("3D Trajectory",color="#6e7e9e",fontsize=8,pad=4)
    a10.set_xlabel("X",fontsize=7); a10.set_ylabel("Y",fontsize=7); a10.set_zlabel("Z",fontsize=7)
    a10.legend(fontsize=6); a10.tick_params(colors="#5a607a",labelsize=6)

    # Stats panel
    a11 = ax_(2,3,"Pipeline Summary")
    a11.axis("off")
    stats = (
        f"  Duration     : {ts[-1]:.2f} s\n"
        f"  Samples      : {len(ts)}\n"
        f"  Avg Δt       : {np.mean(np.diff(ts))*1000:.2f} ms\n"
        f"  Sample rate  : {1/np.mean(np.diff(ts)):.1f} Hz\n"
        f"\n"
        f"  ZUPT frames  : {zupt.sum()} ({100*zupt.mean():.1f}%)\n"
        f"  Peak |ω|     : {gm.max():.2f} rad/s\n"
        f"  Peak |a|     : {amag.max():.2f} m/s²\n"
        f"  Peak |V|     : {vmag.max():.3f} m/s\n"
        f"\n"
        f"  Roll₀        : {roll[0]:.2f}°\n"
        f"  Pitch₀       : {pitch[0]:.2f}°\n"
        f"\n"
        f"  Δpos X       : {pos[-1,0]:.3f} m\n"
        f"  Δpos Y       : {pos[-1,1]:.3f} m\n"
        f"  Δpos Z       : {pos[-1,2]:.3f} m\n"
    )
    a11.text(0.05, 0.95, stats, transform=a11.transAxes,
             fontsize=8.5, verticalalignment="top",
             fontfamily="monospace", color="#9aabb8",
             bbox=dict(facecolor="#1a1f2e",edgecolor="#2a3050",
                       boxstyle="round,pad=0.6"))

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[PLOT]  Saved → {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    imu_path    = sys.argv[1] if len(sys.argv)>1 else "imu.txt"
    out_png     = sys.argv[2] if len(sys.argv)>2 else "imu_slam_result.png"
    out_csv     = sys.argv[3] if len(sys.argv)>3 else "imu_trajectory.csv"

    print("="*60)
    print("  IMU SLAM Dead-Reckoning Pipeline  v2")
    print("="*60)

    data = np.loadtxt(imu_path, comments="#")
    print(f"[LOAD]  {data.shape[0]} samples from '{imu_path}'")

    res = run_pipeline(data,
                       madgwick_beta = 0.05,
                       zupt_thresh   = 0.04,
                       zupt_win      = 15,
                       n_static      = 200)

    # CSV export
    t_rel = res["ts"] - res["ts"][0]
    out = np.column_stack([
        t_rel, res["pos"], res["vel"],
        np.degrees(res["roll"]),
        np.degrees(res["pitch"]),
        np.degrees(res["yaw"]),
        res["zupt"].astype(int),
        np.linalg.norm(res["vel"],axis=1)[imu.txt](https://github.com/user-attachments/files/28005460/imu.txt)

    ])
    hdr = "time_s,x_m,y_m,z_m,vx,vy,vz,roll_deg,pitch_deg,yaw_deg,zupt,speed_ms"
    np.savetxt(out_csv, out, delimiter=",", header=hdr, comments="")
    print(f"[CSV]   Saved → {out_csv}")

    plot_results(res, save_path=out_png)

if __name__ == "__main__":
    main()

    
