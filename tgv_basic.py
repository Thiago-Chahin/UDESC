import numpy as np
import matplotlib.pyplot as plt


# ==========================================================================
# SETUP
# ==========================================================================

# parameters
N = 64
L = 2.0 * np.pi
DX = L / N
NU = 0.000625
DT = 0.005
T_FINAL = 20.0
NSTEPS = int(round(T_FINAL / DT))
LOG_EPS = 1.0e-10

METHODS = ["local_temporal", "logarithmic", "mesh_size", "wavenumber"]
METHOD_LABELS = {
    "local_temporal": "Local Temporal",
    "logarithmic": "Logarithmic",
    "mesh_size": "Mesh Size",
    "wavenumber": "Wavenumber",
}
METHOD_COLORS = {
    "local_temporal": "tab:blue",
    "logarithmic": "tab:red",
    "mesh_size": "tab:green",
    "wavenumber": "tab:orange",
}

# space grid
AXIS = np.linspace(0.0, L, N, endpoint=False)
X, Y, Z = np.meshgrid(AXIS, AXIS, AXIS, indexing="ij")

# wavenumber grid
K_AXIS = np.fft.fftfreq(N, d=DX) * 2.0 * np.pi
KX, KY, KZ = np.meshgrid(K_AXIS, K_AXIS, K_AXIS, indexing="ij")
KSQ = KX * KX + KY * KY + KZ * KZ

# safe denominator
KSQ_SAFE = KSQ.copy()
KSQ_SAFE[0, 0, 0] = 1.0

# dealias mask
KMAX = np.max(np.abs(K_AXIS))
DEALIAS_MASK = np.sqrt(KSQ) <= (2.0 / 3.0) * KMAX


# ==========================================================================
# REUSED PROCEDURES
# ==========================================================================

def grad(field_hat):
    # spectral gradient
    fx = np.real(np.fft.ifftn(1j * KX * field_hat))
    fy = np.real(np.fft.ifftn(1j * KY * field_hat))
    fz = np.real(np.fft.ifftn(1j * KZ * field_hat))
    return fx, fy, fz


def project(au, av, aw):
    # remove parallel part
    dot = (KX * au + KY * av + KZ * aw) / KSQ_SAFE
    au = au - KX * dot
    av = av - KY * dot
    aw = aw - KZ * dot
    return au, av, aw


def curl(u, v, w):
    # vorticity field
    uh = np.fft.fftn(u)
    vh = np.fft.fftn(v)
    wh = np.fft.fftn(w)
    ux, uy, uz = grad(uh)
    vx, vy, vz = grad(vh)
    wx, wy, wz = grad(wh)
    ox = wy - vz
    oy = uz - wx
    oz = vx - uy
    return ox, oy, oz


# ==========================================================================
# ADVECTIVE TERM
# ==========================================================================

def advective_term(method, uh, vh, wh, uh_old, vh_old, wh_old):
    # current physical field
    u = np.real(np.fft.ifftn(uh))
    v = np.real(np.fft.ifftn(vh))
    w = np.real(np.fft.ifftn(wh))

    # current gradients
    ux, uy, uz = grad(uh)
    vx, vy, vz = grad(vh)
    wx, wy, wz = grad(wh)

    if method == "full":
        nu = u * ux + v * uy + w * uz
        nv = u * vx + v * vy + w * vz
        nw = u * wx + v * wy + w * wz

    elif method == "local_temporal":
        # previous physical field
        u_old = np.real(np.fft.ifftn(uh_old))
        v_old = np.real(np.fft.ifftn(vh_old))
        w_old = np.real(np.fft.ifftn(wh_old))

        # previous gradients
        ux_old, uy_old, uz_old = grad(uh_old)
        vx_old, vy_old, vz_old = grad(vh_old)
        wx_old, wy_old, wz_old = grad(wh_old)

        # old advects current
        t1u = u_old * ux + v_old * uy + w_old * uz
        t1v = u_old * vx + v_old * vy + w_old * vz
        t1w = u_old * wx + v_old * wy + w_old * wz

        # current advects old
        t2u = u * ux_old + v * uy_old + w * uz_old
        t2v = u * vx_old + v * vy_old + w * vz_old
        t2w = u * wx_old + v * wy_old + w * wz_old

        # old advects old
        t3u = u_old * ux_old + v_old * uy_old + w_old * uz_old
        t3v = u_old * vx_old + v_old * vy_old + w_old * vz_old
        t3w = u_old * wx_old + v_old * wy_old + w_old * wz_old

        nu = t1u + t2u - t3u
        nv = t1v + t2v - t3v
        nw = t1w + t2w - t3w

    elif method == "logarithmic":
        # velocity magnitude
        magnitude = np.sqrt(u * u + v * v + w * w)
        magnitude_safe = magnitude + LOG_EPS

        # unit direction
        dir_u = u / magnitude_safe
        dir_v = v / magnitude_safe
        dir_w = w / magnitude_safe

        # log magnitude gradient
        log_mag_hat = np.fft.fftn(np.log(magnitude_safe))
        lx, ly, lz = grad(log_mag_hat)

        # advective scale
        rate = np.abs(u * lx + v * ly + w * lz)

        nu = rate * magnitude * dir_u
        nv = rate * magnitude * dir_v
        nw = rate * magnitude * dir_w

    elif method == "mesh_size":
        # smooth advecting field
        u_s = np.zeros_like(u)
        v_s = np.zeros_like(v)
        w_s = np.zeros_like(w)
        for axis_index in range(3):
            u_s = u_s + 0.5 * (u + 0.5 * (np.roll(u, 1, axis_index) + np.roll(u, -1, axis_index)))
            v_s = v_s + 0.5 * (v + 0.5 * (np.roll(v, 1, axis_index) + np.roll(v, -1, axis_index)))
            w_s = w_s + 0.5 * (w + 0.5 * (np.roll(w, 1, axis_index) + np.roll(w, -1, axis_index)))
        u_s = u_s / 3.0
        v_s = v_s / 3.0
        w_s = w_s / 3.0

        nu = u_s * ux + v_s * uy + w_s * uz
        nv = u_s * vx + v_s * vy + w_s * vz
        nw = u_s * wx + v_s * wy + w_s * wz

    elif method == "wavenumber":
        # smooth u spectrum
        mag = np.abs(uh)
        phase = np.angle(uh)
        neighbor = np.zeros_like(mag)
        for axis_index in range(3):
            neighbor = neighbor + 0.5 * (np.roll(mag, 1, axis_index) + np.roll(mag, -1, axis_index))
        neighbor = neighbor / 3.0
        uh_s = 0.5 * (mag + neighbor) * np.exp(1j * phase)

        # smooth v spectrum
        mag = np.abs(vh)
        phase = np.angle(vh)
        neighbor = np.zeros_like(mag)
        for axis_index in range(3):
            neighbor = neighbor + 0.5 * (np.roll(mag, 1, axis_index) + np.roll(mag, -1, axis_index))
        neighbor = neighbor / 3.0
        vh_s = 0.5 * (mag + neighbor) * np.exp(1j * phase)

        # smooth w spectrum
        mag = np.abs(wh)
        phase = np.angle(wh)
        neighbor = np.zeros_like(mag)
        for axis_index in range(3):
            neighbor = neighbor + 0.5 * (np.roll(mag, 1, axis_index) + np.roll(mag, -1, axis_index))
        neighbor = neighbor / 3.0
        wh_s = 0.5 * (mag + neighbor) * np.exp(1j * phase)

        u_s = np.real(np.fft.ifftn(uh_s))
        v_s = np.real(np.fft.ifftn(vh_s))
        w_s = np.real(np.fft.ifftn(wh_s))

        nu = u_s * ux + v_s * uy + w_s * uz
        nv = u_s * vx + v_s * vy + w_s * vz
        nw = u_s * wx + v_s * wy + w_s * wz

    else:
        raise ValueError("unknown method: " + method)

    # dealias and project
    au = np.fft.fftn(nu) * DEALIAS_MASK
    av = np.fft.fftn(nv) * DEALIAS_MASK
    aw = np.fft.fftn(nw) * DEALIAS_MASK
    au, av, aw = project(au, av, aw)
    return au, av, aw


# ==========================================================================
# MAIN
# ==========================================================================

print("Re = %.0f, grid = %d^3, dt = %g, t_final = %g (%d steps)"
      % (1.0 / NU, N, DT, T_FINAL, NSTEPS))
print("Reference: full nonlinear TGV on the same grid")
print("")

# storage for the final report
all_metrics = {}
enst_histories = {}
energy_histories = {}
spectra = {}
final_fields = {}

# reference first, then methods
RUN_ORDER = ["full"] + METHODS

for method in RUN_ORDER:

    if method == "full":
        print("running reference (full nonlinear)...")
    else:
        print("running %s..." % METHOD_LABELS[method])

    # ---- initial condition ----
    u = np.sin(X) * np.cos(Y) * np.cos(Z)
    v = -np.cos(X) * np.sin(Y) * np.cos(Z)
    w = np.zeros_like(u)

    uh = np.fft.fftn(u)
    vh = np.fft.fftn(v)
    wh = np.fft.fftn(w)

    uh_old = uh.copy()
    vh_old = vh.copy()
    wh_old = wh.copy()

    # ---- history arrays ----
    times = np.zeros(NSTEPS + 1)
    enst_history = np.zeros(NSTEPS + 1)
    energy_history = np.zeros(NSTEPS + 1)

    ox, oy, oz = curl(u, v, w)
    enst_history[0] = 0.5 * np.mean(ox * ox + oy * oy + oz * oz)
    energy_history[0] = 0.5 * np.mean(u * u + v * v + w * w)

    # ---- time loop ----
    for step in range(NSTEPS):

        # stage 1
        au, av, aw = advective_term(method, uh, vh, wh, uh_old, vh_old, wh_old)
        ru = -au - NU * KSQ * uh
        rv = -av - NU * KSQ * vh
        rw = -aw - NU * KSQ * wh
        u1 = uh + DT * ru
        v1 = vh + DT * rv
        w1 = wh + DT * rw

        # stage 2
        au, av, aw = advective_term(method, u1, v1, w1, uh_old, vh_old, wh_old)
        ru = -au - NU * KSQ * u1
        rv = -av - NU * KSQ * v1
        rw = -aw - NU * KSQ * w1
        u2 = 0.75 * uh + 0.25 * (u1 + DT * ru)
        v2 = 0.75 * vh + 0.25 * (v1 + DT * rv)
        w2 = 0.75 * wh + 0.25 * (w1 + DT * rw)

        # stage 3
        au, av, aw = advective_term(method, u2, v2, w2, uh_old, vh_old, wh_old)
        ru = -au - NU * KSQ * u2
        rv = -av - NU * KSQ * v2
        rw = -aw - NU * KSQ * w2
        un = (1.0 / 3.0) * uh + (2.0 / 3.0) * (u2 + DT * ru)
        vn = (1.0 / 3.0) * vh + (2.0 / 3.0) * (v2 + DT * rv)
        wn = (1.0 / 3.0) * wh + (2.0 / 3.0) * (w2 + DT * rw)

        # shift fields
        uh_old = uh
        vh_old = vh
        wh_old = wh
        uh = un
        vh = vn
        wh = wn

        # record diagnostics
        u = np.real(np.fft.ifftn(uh))
        v = np.real(np.fft.ifftn(vh))
        w = np.real(np.fft.ifftn(wh))
        times[step + 1] = (step + 1) * DT
        ox, oy, oz = curl(u, v, w)
        enst_history[step + 1] = 0.5 * np.mean(ox * ox + oy * oy + oz * oz)
        energy_history[step + 1] = 0.5 * np.mean(u * u + v * v + w * w)

        # stop if unstable
        if not np.isfinite(enst_history[step + 1]):
            print("  unstable at t = %.3f" % times[step + 1])
            break

    # ---- final physical field ----
    u = np.real(np.fft.ifftn(uh))
    v = np.real(np.fft.ifftn(vh))
    w = np.real(np.fft.ifftn(wh))

    # ---- energy spectrum at final time ----
    uh = np.fft.fftn(u)
    vh = np.fft.fftn(v)
    wh = np.fft.fftn(w)
    norm = 1.0 / (N ** 3) ** 2
    density = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2 + np.abs(wh) ** 2) * norm
    shell = np.round(np.sqrt(KSQ)).astype(int)
    shell_max = int(np.max(shell))
    spectrum = np.zeros(shell_max + 1)
    for k in range(shell_max + 1):
        spectrum[k] = np.sum(density[shell == k])

    # ---- the reference is stored, not compared ----
    if method == "full":
        ref_u = u
        ref_v = v
        ref_w = w
        ref_times = times
        ref_enst = enst_history
        ref_energy = energy_history
        ref_spectrum = spectrum
        continue

    # ---- after this part the code was done by Claude ----
    # ---- Prompt: code the error metrics and the printing of figures while matching my style of coding ----
    # ---- error metrics against the reference ----
    count = u.size
    du = u - ref_u
    dv = v - ref_v
    dw = w - ref_w

    mse = (np.sum(du ** 2) + np.sum(dv ** 2) + np.sum(dw ** 2)) / (3.0 * count)
    ae = (np.sum(np.abs(du)) + np.sum(np.abs(dv)) + np.sum(np.abs(dw))) / (3.0 * count)
    max_ae = max(np.max(np.abs(du)), np.max(np.abs(dv)), np.max(np.abs(dw)))

    rel_num = np.sum(du ** 2 + dv ** 2 + dw ** 2)
    rel_den = np.sum(ref_u ** 2 + ref_v ** 2 + ref_w ** 2)
    rel_l2 = np.sqrt(rel_num / rel_den)

    ox, oy, oz = curl(u, v, w)
    rox, roy, roz = curl(ref_u, ref_v, ref_w)
    vort_num = np.sum((ox - rox) ** 2 + (oy - roy) ** 2 + (oz - roz) ** 2)
    vort_den = np.sum(rox ** 2 + roy ** 2 + roz ** 2)
    vort_rel_l2 = np.sqrt(vort_num / vort_den)

    enst = 0.5 * np.mean(ox * ox + oy * oy + oz * oz)
    enst_ref = 0.5 * np.mean(rox * rox + roy * roy + roz * roz)
    enst_err = np.abs(enst - enst_ref) / enst_ref

    dh = 1j * KX * uh + 1j * KY * vh + 1j * KZ * wh
    divergence = np.real(np.fft.ifftn(dh))
    div_err = np.max(np.abs(divergence))

    metrics = {
        "MSE": mse, "AE": ae, "MaxAE": max_ae, "RelL2": rel_l2,
        "VortRelL2": vort_rel_l2, "EnstErr": enst_err, "DivErr": div_err,
    }

    # ---- save this method before the next ----
    filename = "results_" + method + ".npz"
    np.savez(
        filename, method=method, times=times,
        enst_history=enst_history, energy_history=energy_history,
        spectrum=spectrum, u=u, v=v, w=w,
        mse=mse, ae=ae, max_ae=max_ae, rel_l2=rel_l2,
        vort_rel_l2=vort_rel_l2, enst_err=enst_err, div_err=div_err,
    )
    print("  saved %s" % filename)

    # ---- keep for the final report ----
    all_metrics[method] = metrics
    enst_histories[method] = enst_history
    energy_histories[method] = energy_history
    spectra[method] = spectrum
    final_fields[method] = (u, v, w)


# ==========================================================================
# REPORT: METRICS TABLE
# ==========================================================================

print("")
print("Error metrics at t = %g (vs full nonlinear reference)" % T_FINAL)
header = "%-16s %11s %11s %11s %11s %11s %11s %11s" % (
    "Method", "MSE", "AE", "MaxAE", "RelL2", "VortRelL2", "EnstErr", "DivErr")
print(header)
print("-" * len(header))
for method in METHODS:
    row = all_metrics[method]
    print("%-16s %11.3e %11.3e %11.3e %11.3e %11.3e %11.3e %11.3e" % (
        METHOD_LABELS[method], row["MSE"], row["AE"], row["MaxAE"],
        row["RelL2"], row["VortRelL2"], row["EnstErr"], row["DivErr"]))


# ==========================================================================
# REPORT: FIGURE 1, ENSTROPHY HISTORY
# ==========================================================================

plt.figure(figsize=(8, 5))
plt.plot(ref_times, ref_enst, "k-", linewidth=2.5, label="Reference")
for method in METHODS:
    plt.plot(ref_times, enst_histories[method], "--",
             color=METHOD_COLORS[method], label=METHOD_LABELS[method])
plt.xlabel("time")
plt.ylabel("enstrophy")
plt.title("Enstrophy evolution of the Taylor-Green vortex")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fig_enstrophy_history.png", dpi=150)
plt.close()


# ==========================================================================
# REPORT: FIGURE 2, KINETIC ENERGY DECAY
# ==========================================================================

plt.figure(figsize=(8, 5))
plt.plot(ref_times, ref_energy, "k-", linewidth=2.5, label="Reference")
for method in METHODS:
    plt.plot(ref_times, energy_histories[method], "--",
             color=METHOD_COLORS[method], label=METHOD_LABELS[method])
plt.xlabel("time")
plt.ylabel("kinetic energy")
plt.title("Kinetic energy decay")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fig_energy_decay.png", dpi=150)
plt.close()


# ==========================================================================
# REPORT: FIGURE 3, ENERGY SPECTRUM
# ==========================================================================

plt.figure(figsize=(8, 5))
shells = np.arange(1, len(ref_spectrum))
plt.loglog(shells, ref_spectrum[1:], "k-", linewidth=2.5, label="Reference")
for method in METHODS:
    spectrum = spectra[method]
    plt.loglog(shells, spectrum[1:len(shells) + 1], "--",
               color=METHOD_COLORS[method], label=METHOD_LABELS[method])
plt.xlabel("wavenumber k")
plt.ylabel("E(k)")
plt.title("Energy spectrum at t = %g" % T_FINAL)
plt.legend()
plt.grid(True, alpha=0.3, which="both")
plt.tight_layout()
plt.savefig("fig_energy_spectrum.png", dpi=150)
plt.close()


# ==========================================================================
# REPORT: FIGURE 4, ERROR BARS
# ==========================================================================

metric_names = ["MSE", "RelL2", "VortRelL2", "EnstErr"]
positions = np.arange(len(metric_names))
width = 0.2
plt.figure(figsize=(9, 5))
for index, method in enumerate(METHODS):
    values = [all_metrics[method][name] for name in metric_names]
    plt.bar(positions + index * width, values, width,
            color=METHOD_COLORS[method], label=METHOD_LABELS[method])
plt.yscale("log")
plt.xticks(positions + 1.5 * width, metric_names)
plt.ylabel("error (log scale)")
plt.title("Error metrics by method at t = %g" % T_FINAL)
plt.legend()
plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("fig_error_bars.png", dpi=150)
plt.close()


# ==========================================================================
# REPORT: FIGURE 5, VORTICITY SLICES
# ==========================================================================

mid = N // 2

ox, oy, oz = curl(ref_u, ref_v, ref_w)
ref_slice = np.sqrt(ox * ox + oy * oy + oz * oz)[:, :, mid]
vmax = np.max(ref_slice)

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
image = axes[0].imshow(ref_slice.T, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
axes[0].set_title("Reference")
axes[0].set_xticks([])
axes[0].set_yticks([])
for index, method in enumerate(METHODS):
    u, v, w = final_fields[method]
    ox, oy, oz = curl(u, v, w)
    method_slice = np.sqrt(ox * ox + oy * oy + oz * oz)[:, :, mid]
    panel = axes[index + 1]
    panel.imshow(method_slice.T, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
    panel.set_title(METHOD_LABELS[method])
    panel.set_xticks([])
    panel.set_yticks([])
fig.colorbar(image, ax=axes, fraction=0.012, pad=0.01, label="vorticity magnitude")
fig.suptitle("Vorticity magnitude on the mid-plane (z = pi) at t = %g" % T_FINAL)
plt.savefig("fig_vorticity_slices.png", dpi=150, bbox_inches="tight")
plt.close()

print("")
print("figures saved.")
