import numpy as np
import matplotlib.pyplot as plt

# ---------------- constants / geometry ----------------
au = 1.496e11
do = 40 * au

lb, dlb = 445e-9, 94e-9
lam_min, lam_max = lb - 0.5*dlb, lb + 0.5*dlb
nlam = 31
lams = np.linspace(lam_min, lam_max, nlam)
weights = np.ones_like(lams); weights /= weights.sum()

fs0 = np.sqrt(lb * do / 2)
print(f"Fresnel scale at 445 nm, 40 AU: {fs0/1e3:.2f} km")

# ---------------- base grid (object plane) ----------------
N  = 512
dx = fs0 / 20
L  = N * dx
x  = np.linspace(-L/2, L/2, N)
y  = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

# ---------------- occulter parameters (meters) ----------------
radius_lobe = 2400.0       # radius of each lobe
sep = 2.0 * radius_lobe    # center-to-center separation for 2-lobe case
x0, y0 = 0.0, 0.0          # reference center

print(f"Lobe radius: {radius_lobe:.1f} m; sep={sep:.1f} m; dx={dx:.2f} m")

# ---------------- strong zero-padding ----------------
pad = 4
Np  = pad * N
dxp = dx

# frequency grid
fx = np.fft.fftfreq(Np, d=dxp)
fy = np.fft.fftfreq(Np, d=dxp)
FX, FY = np.meshgrid(fx, fy)

# impact parameters (same for all rows)
b_list_km = np.array([-4.0, -1.0, +1.75, +3.5])
b_list = b_list_km * 1e3
colors = ['cyan', 'lime', 'purple', 'red']
labels = [f"T{i+1}: b={bk:.2f} km" for i, bk in enumerate(b_list_km)]

# ---------------- configuration for each row (easy to edit) ----------------
configs = [
    {"angle_deg": 0.0,  "num_lobes": 2},
    {"angle_deg": 0.0, "num_lobes": 2},
    {"angle_deg": 90.0, "num_lobes": 2},
    # examples to try:
    # {"angle_deg": 0.0, "num_lobes": 1},
    # {"angle_deg": 30.0, "num_lobes": 1},
]

n_rows = len(configs)

# ---------------- helper: build mask for arbitrary number of lobes ----------

def build_lobe_mask(X, Y, radius, num_lobes=2, angle_deg=0.0,
                    sep=None, x0=0.0, y0=0.0):
    """
    Build an occulting mask of one or more circular lobes.

    num_lobes = 1: single disk centered at (x0, y0)
    num_lobes = 2: 'contact binary' with separation sep along angle_deg
    num_lobes >2: arranged evenly on a ring of radius sep/2
    """
    if sep is None:
        sep = 2.0 * radius

    theta = np.deg2rad(angle_deg)
    mask = np.zeros_like(X, dtype=bool)
    centers = []

    if num_lobes == 1:
        centers = [(x0, y0)]
    elif num_lobes == 2:
        dxc = (sep / 2.0) * np.cos(theta)
        dyc = (sep / 2.0) * np.sin(theta)
        centers = [(x0 - dxc, y0 - dyc),
                   (x0 + dxc, y0 + dyc)]
    else:
        ring_R = sep / 2.0
        for k in range(num_lobes):
            phi = theta + 2*np.pi * k / num_lobes
            cx = x0 + ring_R * np.cos(phi)
            cy = y0 + ring_R * np.sin(phi)
            centers.append((cx, cy))

    for (cx, cy) in centers:
        disk = ((X - cx)**2 + (Y - cy)**2) <= radius**2
        mask |= disk

    return mask.astype(float), centers

# ---------------- figure: 3 × N grid ----------------
fig, axes = plt.subplots(
    n_rows, 3,
    figsize=(17, 4.5 * n_rows),
    constrained_layout=True
)

if n_rows == 1:
    axes = np.array([axes])

for row_idx, cfg in enumerate(configs):
    angle_deg = cfg.get("angle_deg", 0.0)
    num_lobes = cfg.get("num_lobes", 1)

    ax_mask, ax_diff, ax_lc = axes[row_idx, 0], axes[row_idx, 1], axes[row_idx, 2]

    # ---------- build mask ----------
    mask, centers = build_lobe_mask(
        X, Y,
        radius=radius_lobe,
        num_lobes=num_lobes,
        angle_deg=angle_deg,
        sep=sep,
        x0=x0, y0=y0
    )
    A = 1.0 - mask

    # embed in padded aperture
    Ap = np.zeros((Np, Np), dtype=float)
    i0 = (Np - N) // 2
    Ap[i0:i0+N, i0:i0+N] = A

    # ---------- Fresnel propagation ----------
    I_poly = np.zeros((Np, Np))
    for lam, w in zip(lams, weights):
        H  = np.exp(-1j * np.pi * lam * do * (FX**2 + FY**2))
        I  = np.abs(np.fft.ifft2(np.fft.fft2(Ap) * H))**2
        m = 64
        edge = np.r_[I[:m].ravel(), I[-m:].ravel(), I[:, :m].ravel(), I[:, -m:].ravel()]
        I /= np.median(edge)
        I_poly += w * I

    crop = 64
    I_poly_c = I_poly[i0+crop:i0+N-crop, i0+crop:i0+N-crop]
    x_c = x[crop:N-crop]
    y_c = y[crop:N-crop]
    x_profile = x_c.copy()

    # ---------- column 1: MASK ----------
    im1 = ax_mask.imshow(
        mask, cmap='gray_r',
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin='lower'
    )
    if len(centers) >= 2:
        xs, ys = zip(*centers)
        ax_mask.plot(xs, ys, 'k--')
    xs, ys = zip(*centers)
    ax_mask.scatter(xs, ys, c='k')

    ax_mask.set_title(f"Mask: {num_lobes} lobe(s), angle={angle_deg:.0f}°")
    ax_mask.set_xlabel("X (m)")
    ax_mask.set_ylabel("Y (m)")
    fig.colorbar(im1, ax=ax_mask, fraction=0.05, pad=0.04)

    # ---------- column 2: DIFFRACTION ----------
    im2 = ax_diff.imshow(
        np.log10(I_poly_c + 1e-12), cmap='magma',
        extent=[x_c.min(), x_c.max(), y_c.min(), y_c.max()],
        origin='lower'
    )
    ax_diff.set_title("Diffraction")
    ax_diff.set_xlabel("X (m)")
    ax_diff.set_ylabel("Y (m)")
    fig.colorbar(im2, ax=ax_diff, fraction=0.05, pad=0.04)

    # impact-parameter lines + legend on diffraction plots
    for bk, col, lab in zip(b_list_km, colors, labels):
        ax_diff.axhline(bk * 1e3, color=col, lw=1, label=lab)
    ax_diff.legend(fontsize=7, loc='upper right', framealpha=0.7)

    # ---------- column 3: LIGHT CURVES ----------
    ax_lc.set_title("Light Curves")
    x_min_km, x_max_km = -5, 5
    mask_x = (x_profile >= x_min_km*1e3) & (x_profile <= x_max_km*1e3)

    for bk, col, lab in zip(b_list_km, colors, labels):
        b = bk * 1e3
        if not (y_c.min() <= b <= y_c.max()):
            continue
        j = np.argmin(np.abs(y_c - b))
        lc_row = I_poly_c[j]

        n = lc_row.size
        k = int(0.15 * n)
        wings = np.r_[lc_row[:k], lc_row[-k:]]
        baseline = np.median(wings)
        lc = lc_row / baseline

        ax_lc.plot(x_profile[mask_x]/1e3, lc[mask_x],
                   color=col, lw=1.5, label=lab)

    ax_lc.set_xlabel("Distance (km)")
    ax_lc.set_ylabel("Normalized Intensity")
    ax_lc.grid(alpha=0.3)
    ax_lc.set_xlim(x_min_km, x_max_km)
    ax_lc.legend(fontsize=7, ncol=2, loc='best')

# Title
fig.suptitle("Occultation Masks, Diffraction, and Light Curves", fontsize=15)
plt.show()
