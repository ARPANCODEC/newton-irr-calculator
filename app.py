# ============================================================
# app.py — Newton's Method (IRR-style) Root Finder
# MADE BY ARPAN ARI (arpancodec) — ALL RIGHTS RESERVED 2025
# ============================================================
# This Streamlit app computes a root of
#   f(λ) = -a0 + a1*λ + a2*λ^2 + ... + an*λ^n
# via Newton's Method, shows all iterates in a table,
# plots f(λ) near the root, and (optionally) reports IRR:
#   IRR = 1/λ - 1  (if λ > 0)
#
# Example (PJ1):
#   f(λ) = -1 + λ + λ^2  →  coeffs: [1, 1, 1]
#
# Run:
#   streamlit run app.py
#
# MADE BY ARPAN ARI (arpancodec) — ALL RIGHTS RESERVED 2025
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------- Page Setup ----------------------
st.set_page_config(
    page_title="Newton's Method (IRR-style) Calculator",
    page_icon="🧮",
    layout="centered"
)

st.title("🧮 Newton's Method (IRR-style) Calculator")
st.caption("MADE BY **ARPAN ARI (arpancodec)** — **ALL RIGHTS RESERVED 2025**")

st.write(
    "Solve roots of the polynomial  "
    r"$f(\lambda) = -a_0 + a_1\lambda + a_2\lambda^2 + \dots + a_n\lambda^n$ "
    "using Newton's Method, view iterates, and (optionally) compute **IRR** via "
    r"$\text{IRR} = \frac{1}{\lambda} - 1$."
)

# ---------------- Helper functions ----------------
def eval_poly_and_derivative(lmbda: float, coeffs: list[float]) -> tuple[float, float]:
    """
    Evaluate f(λ) and f'(λ) for:
        f(λ) = -a0 + a1*λ + a2*λ^2 + ... + an*λ^n
    coeffs = [a0, a1, ..., an]
    """
    a = np.array(coeffs, dtype=float)
    n = len(a) - 1

    # Powers [λ^0, λ^1, ..., λ^n]
    powers = np.array([lmbda**k for k in range(n + 1)], dtype=float)

    # f(λ) = -a0 + sum_{k=1..n} a_k * λ^k
    f_val = (-a[0]) + np.dot(a[1:], powers[1:]) if n >= 1 else (-a[0])

    # f'(λ) = a1 + 2*a2*λ + ... + n*an*λ^{n-1}
    if n >= 1:
        ks = np.arange(1, n + 1, dtype=float)
        fprime_val = np.dot(ks * a[1:], powers[:-1])
    else:
        fprime_val = 0.0

    return float(f_val), float(fprime_val)

def polynomial_latex(coeffs: list[float]) -> str:
    """Return a LaTeX string for f(λ) in the app's format."""
    terms = [f"-{coeffs[0]:g}"]  # -a0 term
    for k, ak in enumerate(coeffs[1:], start=1):
        if ak == 0:
            continue
        if k == 1:
            terms.append(f"{ak:g}\\lambda")
        else:
            terms.append(f"{ak:g}\\lambda^{k}")
    # make "+ -" look like "-"
    return " + ".join(terms).replace("+ -", "- ")

def parse_coeffs(text: str) -> list[float]:
    tokens = text.replace(",", " ").split()
    return [float(t) for t in tokens]

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("Inputs")

coeffs_text = st.sidebar.text_input(
    "Coefficients [a0, a1, ..., an]",
    value="1, 1, 1",
    help="Example: 1, 1, 1 → f(λ) = -1 + 1·λ + 1·λ² (PJ1)."
)

lmbda0 = st.sidebar.number_input("Initial guess λ₀", value=1.0, step=0.1)
tolerance = st.sidebar.number_input("Tolerance (|Δλ| stop)", value=1e-10, step=1e-10, format="%.1e")
max_iter = st.sidebar.number_input("Maximum iterations", min_value=1, value=50, step=1)

show_irr = st.sidebar.checkbox("Show IRR (if λ>0): IRR = 1/λ - 1", value=True)

col_preset1, col_preset2 = st.sidebar.columns(2)
with col_preset1:
    demo1 = st.button("Load PJ1 Demo", use_container_width=True)
with col_preset2:
    demo2 = st.button("Load Cubic Demo", use_container_width=True)

run = st.sidebar.button("Run Newton's Method", type="primary", use_container_width=True)

# ---------------- Handle Presets ----------------
if demo1:
    coeffs_text = "1, 1, 1"   # f(λ) = -1 + λ + λ²
    lmbda0 = 1.0
    st.sidebar.success("Loaded PJ1 demo: f(λ) = -1 + λ + λ²")

if demo2:
    coeffs_text = "2, 0, 0, 1"  # f(λ) = -2 + λ³
    lmbda0 = 1.0
    st.sidebar.success("Loaded Cubic demo: f(λ) = -2 + λ³")

# ---------------- Parse Coeffs ----------------
parse_error = None
try:
    coeffs = parse_coeffs(coeffs_text)
    if len(coeffs) < 2:
        parse_error = "Please provide at least [a0, a1]."
except Exception as e:
    parse_error = f"Could not parse coefficients: {e}"

if parse_error:
    st.error(parse_error)
    st.stop()

# ---------------- Show polynomial ----------------
st.markdown(r"**Using:** $f(\lambda) = " + polynomial_latex(coeffs) + "$")

# ---------------- Run Newton's Method ----------------
if run:
    rows = []
    lmbda = float(lmbda0)
    converged = False

    for k in range(int(max_iter)):
        f_val, fprime_val = eval_poly_and_derivative(lmbda, coeffs)

        if abs(fprime_val) < 1e-14:
            st.warning(
                f"Stopped at iteration {k}: derivative too small (f'(λ)≈0). "
                "Try a different initial guess."
            )
            break

        next_lmbda = lmbda - f_val / fprime_val

        rows.append({
            "k": k,
            "λ_k": lmbda,
            "f(λ_k)": f_val,
            "f'(λ_k)": fprime_val,
            "λ_{k+1}": next_lmbda,
            "|Δλ|": abs(next_lmbda - lmbda),
        })

        if abs(next_lmbda - lmbda) < tolerance:
            lmbda = next_lmbda
            converged = True
            rows.append({
                "k": k + 1,
                "λ_k": lmbda,
                "f(λ_k)": eval_poly_and_derivative(lmbda, coeffs)[0],
                "f'(λ_k)": eval_poly_and_derivative(lmbda, coeffs)[1],
                "λ_{k+1}": np.nan,
                "|Δλ|": 0.0,
            })
            break

        lmbda = next_lmbda

    # ---- Results table ----
    if rows:
        df = pd.DataFrame(rows)
        st.subheader("Iterations")
        st.dataframe(
            df.style.format({
                "λ_k": "{:.12f}",
                "f(λ_k)": "{:.12e}",
                "f'(λ_k)": "{:.12e}",
                "λ_{k+1}": "{:.12f}",
                "|Δλ|": "{:.3e}",
            })
        )

        # ---- Final status ----
        if converged:
            st.success(f"Converged to λ ≈ {lmbda:.12f} within tolerance {tolerance:g}.")
            if show_irr and lmbda > 0:
                irr = 1.0 / lmbda - 1.0
                st.info(f"IRR = 1/λ - 1 = **{irr:.12f}**  →  **{irr*100:.8f}%**")
            elif show_irr:
                st.warning("IRR not shown because λ ≤ 0 (requires λ > 0).")
        else:
            st.warning("Did not meet tolerance within the given maximum iterations.")

        # ---- Plot near the estimated root ----
        st.subheader("Function Plot near the estimated root")
        center = lmbda if np.isfinite(lmbda) else float(lmbda0)
        span = 2.0
        xs = np.linspace(center - span, center + span, 400)
        ys = [eval_poly_and_derivative(float(x), coeffs)[0] for x in xs]

        fig = plt.figure()
        plt.axhline(0, linewidth=1)
        plt.plot(xs, ys, linewidth=2)
        plt.scatter([center], [eval_poly_and_derivative(center, coeffs)[0]], s=50)
        plt.xlabel("λ")
        plt.ylabel("f(λ)")
        plt.title("f(λ) around the estimated root")
        st.pyplot(fig)
    else:
        st.info("No iterations were performed. Check your inputs and try again.")

else:
    st.info("Enter coefficients and press **Run Newton's Method** to begin. Use sidebar presets for quick demos.")

# ---------------- Footer credit ----------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; font-size:0.9rem;'>"
    "MADE BY <b>ARPAN ARI (arpancodec)</b> &nbsp;•&nbsp; <b>ALL RIGHTS RESERVED 2025</b>"
    "</div>",
    unsafe_allow_html=True
)
