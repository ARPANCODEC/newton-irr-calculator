# 🧮 Newton's Method (IRR-style) Calculator

**MADE BY ARPAN ARI (arpancodec) — ALL RIGHTS RESERVED 2025**

This is a simple **Streamlit web app** to compute roots of polynomials using **Newton's Method**, with a financial interpretation for **IRR (Internal Rate of Return)** problems.

---

## 📌 Features
- Solve any polynomial of the form:

  \[
  f(\lambda) = -a_0 + a_1\lambda + a_2\lambda^2 + \dots + a_n\lambda^n
  \]

- Iterative solution using Newton’s Method:

  \[
  \lambda_{k+1} = \lambda_k - \frac{f(\lambda_k)}{f'(\lambda_k)}
  \]

- Step-by-step iteration table (λₖ, f(λₖ), f′(λₖ), error).
- Function plot around the estimated root.
- Automatic **IRR calculation** (if λ > 0):

  \[
  IRR = \frac{1}{\lambda} - 1
  \]

- Sidebar presets for quick demos:
  - **PJ1 Demo** → f(λ) = -1 + λ + λ²
  - **Cubic Demo** → f(λ) = -2 + λ³

---

## 📦 Installation

1. Clone or download this repo.

2. Install dependencies (Python 3.9+ recommended):

   ```bash
   pip install -r requirements.txt
