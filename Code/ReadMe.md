````md
# Running the 200m Freestyle Optimizer (Streamlit App)

This README explains **exactly** how to install requirements and run `app.py` on Windows (your setup), plus quick fixes for the most common Streamlit errors.

---

## What you need

- **Python 3.9+** (Python 3.11 is perfect)
- A terminal (PowerShell or Command Prompt)
- This project folder contains:
  - `app.py`
  - (optional) `requirements.txt`

---

## 1) Open a terminal in the project folder

Example (your path):
```bash
cd "C:\Users\shaya\OneDrive\Desktop\Calc optimization research\code"
````

---

## 2) (Recommended) Create and activate a virtual environment

This prevents version conflicts.

### PowerShell or CMD:

```bash
python -m venv .venv
.venv\Scripts\activate
```

You should now see `(.venv)` in your terminal prompt.

### If activation is blocked (PowerShell)

Run this once:

```bash
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again:

```bash
.venv\Scripts\activate
```

---

## 3) Install dependencies

### Option A — Install in one command

```bash
pip install streamlit numpy pandas matplotlib pillow reportlab
```

### Option B — Use `requirements.txt` (cleaner)

Create `requirements.txt` in the same folder as `app.py`:

```txt
streamlit
numpy
pandas
matplotlib
pillow
reportlab
```

Then install:

```bash
pip install -r requirements.txt
```

---

## 4) Upgrade Streamlit (important)

If you saw this error:

> `TypeError: ImageMixin.image() got an unexpected keyword argument 'use_container_width'`

That means your Streamlit version is older.

Upgrade:

```bash
python -m pip install --upgrade streamlit
```

---

## 5) Run the app

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually):

* [http://localhost:8501](http://localhost:8501)

Open it in your browser.

---

## If the `use_container_width` error still appears

### Fix it in code (works on older Streamlit)

In `app.py`, replace:

```python
st.image(wf, caption="Workflow diagram", use_container_width=True)
```

with:

```python
st.image(wf, caption="Workflow diagram", use_column_width=True)
```

### Best version-safe fix

Use both (auto-fallback):

```python
try:
    st.image(wf, caption="Workflow diagram", use_container_width=True)
except TypeError:
    st.image(wf, caption="Workflow diagram", use_column_width=True)
```

---

## Common troubleshooting

### A) `streamlit` not recognized / command not found

Run Streamlit via Python:

```bash
python -m streamlit run app.py
```

### B) Wrong Python environment / wrong packages

Check:

```bash
where python
python --version
python -c "import streamlit as st; print(st.__version__)"
```

Make sure you activated your venv first:

```bash
.venv\Scripts\activate
```

### C) Port already in use

Run on another port:

```bash
streamlit run app.py --server.port 8502
```

### D) Reinstall everything (clean reset)

```bash
pip uninstall -y streamlit numpy pandas matplotlib pillow reportlab
pip install streamlit numpy pandas matplotlib pillow reportlab
```

---

## Quick “fastest possible” run (no venv)

If you don’t care about virtual environments:

```bash
pip install streamlit numpy pandas matplotlib pillow reportlab
streamlit run app.py
```

---

## Optional: add GitHub-friendly files

### `requirements.txt`

```txt
streamlit
numpy
pandas
matplotlib
pillow
reportlab
```

---

## Verify your setup (1-minute check)

```bash
python -c "import streamlit as st; import numpy; import pandas; import matplotlib; import PIL; import reportlab; print('OK', st.__version__)"
```

If that prints `OK ...` you’re ready.

---

```
::contentReference[oaicite:0]{index=0}
```
