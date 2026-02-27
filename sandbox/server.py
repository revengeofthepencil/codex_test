import os
import builtins
import contextlib
import io
import multiprocessing as mp
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

MAX_SECONDS = float(os.getenv("SANDBOX_MAX_SECONDS", "5"))
WORKDIR = os.getenv("SANDBOX_WORKDIR", "/workspace")
OUTPUT_DIR = os.getenv("SANDBOX_OUTPUT_DIR", "/outputs")
_ALLOWED_IMPORTS = os.getenv("SANDBOX_ALLOWED_IMPORTS", "numpy,math,json").strip()
ALLOWED_IMPORTS = {m.strip() for m in _ALLOWED_IMPORTS.split(",") if m.strip()}

SESSIONS: Dict[str, Dict[str, Any]] = {}


class ExecRequest(BaseModel):
    session_id: str
    code: str


class ExecResponse(BaseModel):
    ok: bool
    stdout: str
    new_vars: Dict[str, Any]
    error: Optional[str] = None


def _session_dir(session_id: str) -> str:
    safe = "".join(ch for ch in session_id if ch.isalnum() or ch in ("-", "_"))
    if not safe:
        safe = "default"
    return os.path.join(WORKDIR, safe)


def _jsonable(val: Any) -> Any:
    if isinstance(val, (int, float, str, bool)) or val is None:
        return val
    if isinstance(val, dict):
        out = {}
        for k, v in val.items():
            if isinstance(k, str):
                jv = _jsonable(v)
                if jv is not None:
                    out[k] = jv
        return out
    if isinstance(val, list):
        out = []
        for v in val:
            jv = _jsonable(v)
            if jv is not None:
                out.append(jv)
        return out
    if isinstance(val, tuple):
        out = [_jsonable(v) for v in val]
        out = [v for v in out if v is not None]
        return out
    if isinstance(val, set):
        out = [_jsonable(v) for v in val]
        out = [v for v in out if v is not None]
        return out
    if isinstance(val, np.ndarray):
        return val.tolist()
    return None


def _make_safe_import(allowed: set[str]):
    real_import = builtins.__import__

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        top = name.split(".")[0]
        if top not in allowed:
            raise ImportError(f"Import blocked: '{top}' not in allowlist")
        return real_import(name, globals, locals, fromlist, level)

    return safe_import


def _make_safe_open(session_path: str):
    real_open = builtins.open

    def safe_open(file, mode="r", *args, **kwargs):
        write_mode = any(ch in mode for ch in ("w", "a", "x", "+"))
        if write_mode:
            if not isinstance(file, (str, bytes, os.PathLike)):
                raise PermissionError("File writes blocked (invalid path)")
            p = os.path.abspath(os.fspath(file))
            if not p.startswith(session_path + os.sep):
                raise PermissionError(f"File writes blocked outside {session_path}")
        return real_open(file, mode, *args, **kwargs)

    return safe_open


def _safe_builtins(session_path: str, allowed_imports: set[str]) -> Dict[str, Any]:
    safe_import = _make_safe_import(allowed_imports)
    safe_open = _make_safe_open(session_path)

    return {
        "print": builtins.print,
        "len": builtins.len,
        "range": builtins.range,
        "enumerate": builtins.enumerate,
        "zip": builtins.zip,
        "int": builtins.int,
        "float": builtins.float,
        "str": builtins.str,
        "bool": builtins.bool,
        "list": builtins.list,
        "dict": builtins.dict,
        "set": builtins.set,
        "tuple": builtins.tuple,
        "min": builtins.min,
        "max": builtins.max,
        "sum": builtins.sum,
        "abs": builtins.abs,
        "round": builtins.round,
        "Exception": builtins.Exception,
        "ValueError": builtins.ValueError,
        "TypeError": builtins.TypeError,
        "KeyError": builtins.KeyError,
        "__import__": safe_import,
    }


# pulled from https://www.askpython.com/python/examples/principal-component-analysis
def run_pca_tool(X: list[list[float]], num_components: int) -> np.ndarray:
    """
    Perform Principal Component Analysis (PCA) on a dataset.

    Args:
        X (list[list[float]]): The input dataset as a 2D list or array-like structure, where each row is a sample and each column is a feature.
        num_components (int): The number of principal components to retain.

    Returns:
        np.ndarray: The reduced dataset with shape (n_samples, num_components).

    Example:
        >>> data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        >>> reduced = run_pca_tool(data, 2)
        >>> print(reduced.shape)
        (3, 2)
    """
    arr = np.array(X)
    print(f"Running PCA on dataset with shape {arr.shape} to reduce to {num_components} components.")

    # Step 1: Center the data
    X_meaned = arr - np.mean(arr, axis=0)

    # Step 2: Compute covariance matrix
    cov_mat = np.cov(X_meaned, rowvar=False)

    # Step 3: Compute eigenvalues and eigenvectors
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

    # Step 4: Sort eigenvalues and eigenvectors
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]

    # Step 5: Select subset of eigenvectors
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]

    # Step 6: Transform the data
    X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()

    return X_reduced



def save_json_tool(filename: str, data: Any) -> str:
    """
    Save JSON-serializable `data` into the shared /outputs directory.
    Returns the full path written.
    """
    import json
    from datetime import datetime
    now = datetime.now()
    # append the date string to the file name
    date_str = now.strftime("%Y-%m-%d_%H%M")
    name, ext = os.path.splitext(filename)
    filename = f"{name}_{date_str}{ext}"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_name = os.path.basename(filename)
    path = os.path.join(OUTPUT_DIR, safe_name)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path
    
def _worker(code: str, locals_in: Dict[str, Any], session_path: str, q: mp.Queue) -> None:
    try:
        stdout_buf = io.StringIO()

        globals_dict = {
            "__builtins__": _safe_builtins(session_path=session_path, allowed_imports=ALLOWED_IMPORTS),
            "np": np,
            "run_pca_tool": run_pca_tool,
            "save_json_tool": save_json_tool,
        }
        locals_dict = dict(locals_in)

        try:
            with contextlib.redirect_stdout(stdout_buf):
                exec(code, globals_dict, locals_dict)
            out = stdout_buf.getvalue() or "<code ran, no output printed to stdout>"
            
            # Clean up locals for pickling
            safe_locals = {k: _jsonable(v) for k, v in locals_dict.items() if _jsonable(v) is not None}
            q.put((True, out, safe_locals, None))
        except Exception as e:
            out = stdout_buf.getvalue()
            safe_locals = {k: _jsonable(v) for k, v in locals_dict.items() if _jsonable(v) is not None}
            q.put((False, out, safe_locals, repr(e)))

    except BaseException as e:
        # If anything goes wrong before we can exec (or even create globals),
        # never let the parent hang waiting for q.get()
        try:
            q.put((False, "", {}, f"Sandbox worker crashed early: {repr(e)}"))
        except Exception:
            pass

@app.post("/exec", response_model=ExecResponse)
def exec_code(req: ExecRequest) -> ExecResponse:
    if not req.session_id:
        raise HTTPException(status_code=400, detail="session_id required")

    session_path = _session_dir(req.session_id)
    os.makedirs(session_path, exist_ok=True)

    session = SESSIONS.setdefault(req.session_id, {})
    original_keys = set(session.keys())

    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_worker, args=(req.code, session, session_path, q))
    p.start()
    p.join(timeout=MAX_SECONDS)

    if p.is_alive():
        p.kill()
        p.join()
        return ExecResponse(ok=False, stdout="", new_vars={}, error=f"Timeout after {MAX_SECONDS} seconds")

    try:
        ok, stdout, locals_out, err = q.get(timeout=1.0)
    except Exception:
        # Worker exited but never sent a message
        return ExecResponse(ok=False, stdout="", new_vars={}, error="Sandbox worker failed (no response on queue)")

    for k, v in locals_out.items():
        jv = _jsonable(v)
        if jv is not None:
            session[k] = jv

    new_keys = set(session.keys()) - original_keys
    new_vars = {k: session[k] for k in new_keys}

    if not ok:
        return ExecResponse(ok=False, stdout=stdout or "", new_vars=new_vars, error=err)

    return ExecResponse(ok=True, stdout=stdout or "", new_vars=new_vars)


@app.post("/reset/{session_id}")
def reset_session(session_id: str):
    SESSIONS.pop(session_id, None)
    path = _session_dir(session_id)
    try:
        for root, dirs, files in os.walk(path, topdown=False):
            for f in files:
                try:
                    os.remove(os.path.join(root, f))
                except Exception:
                    pass
            for d in dirs:
                try:
                    os.rmdir(os.path.join(root, d))
                except Exception:
                    pass
        try:
            os.rmdir(path)
        except Exception:
            pass
    except Exception:
        pass
    return {"ok": True}
