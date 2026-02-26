import numpy as np
import pytest
from main import add, eval as sandbox_eval, run_pca_tool

class TestSandboxEval:
    def test_captures_stdout(self):
        result, _ = sandbox_eval("print('hello')", {})
        assert result == "hello\n"

    def test_no_output_message(self):
        result, _ = sandbox_eval("x = 1 + 1", {})
        assert result == "<code ran, no output printed to stdout>"

    def test_returns_new_serializable_vars(self):
        _, new_vars = sandbox_eval("x = 42\ny = 'abc'", {})
        assert new_vars == {"x": 42, "y": "abc"}

    def test_excludes_non_serializable_vars(self):
        _, new_vars = sandbox_eval("import math\ndef foo(): pass", {})
        assert "math" not in new_vars
        assert "foo" not in new_vars

    def test_catches_exceptions(self):
        result, _ = sandbox_eval("raise ValueError('oops')", {})
        assert "ValueError" in result
        assert "oops" in result

    def test_does_not_include_preexisting_vars(self):
        _, new_vars = sandbox_eval("y = 10", {"x": 1})
        assert "x" not in new_vars
        assert new_vars == {"y": 10}


class TestRunPcaTool:
    def test_output_shape(self):
        data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        result = run_pca_tool(data, num_components=2)
        assert result.shape == (4, 2)

    def test_single_component(self):
        data = [[1, 2], [3, 4], [5, 6], [7, 8]]
        result = run_pca_tool(data, num_components=1)
        assert result.shape == (4, 1)

    def test_returns_ndarray(self):
        data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        result = run_pca_tool(data, num_components=1)
        assert isinstance(result, np.ndarray)

    def test_reduced_data_is_centered(self):
        # PCA output should have near-zero mean
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        result = run_pca_tool(data, num_components=2)
        assert np.abs(result.mean(axis=0)).max() == pytest.approx(0.0, abs=1e-10)
