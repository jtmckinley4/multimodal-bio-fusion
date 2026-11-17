"""Run batch shells with fake module/Conda/Python commands; no research executes."""

from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT if (ROOT / "run_encoder_swaps.py").is_file() else ROOT / "execution/arcc/encoder_swaps"
TEMPLATES = (
    "newton_swaps.sbatch.example", "newton_preflight.sbatch.example",
    "newton_report.sbatch.example",
)
BASH = (str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe")
        if sys.platform == "win32" else shutil.which("bash"))

HOOK = r'''
conda() {
    case "$1" in
        info) printf '%s\n' "$TEST_BASE" ;;
        deactivate) CONDA_SHLVL=$((CONDA_SHLVL - 1)); export CONDA_SHLVL ;;
        activate) export CONDA_SHLVL=1 CONDA_PREFIX="$2"; printf 'Activated fixture\n' ;;
        *) return 98 ;;
    esac
}
'''

HARNESS = r'''
# Before initialization this stand-in behaves like the bare conda executable:
# it can report its base but cannot deactivate the parent shell's environment.
conda() {
    if [[ "$1" == "info" ]]; then printf '%s\n' "$TEST_BASE"; return; fi
    printf 'CondaError: shell not initialized\n' >&2
    return 91
}
module() {
    printf 'Module cleanup\n'
    for ((attempt=0; attempt < 4 && ${CONDA_SHLVL:-0} > 0; attempt++)); do
        conda deactivate || return 91
    done
    [[ ${CONDA_SHLVL:-0} -eq 0 ]] || return 92
}
python() {
    [[ "${CONDA_PREFIX:-}" == "$ARCC_CONDA_ENV" ]] || return 93
    printf 'Fixture Python: '
    printf '<%s>' "$@"
    printf '\n'
    if [[ "$*" == *"--verify-only"* ]]; then
        [[ "${TEST_VERIFY_FAILURE:-0}" == 0 ]] || return 86
    fi
}
hostname() { printf 'fixture-host\n'; }
'''


@unittest.skipUnless(BASH and Path(BASH).is_file(), "Bash is required for shell fixtures")
class BatchStartupTests(unittest.TestCase):
    def run_template(self, template, level, *, missing_hook=False, overrides=None,
                     batch_rows=None, missing_logs=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            if not missing_logs:
                (root / "outputs/logs").mkdir(parents=True)
            hook = root / "base/etc/profile.d/conda.sh"
            hook.parent.mkdir(parents=True)
            if not missing_hook:
                hook.write_text(HOOK, encoding="utf-8", newline="\n")
            env = {
                key: value for key, value in os.environ.items()
                if not key.startswith(("CONDA", "_CONDA", "BASH_FUNC_", "ARCC_", "SLURM_"))
                and key not in {"BASH_ENV", "ENV"}
            }
            env.update({
                "TEST_BASE": (root / "base").as_posix(),
                "ARCC_PROJECT_ROOT": root.as_posix(),
                "ARCC_ANACONDA_MODULE": "fixture/anaconda",
                "ARCC_CONDA_ENV": (root / "wanted-env").as_posix(),
                "ARCC_KERNEL": "fixture-kernel", "ARCC_EXPECTED_GPU": "fixture-gpu",
                "ARCC_BENCHMARK": "ecoli_proteins", "ARCC_SEED": "0",
                "SLURM_JOB_ID": "fixture", "CONDA_SHLVL": str(level),
                "CONDA_PREFIX": "/inherited/environment",
            })
            for key, value in (overrides or {}).items():
                if value is None:
                    env.pop(key, None)
                else:
                    env[key] = value.replace("{root}", root.as_posix())
            if batch_rows is not None:
                batch = root / "selected batches.tsv"
                batch.write_text("array_index\tbenchmark\tseed\n" + batch_rows,
                                 encoding="utf-8", newline="\n")
                env["ARCC_BATCHES"] = batch.as_posix()
            if env.get("ARCC_OUTPUT_ROOT"):
                (Path(env["ARCC_OUTPUT_ROOT"]) / "logs").mkdir(parents=True)
            command = HARNESS + "\nsource " + shlex.quote((WORKSPACE / template).as_posix())
            return subprocess.run(
                [BASH, "--noprofile", "--norc", "-c", command],
                env=env, capture_output=True, text=True, timeout=15,
            )

    def test_inherited_activation_levels_are_cleared_before_new_activation(self):
        for template in TEMPLATES:
            for level in (1, 2):
                with self.subTest(template=template, inherited_level=level):
                    result = self.run_template(template, level)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("Activated fixture", result.stdout)
                    self.assertIn("Fixture Python:", result.stdout)
                    self.assertNotIn("CondaError", result.stderr)

    def test_clean_shell_still_loads_and_activates(self):
        for template in TEMPLATES:
            with self.subTest(template=template):
                result = self.run_template(template, 0)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Fixture Python:", result.stdout)

    def test_missing_inherited_hook_stops_before_module_and_python(self):
        for template in TEMPLATES:
            with self.subTest(template=template):
                result = self.run_template(template, 2, missing_hook=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("Module cleanup", result.stdout)
                self.assertNotIn("Fixture Python:", result.stdout)

    def test_verify_only_precedes_gpu_preflight_or_notebook_execution(self):
        for template in TEMPLATES:
            with self.subTest(template=template):
                result = self.run_template(template, 1)
                self.assertEqual(result.returncode, 0, result.stderr)
                python_lines = [line for line in result.stdout.splitlines()
                                if line.startswith("Fixture Python:")]
                self.assertIn("<-u><run_encoder_swaps.py><--verify-only>", python_lines[0])
                self.assertGreaterEqual(len(python_lines), 2)
                self.assertEqual(sum("--verify-only" in line for line in python_lines), 1)

    def test_verification_failure_stops_before_gpu_inspection_or_preprocessor(self):
        for template in TEMPLATES:
            with self.subTest(template=template):
                result = self.run_template(template, 0, overrides={"TEST_VERIFY_FAILURE": "1"})
                self.assertEqual(result.returncode, 86, result.stderr)
                python_lines = [line for line in result.stdout.splitlines()
                                if line.startswith("Fixture Python:")]
                self.assertEqual(len(python_lines), 1)
                self.assertIn("--verify-only", python_lines[0])

    def test_slurm_wrappers_refuse_unallocated_shell_before_startup(self):
        for template in TEMPLATES:
            with self.subTest(template=template):
                result = self.run_template(template, 0, overrides={"SLURM_JOB_ID": None})
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("Module cleanup", result.stdout)
                self.assertNotIn("Fixture Python:", result.stdout)

    def test_gpu_wrappers_require_expected_gpu_before_startup(self):
        for template in TEMPLATES[:2]:
            with self.subTest(template=template):
                result = self.run_template(template, 0, overrides={"ARCC_EXPECTED_GPU": None})
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("Fixture Python:", result.stdout)

    def test_custom_cache_output_and_kernel_remain_separate_quoted_arguments(self):
        for template in ("newton_swaps.sbatch.example", "newton_report.sbatch.example"):
            with self.subTest(template=template):
                result = self.run_template(template, 0, overrides={
                    "ARCC_CACHE_ROOT": "{root}/external cache",
                    "ARCC_OUTPUT_ROOT": "{root}/selected outputs",
                    "ARCC_KERNEL": "selected kernel",
                })
                self.assertEqual(result.returncode, 0, result.stderr)
                command = [line for line in result.stdout.splitlines() if "<--mode>" in line][-1]
                self.assertIn("<--kernel><selected kernel>", command)
                self.assertIn("/external cache>", command)
                self.assertIn("/selected outputs>", command)
                self.assertIn("<--purpose><newton>", command)
                self.assertIn("<--device><cuda>" if "swaps" in template else "<--device><cpu>", command)

    def test_selected_array_row_is_forwarded_without_an_implicit_full_grid(self):
        result = self.run_template("newton_swaps.sbatch.example", 0,
            overrides={"SLURM_ARRAY_TASK_ID": "17"},
            batch_rows="17\tmrna_stability\t2\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("<--benchmark><mrna_stability><--seed><2>", result.stdout)

    def test_missing_duplicate_or_invalid_array_index_does_not_start_notebook(self):
        cases = (("5", "2\tecoli_proteins\t0\n"),
                 ("2", "2\tecoli_proteins\t0\n2\tmrna_stability\t1\n"),
                 ("bad", "2\tecoli_proteins\t0\n"))
        for task, rows in cases:
            with self.subTest(task=task, rows=rows):
                result = self.run_template("newton_swaps.sbatch.example", 0,
                    overrides={"SLURM_ARRAY_TASK_ID": task}, batch_rows=rows)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--verify-only", result.stdout)
                self.assertNotIn("<--mode>", result.stdout)

    def test_missing_prepared_log_destination_stops_without_creating_it(self):
        for template in TEMPLATES[:2]:
            with self.subTest(template=template):
                result = self.run_template(template, 0, missing_logs=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("logs directory", result.stderr)
                self.assertNotIn("<--mode>", result.stdout)


class PreflightEnvironmentTests(unittest.TestCase):
    """The inline observation uses fake CUDA and package metadata, never real devices."""

    def inspect(self, *, available=True, gpu="Fixture GPU", expected="fixture"):
        template = (WORKSPACE / "newton_preflight.sbatch.example").read_text(encoding="utf-8")
        source = template.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: available, get_device_name=lambda _: gpu),
            version=SimpleNamespace(cuda="fixture-cuda"),
            backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 123)),
        )
        versions = Mock(return_value="installed-fixture-version")
        output = StringIO()
        with patch.dict(sys.modules, {"torch": fake_torch, "numpy": SimpleNamespace()}), \
                patch.object(sys, "argv", ["-", expected]), \
                patch("importlib.metadata.version", versions), redirect_stdout(output):
            exec(compile(source, "synthetic-preflight-environment", "exec"), {})
        return output.getvalue(), versions

    def test_cuda_unavailable_refuses_cpu_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "refusing CPU fallback"):
            self.inspect(available=False)

    def test_different_or_empty_expected_gpu_refuses(self):
        for expected in ("another GPU", "   "):
            with self.subTest(expected=expected), self.assertRaisesRegex(RuntimeError, "does not match"):
                self.inspect(expected=expected)

    def test_preflight_reports_actual_mocked_environment_without_installation(self):
        output, versions = self.inspect()
        payload = json.loads(output[:output.rfind("}") + 1])
        self.assertEqual(payload["gpu"], "Fixture GPU")
        self.assertEqual(payload["cuda"], "fixture-cuda")
        self.assertEqual(payload["cudnn"], 123)
        self.assertTrue(all(value == "installed-fixture-version"
                            for value in payload["packages"].values()))
        self.assertEqual(set(payload["packages"]), {call.args[0] for call in versions.call_args_list})
        self.assertIn("no notebook, weights, embeddings or training", output)


if __name__ == "__main__":
    unittest.main()
