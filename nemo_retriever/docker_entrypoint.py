#!/opt/retriever_runtime/bin/python3
"""
docker_entrypoint.py

Docker container entrypoint for nemo_retriever harness CI.

Workflow:
  1. Pull / clone the repo at REPO_URL (branch REPO_BRANCH) into REPO_DIR
  2. Reinstall nemo-retriever (editable install)
  3. Generate a nightly_config for the requested DATASETS subset
  4. Run `retriever harness nightly`, which posts results to Slack automatically

Required environment variables:
  SLACK_WEBHOOK_URL   Slack incoming-webhook URL (harness reads this directly)

Optional environment variables (shown with defaults):
  REPO_URL            https://github.com/NVIDIA/nv-ingest.git
  REPO_BRANCH         main
  REPO_DIR            /raid/nv-ingest
  REPO_BRANCH         main
  CLONE_DIR           (unset) – if set, clone the repo here and use this
                      directory as the source for all install and harness
                      actions, leaving REPO_DIR untouched.
                      Example: CLONE_DIR=/tmp/nv-ingest-work
  GIT_TOKEN           (unset) – set to a PAT for private forks, e.g. ghp_xxx
  DATASETS            jp20,bo20 – comma-separated subset of dataset keys
                      defined in harness/test_configs.yaml:
                      bo20 | bo767 | jp20 | earnings | bo10k | financebench
  PRESET              single_gpu – single_gpu | dgx_8gpu
  SKIP_SLACK          false – set to "true" to suppress the Slack post
  SLACK_TITLE         "nemo_retriever CI Harness"
  INSTALL_DEPS        false – set to "true" to also upgrade dependencies on
                      reinstall (slower; useful when pyproject.toml changed)
  HARNESS_ONLY        false – set to "true" to skip steps 1 & 2 (no git sync,
                      no reinstall) and run steps 3 & 4 against the current
                      codebase. REPO_DIR / CLONE_DIR still determines where
                      the harness is imported from.
  DATASET_DIR         (unset) – override the dataset directory for all runs;
                      forwarded to the harness as HARNESS_DATASET_DIR.
                      Example: DATASET_DIR=/raid/datasets/bo20
  QUERY_CSV           (unset) – override the query CSV path for all runs;
                      forwarded to the harness as HARNESS_QUERY_CSV.
                      Example: QUERY_CSV=/raid/datasets/bo20/queries.csv
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml


# ── helpers ───────────────────────────────────────────────────────────────────


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {msg}", flush=True)


def run(cmd: list[str], **kwargs) -> None:
    """Run a command, streaming output to the terminal; raise on failure."""
    subprocess.run(cmd, check=True, **kwargs)


def git_short_commit(work_root: str) -> str:
    result = subprocess.run(
        ["git", "-C", work_root, "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


# ── step 1: sync repo ─────────────────────────────────────────────────────────


def sync_repo(
    *,
    repo_url: str,
    repo_branch: str,
    repo_dir: str,
    clone_dir: str,
    git_token: str,
) -> str:
    """Clone or pull the repo; returns the work_root path used."""
    effective_url = repo_url
    if git_token:
        effective_url = repo_url.replace("https://", f"https://{git_token}@")

    if clone_dir:
        log(f"CLONE_DIR set – cloning fresh into {clone_dir}")
        shutil.rmtree(clone_dir, ignore_errors=True)
        run(["git", "clone", "--branch", repo_branch, "--depth", "1", effective_url, clone_dir])
        work_root = clone_dir
    elif Path(repo_dir, ".git").is_dir():
        log(f"Repo exists at REPO_DIR – fetching and resetting to origin/{repo_branch}")
        run(["git", "-C", repo_dir, "remote", "set-url", "origin", effective_url])
        run(["git", "-C", repo_dir, "fetch", "--depth=1", "origin", repo_branch])
        run(["git", "-C", repo_dir, "reset", "--hard", f"origin/{repo_branch}"])
        run(["git", "-C", repo_dir, "clean", "-fd"])
        work_root = repo_dir
    else:
        log(f"Cloning into REPO_DIR ({repo_dir})")
        run(["git", "clone", "--branch", repo_branch, "--depth", "1", effective_url, repo_dir])
        work_root = repo_dir

    # Scrub token from remote URL so it doesn't appear in git output
    if git_token:
        run(["git", "-C", work_root, "remote", "set-url", "origin", repo_url])

    commit = git_short_commit(work_root)
    log(f"Repo synced at {work_root} – commit: {commit}")
    return work_root


# ── step 2: reinstall packages ────────────────────────────────────────────────


def install_packages(*, nemo_retriever_dir: str, install_deps: bool) -> None:
    pip = [sys.executable, "-m", "pip", "install", "--quiet"]
    if not install_deps:
        pip.append("--no-deps")
    else:
        log("INSTALL_DEPS=true – will also upgrade transitive dependencies")

    log("Installing nemo-retriever (editable)...")
    run(pip + ["-e", nemo_retriever_dir])
    log("Package reinstall complete.")


# ── step 3: generate nightly config ──────────────────────────────────────────


def build_nightly_config(
    *,
    datasets: str,
    preset: str,
    slack_title: str,
    config_path: str,
) -> None:
    selected = [d.strip() for d in datasets.split(",") if d.strip()]
    if not selected:
        print("ERROR: DATASETS is empty – nothing to run.", file=sys.stderr)
        sys.exit(1)

    runs = [{"name": f"{d}_{preset}", "dataset": d} for d in selected]

    nightly_cfg = {
        "preset": preset,
        "runs": runs,
        "slack": {
            "enabled": True,
            "title": slack_title,
            "post_artifact_paths": True,
            "metric_keys": ["pages", "ingest_secs", "pages_per_sec_ingest", "recall_5"],
        },
    }

    with open(config_path, "w") as fh:
        yaml.dump(nightly_cfg, fh, default_flow_style=False, sort_keys=False)

    log(f"Nightly config written to: {config_path}")
    for entry in runs:
        log(f"  - {entry['name']}  (dataset={entry['dataset']}, preset={preset})")


# ── step 4: run harness ───────────────────────────────────────────────────────


def run_harness(
    *,
    nemo_retriever_dir: str,
    config_path: str,
    skip_slack: bool,
    dataset_dir: str,
    query_csv: str,
) -> int:
    # Make nemo_retriever source importable for editable installs before import
    src_path = str(Path(nemo_retriever_dir) / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    # Set env-var overrides that load_harness_config reads via _apply_env_overrides
    if dataset_dir:
        os.environ["HARNESS_DATASET_DIR"] = dataset_dir
        log(f"DATASET_DIR override: {dataset_dir}")
    if query_csv:
        os.environ["HARNESS_QUERY_CSV"] = query_csv
        log(f"QUERY_CSV override: {query_csv}")

    if skip_slack:
        log("Slack posting suppressed (SKIP_SLACK=true)")

    # Import harness internals directly — no subprocess needed for the
    # orchestration layer (batch_pipeline still runs as a subprocess via Ray).
    from nemo_retriever.harness.artifacts import write_session_summary
    from nemo_retriever.harness.config import load_nightly_config
    from nemo_retriever.harness.nightly import _maybe_post_to_slack
    from nemo_retriever.harness.run import execute_runs

    nightly_cfg = load_nightly_config(config_path)
    runs = nightly_cfg["runs"]
    slack_config = nightly_cfg["slack"]
    preset_override = nightly_cfg.get("preset")
    session_dir, run_results = execute_runs(
        runs=runs,
        config_file=None,
        session_prefix="nightly",
        preset_override=preset_override,
    )

    summary_path = write_session_summary(
        session_dir,
        run_results,
        session_type="nightly",
        config_path=str(Path(config_path).expanduser().resolve()),
    )

    log(f"Nightly session: {session_dir}")
    log(f"Session summary: {summary_path}")

    slack_failed = False
    try:
        _maybe_post_to_slack(
            report_path=summary_path,
            replay_paths=None,
            slack_config=slack_config,
            skip_slack=skip_slack,
        )
    except RuntimeError as exc:
        log(f"Slack post failed: {exc}")
        slack_failed = True

    failed = [r for r in run_results if not r["success"]]
    return 0 if not failed and not slack_failed else 1


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    repo_url = os.environ.get("REPO_URL", "https://github.com/NVIDIA/nv-ingest.git")
    repo_branch = os.environ.get("REPO_BRANCH", "main")
    repo_dir = os.environ.get("REPO_DIR", "/raid/nv-ingest")
    clone_dir = os.environ.get("CLONE_DIR", "")
    git_token = os.environ.get("GIT_TOKEN", "")

    datasets = os.environ.get("DATASETS", "jp20,bo20")
    preset = os.environ.get("PRESET", "single_gpu")
    skip_slack = os.environ.get("SKIP_SLACK", "false").lower() == "true"
    slack_title = os.environ.get("SLACK_TITLE", "nemo_retriever CI Harness")

    install_deps = os.environ.get("INSTALL_DEPS", "false").lower() == "true"
    harness_only = os.environ.get("HARNESS_ONLY", "false").lower() == "true"
    dataset_dir = os.environ.get("DATASET_DIR", "")
    query_csv = os.environ.get("QUERY_CSV", "")

    if harness_only:
        log("=== HARNESS_ONLY=true – skipping steps 1 & 2 ===")
        work_root = clone_dir or repo_dir
        nemo_retriever_dir = str(Path(work_root) / "nemo_retriever")
    else:
        # ── step 1 ────────────────────────────────────────────────────────
        log(f"=== Step 1: Sync repo ({repo_url}, branch: {repo_branch}) ===")
        work_root = sync_repo(
            repo_url=repo_url,
            repo_branch=repo_branch,
            repo_dir=repo_dir,
            clone_dir=clone_dir,
            git_token=git_token,
        )
        nemo_retriever_dir = str(Path(work_root) / "nemo_retriever")

        # ── step 2 ────────────────────────────────────────────────────────
        log("=== Step 2: Reinstall nemo-retriever packages ===")
        install_packages(nemo_retriever_dir=nemo_retriever_dir, install_deps=install_deps)

    # ── step 3 ────────────────────────────────────────────────────────────
    log(f"=== Step 3: Build nightly config (datasets: {datasets}, preset: {preset}) ===")
    fd, config_path = tempfile.mkstemp(suffix=".yaml", prefix="nightly_config_")
    os.close(fd)

    try:
        build_nightly_config(
            datasets=datasets,
            preset=preset,
            slack_title=slack_title,
            config_path=config_path,
        )

        # ── step 4 ────────────────────────────────────────────────────────
        log("=== Step 4: Running harness nightly ===")
        rc = run_harness(
            nemo_retriever_dir=nemo_retriever_dir,
            config_path=config_path,
            skip_slack=skip_slack,
            dataset_dir=dataset_dir,
            query_csv=query_csv,
        )
    finally:
        Path(config_path).unlink(missing_ok=True)

    log("=== Harness finished ===")
    sys.exit(rc)


if __name__ == "__main__":
    main()
