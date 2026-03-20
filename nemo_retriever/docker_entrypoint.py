#!/opt/retriever_runtime/bin/python3
"""
docker_entrypoint.py

Docker container entrypoint for nemo_retriever harness CI.

Workflow:
  1. Pull / clone the repo at REPO_URL (branch REPO_BRANCH) into REPO_DIR
  2. Reinstall nemo-retriever (editable install)
  3. Generate a nightly_config for the requested DATASETS subset
  4. Run `retriever harness nightly`, which posts results to Slack automatically

Required (via CLI flag or environment variable):
  SLACK_WEBHOOK_URL   Slack incoming-webhook URL (--slack-webhook-url takes precedence)

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
  LOOP                false – set to "true" to repeat the full
                      sync→install→harness cycle indefinitely.
  LOOP_INTERVAL       0 – seconds to sleep between loop iterations.
  DATASET_DIR         (unset) – override the dataset directory for all runs;
                      forwarded to the harness as HARNESS_DATASET_DIR.
                      Example: DATASET_DIR=/raid/datasets/bo20
  QUERY_CSV           (unset) – override the query CSV path for all runs;
                      forwarded to the harness as HARNESS_QUERY_CSV.
                      Example: QUERY_CSV=/raid/datasets/bo20/queries.csv
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
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
    slack_webhook_url: str,
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

    os.environ["SLACK_WEBHOOK_URL"] = slack_webhook_url or os.environ.get("SLACK_WEBHOOK_URL", "")
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="nemo_retriever CI harness entrypoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── repo / git ────────────────────────────────────────────────────────
    p.add_argument("--repo-url", default=None, metavar="URL", help="Git remote URL to clone/pull  [env: REPO_URL]")
    p.add_argument("--repo-branch", default=None, metavar="BRANCH", help="Branch to sync  [env: REPO_BRANCH]")
    p.add_argument("--repo-dir", default=None, metavar="PATH", help="Local path for the in-place pull  [env: REPO_DIR]")
    p.add_argument(
        "--clone-dir",
        default=None,
        metavar="PATH",
        help="Clone repo here fresh (leaves --repo-dir untouched)  [env: CLONE_DIR]",
    )
    p.add_argument("--git-token", default=None, metavar="TOKEN", help="PAT for private forks  [env: GIT_TOKEN]")

    # ── harness ───────────────────────────────────────────────────────────
    p.add_argument(
        "--datasets", default=None, metavar="LIST", help="Comma-separated dataset keys, e.g. jp20,bo20  [env: DATASETS]"
    )
    p.add_argument("--preset", default=None, metavar="PRESET", help="Preset name, e.g. single_gpu  [env: PRESET]")
    p.add_argument(
        "--slack-title", default=None, metavar="TITLE", help="Title shown in the Slack post  [env: SLACK_TITLE]"
    )
    p.add_argument(
        "--slack-webhook-url",
        default=None,
        metavar="URL",
        help="Slack webhook URL. Overrides SLACK_WEBHOOK_URL env var  [env: SLACK_WEBHOOK_URL]",
    )
    p.add_argument(
        "--dataset-dir",
        default=None,
        metavar="PATH",
        help="Override dataset directory (HARNESS_DATASET_DIR)  [env: DATASET_DIR]",
    )
    p.add_argument(
        "--query-csv",
        default=None,
        metavar="PATH",
        help="Override query CSV path (HARNESS_QUERY_CSV)  [env: QUERY_CSV]",
    )

    # ── flags ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--skip-slack", action="store_true", default=None, help="Suppress Slack posting  [env: SKIP_SLACK=true]"
    )
    p.add_argument(
        "--install-deps",
        action="store_true",
        default=None,
        help="Upgrade transitive deps on reinstall  [env: INSTALL_DEPS=true]",
    )
    p.add_argument(
        "--harness-only",
        action="store_true",
        default=None,
        help="Skip steps 1 & 2; run steps 3 & 4 against current codebase  [env: HARNESS_ONLY=true]",
    )
    p.add_argument(
        "--loop",
        action="store_true",
        default=None,
        help="Run indefinitely, repeating sync→install→harness  [env: LOOP=true]",
    )
    p.add_argument(
        "--loop-interval",
        default=None,
        type=int,
        metavar="SECS",
        help="Seconds to sleep between loop iterations  [env: LOOP_INTERVAL, default: 0]",
    )

    return p.parse_args()


def _resolve(arg_val, env_key: str, default: str) -> str:
    """Return first non-None value in priority order: CLI arg → env var → default."""
    if arg_val is not None:
        return str(arg_val)
    return os.environ.get(env_key) or default


def _resolve_bool(arg_val, env_key: str) -> bool:
    if arg_val:  # store_true sets True when flag is present
        return True
    return os.environ.get(env_key, "false").lower() == "true"


def main() -> None:
    args = _parse_args()

    repo_url = _resolve(args.repo_url, "REPO_URL", "https://github.com/NVIDIA/nv-ingest.git")
    repo_branch = _resolve(args.repo_branch, "REPO_BRANCH", "main")
    repo_dir = _resolve(args.repo_dir, "REPO_DIR", "/raid/nv-ingest")
    clone_dir = _resolve(args.clone_dir, "CLONE_DIR", "")
    git_token = _resolve(args.git_token, "GIT_TOKEN", "")

    datasets = _resolve(args.datasets, "DATASETS", "jp20,bo20")
    preset = _resolve(args.preset, "PRESET", "single_gpu")
    slack_title = _resolve(args.slack_title, "SLACK_TITLE", "nemo_retriever CI Harness")
    slack_webhook_url = _resolve(args.slack_webhook_url, "SLACK_WEBHOOK_URL", "")
    dataset_dir = _resolve(args.dataset_dir, "DATASET_DIR", "")
    query_csv = _resolve(args.query_csv, "QUERY_CSV", "")

    skip_slack = _resolve_bool(args.skip_slack, "SKIP_SLACK")
    install_deps = _resolve_bool(args.install_deps, "INSTALL_DEPS")
    harness_only = _resolve_bool(args.harness_only, "HARNESS_ONLY")
    loop = _resolve_bool(args.loop, "LOOP")
    loop_interval = int(_resolve(args.loop_interval, "LOOP_INTERVAL", "0"))

    # Graceful shutdown on SIGINT / SIGTERM so the loop exits cleanly.
    _shutdown = False

    def _handle_signal(signum, _frame):
        nonlocal _shutdown
        log(f"Caught signal {signal.Signals(signum).name} – finishing current iteration then exiting")
        _shutdown = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    iteration = 0

    while True:
        iteration += 1
        if loop:
            log(f"=== Loop iteration {iteration} ===")

        _run_iteration(
            harness_only=harness_only,
            repo_url=repo_url,
            repo_branch=repo_branch,
            repo_dir=repo_dir,
            clone_dir=clone_dir,
            git_token=git_token,
            install_deps=install_deps,
            datasets=datasets,
            preset=preset,
            slack_title=slack_title,
            skip_slack=skip_slack,
            slack_webhook_url=slack_webhook_url,
            dataset_dir=dataset_dir,
            query_csv=query_csv,
        )

        if not loop or _shutdown:
            break

        if loop_interval > 0:
            log(f"Sleeping {loop_interval}s before next iteration...")
            time.sleep(loop_interval)

    log("=== Harness finished ===")


def _run_iteration(
    *,
    harness_only: bool,
    repo_url: str,
    repo_branch: str,
    repo_dir: str,
    clone_dir: str,
    git_token: str,
    install_deps: bool,
    datasets: str,
    preset: str,
    slack_title: str,
    skip_slack: bool,
    slack_webhook_url: str,
    dataset_dir: str,
    query_csv: str,
) -> int:
    """Execute one full sync → install → config → harness cycle. Returns exit code."""

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
            slack_webhook_url=slack_webhook_url,
            dataset_dir=dataset_dir,
            query_csv=query_csv,
        )
    finally:
        Path(config_path).unlink(missing_ok=True)

    return rc


if __name__ == "__main__":
    main()
