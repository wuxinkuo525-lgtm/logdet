#!/usr/bin/env python
"""Run one cluster command with progress, GPU telemetry and explicit terminal status."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from logodet.dino_run import write_json_atomic


def snapshot(progress_file, started, stale_seconds, now=None):
    now = time.time() if now is None else now
    try:
        item = json.loads(progress_file.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        item = {'time': started, 'phase': 'starting'}
    age = max(0, now - item['time'])
    return dict(progress=item, seconds_without_progress=round(age), stale=age >= stale_seconds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--status', required=True)
    ap.add_argument('--interval', type=float, default=60)
    ap.add_argument('--stale-seconds', type=float, default=900)
    ap.add_argument('command', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if args.interval <= 0 or args.stale_seconds <= 0:
        ap.error('interval and stale-seconds must be positive')
    status = Path(args.status)
    status.parent.mkdir(parents=True, exist_ok=True)
    progress_file = status.with_suffix('.progress.json')
    started = time.time()
    write_json_atomic(progress_file, {'time': started, 'phase': 'starting'})
    env = {**os.environ, 'LOGDET_PROGRESS_FILE': str(progress_file.resolve()), 'PYTHONUNBUFFERED': '1'}
    cmd = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not cmd:
        ap.error('a command is required')
    try:
        child = subprocess.Popen(cmd, env=env, start_new_session=os.name != 'nt')
    except OSError as exc:
        write_json_atomic(status, dict(time=time.time(), state='FAILED', command=cmd, error=str(exc)))
        raise
    interrupted = []

    def stop(signum, frame):
        interrupted.append(signum)
        if child.poll() is None:
            if os.name == 'nt':
                child.terminate()
            else:
                os.killpg(child.pid, signal.SIGTERM)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    next_log = 0
    while True:
        rc = child.poll()
        now = time.time()
        if now >= next_log or rc is not None:
            state = snapshot(progress_file, started, args.stale_seconds, now)
            try:
                gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                                      '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
                telemetry = gpu.stdout.strip() if gpu.returncode == 0 else gpu.stderr.strip()
            except (OSError, subprocess.TimeoutExpired) as exc:
                telemetry = f'unavailable: {exc}'
            label = ('INTERRUPTED' if interrupted else ('SUCCEEDED' if rc == 0 else 'FAILED')) if rc is not None else ('STALE_WARNING' if state['stale'] else 'RUNNING')
            record = dict(time=now, state=label, exit_code=rc, command=cmd,
                          job_id=os.environ.get('SLURM_JOB_ID'), gpu=telemetry, **state)
            write_json_atomic(status, record)
            print('[HEALTH] ' + json.dumps(record, ensure_ascii=False), file=sys.stderr, flush=True)
            next_log = now + args.interval
        if rc is not None:
            return 128 + interrupted[-1] if interrupted else (rc if rc >= 0 else 128 - rc)
        if interrupted and now - started > 0:
            # Bound shutdown even if a loader worker ignores TERM.
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == 'nt': child.kill()
                else: os.killpg(child.pid, signal.SIGKILL)
        time.sleep(min(1, args.interval))


if __name__ == '__main__':
    raise SystemExit(main())
