#!/usr/bin/env python3
"""sv-3: drive the REAL `gateway serve` daemon (unmodified
demo/agent-trace/gateway/src/main.rs, built --release) through the
100-task corpus in tasks.json, over its real Unix-socket wire protocol.
Records ALLOW/DENY/outcome and latency for every task; for the
verify_then_execute_race pairs, submits both requests from separate
threads with a tight join to actually race the PENDING window.
"""
import json
import os
import socket
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SOCK = os.environ.get("SV3_SOCK", "/tmp/sv3-gateway.sock")


def send_request(lines, timeout=5.0):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(SOCK)
    s.sendall(("\n".join(lines) + "\n\n").encode())
    s.shutdown(socket.SHUT_WR)
    buf = b""
    try:
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
    except socket.timeout:
        pass
    s.close()
    return buf.decode(errors="replace").strip()


def decide_and_poll(task, poll_interval=0.02, poll_timeout=5.0):
    t0 = time.perf_counter()
    req = [
        f"RECEIPT {task['receipt']}",
        f"ACTION {task['action_hex']}",
        f"SESSION {task['session']}",
        f"COUNTER {task['counter']}",
        f"TOOL {task['tool']}",
    ]
    first = send_request(req)
    if first.startswith("PENDING ticket="):
        ticket = first.split("ticket=", 1)[1].strip()
        deadline = time.perf_counter() + poll_timeout
        final = first
        while time.perf_counter() < deadline:
            time.sleep(poll_interval)
            r = send_request([f"POLL ticket={ticket}"])
            if r.startswith("ALLOW") or r.startswith("DENY"):
                final = r
                break
            final = r
        result_line = final
    else:
        result_line = first
    t1 = time.perf_counter()
    return result_line, (t1 - t0) * 1000.0  # ms


def run_race_pair(tasks_by_pair, pair_id, results, lock):
    pair_tasks = tasks_by_pair[pair_id]
    outcomes = [None, None]
    latencies = [None, None]

    def worker(idx, task):
        line, ms = decide_and_poll(task)
        outcomes[idx] = line
        latencies[idx] = ms

    threads = [threading.Thread(target=worker, args=(i, t)) for i, t in enumerate(pair_tasks)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    with lock:
        for i, t in enumerate(pair_tasks):
            results.append({**t, "result_line": outcomes[i], "latency_ms": latencies[i]})


def main():
    with open(os.path.join(HERE, "tasks.json")) as f:
        tasks = json.load(f)

    results = []
    lock = threading.Lock()

    race_tasks = [t for t in tasks if t["category"] == "verify_then_execute_race"]
    by_pair = {}
    for t in race_tasks:
        by_pair.setdefault(t["pair_id"], []).append(t)

    non_race = [t for t in tasks if t["category"] != "verify_then_execute_race"]

    # Replay pairs must run in order (first then replay), sequentially,
    # same for everything else -- natural order in tasks.json already
    # places the legitimate submission immediately before its replay.
    for t in non_race:
        line, ms = decide_and_poll(t)
        results.append({**t, "result_line": line, "latency_ms": ms})

    for pair_id in by_pair:
        run_race_pair(by_pair, pair_id, results, lock)

    results.sort(key=lambda r: r["id"])
    with open(os.path.join(HERE, "out", "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"ran {len(results)} tasks, wrote out/results.json")


if __name__ == "__main__":
    main()
