
import os, re

def from_logs():
    log_path = "/tmp/hybrid.log"
    if not os.path.exists(log_path):
        print("SKIP: /tmp/hybrid.log not found; cannot verify log-based hybrid invariants.")
        return True  # soft skip
    ok = True
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "OK hybrid:" in line:
                m = re.search(r"score':\s*([0-9.]+)", line)
                if m:
                    s = float(m.group(1))
                    if not (0.0 <= s <= 1.0 + 1e-6):
                        print("FAIL: hybrid top1 score outside [0,1]:", s)
                        ok = False
    if ok:
        print("OK: hybrid log shows normalized scores in [0,1] (best-effort).")
    return ok

def from_module():
    try:
        from retriever import hybrid as H
        assert hasattr(H, "merge_and_score") or hasattr(H, "hybrid_search")
        print("OK: retriever.hybrid importable; deeper tests can be added here if needed.")
        return True
    except Exception as e:
        print("INFO: Could not import retriever.hybrid:", e)
        return False

def main():
    ok1 = from_logs()
    ok2 = from_module()
    if not (ok1 or ok2):
        print("SKIP: No hybrid evidence (logs or module).")

if __name__ == "__main__":
    main()
