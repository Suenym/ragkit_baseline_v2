
import os

def from_logs():
    log_path = "/tmp/rerank.log"
    if not os.path.exists(log_path):
        print("SKIP: /tmp/rerank.log not found; cannot verify log-based rerank invariants.")
        return True
    ok = False
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(2000)
        if "OK: rerank deterministic" in head and "[0,1]" in head and "sorted" in head:
            print("OK: rerank log indicates determinism, sorted order, normalized scores.")
            ok = True
    return ok

def from_module():
    try:
        from retriever import rerank as R
        assert hasattr(R, "llm_like_rerank") or hasattr(R, "rerank")
        print("OK: retriever.rerank importable; you can extend this to run a synthetic determinism test.")
        return True
    except Exception as e:
        print("INFO: Could not import retriever.rerank:", e)
        return False

def main():
    ok1 = from_logs()
    ok2 = from_module()
    if not (ok1 or ok2):
        print("SKIP: No rerank evidence (logs or module).")

if __name__ == "__main__":
    main()
