
import os, json, re

def main():
    log_path = "/tmp/cache.log"
    if not os.path.exists(log_path):
        print("SKIP: /tmp/cache.log not found. Run your pipeline twice to generate it.")
        return
    runs = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"\{.*\}", line)
            if m:
                try:
                    runs.append(json.loads(m.group(0).replace("'", '"')))
                except Exception:
                    pass
    if len(runs) < 2:
        print("SKIP: Need at least two runs in /tmp/cache.log to compare caching behavior.")
        return
    r1, r2 = runs[0], runs[1]
    r1_ret = r1.get("cache_retrieval_hits", 0)
    r2_ret = r2.get("cache_retrieval_hits", 0)
    r1_rr  = r1.get("cache_rerank_hits", 0)
    r2_rr  = r2.get("cache_rerank_hits", 0)

    print(f"[run 1] retrieval_hits={r1_ret} rerank_hits={r1_rr}")
    print(f"[run 2] retrieval_hits={r2_ret} rerank_hits={r2_rr}")
    if r2_ret >= r1_ret and r2_rr >= r1_rr:
        print("OK: Cache improved or stayed the same on second run.")
    else:
        print("WARN: Cache hits did not improve on second run. Investigate cache keys and storage.")

if __name__ == "__main__":
    main()
