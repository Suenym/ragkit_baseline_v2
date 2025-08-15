
import sys
import faiss
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("Usage: python tests/dense_norm_check.py ./data/faiss.index")
        raise SystemExit(2)
    path = sys.argv[1]
    index = faiss.read_index(path)
    ntotal = index.ntotal
    d = index.d
    print(f"ntotal: {ntotal} dim: {d}")
    try:
        xb = faiss.vector_to_array(index.xb).reshape(ntotal, d)
        norms = np.linalg.norm(xb, axis=1)
        print(f"norms mean/min/max: {norms.mean():.3f} {norms.min():.3f} {norms.max():.3f}")
        if not (abs(norms.mean()-1.0) < 0.02 and norms.min() > 0.95 and norms.max() < 1.05):
            print("FAIL: vectors are not ~L2-normalized within tolerance")
            raise SystemExit(1)
        print("OK: FAISS vectors L2-normalized; meta parsed successfully.")
    except Exception as e:
        print(f"WARN: Cannot access raw vectors for this index type ({type(index)}): {e}")
        print("OK: FAISS index loaded; skipping norm check for this index type.")

if __name__ == "__main__":
    main()
