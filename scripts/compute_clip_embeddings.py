"""
Compute per-business CLIP image embeddings (offline precompute).

Why this exists:
- TensorFlow training stays pure TF/TFRS.
- CLIP embedding extraction is done once offline and stored as tabular features.
- Candidate tower consumes these embeddings as "image_emb".

Requirements (optional):
  pip install torch open_clip_torch pillow tqdm

Inputs expected:
- Yelp photos downloaded separately (not in the core Yelp JSON).
- A mapping from photo_id -> local image file.
- Yelp photo metadata file: yelp_academic_dataset_photo.json

Output:
- CSV (or parquet) with columns: business_id, image_emb_0 ... image_emb_{d-1}

Aggregation:
- For each business, take mean of all photo embeddings.
"""

import argparse
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo_json", required=True, help="Path to yelp_academic_dataset_photo.json")
    ap.add_argument("--images_dir", required=True, help="Directory containing photo files named <photo_id>.jpg/png/...")
    ap.add_argument("--out_csv", default="outputs/business_image_embeddings.csv")
    ap.add_argument("--model", default="ViT-B-32", help="OpenCLIP model name")
    ap.add_argument("--pretrained", default="openai", help="OpenCLIP pretrained tag")
    ap.add_argument("--max_images_per_business", type=int, default=20)
    args = ap.parse_args()

    import torch
    import open_clip
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(device)
    model.eval()

    # Read photo metadata
    photo_path = Path(args.photo_json)
    rows = []
    with photo_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    # Group photo_ids per business
    biz_to_photos = {}
    for r in rows:
        bid = str(r["business_id"])
        pid = str(r["photo_id"])
        biz_to_photos.setdefault(bid, []).append(pid)

    images_dir = Path(args.images_dir)
    out = {}

    for bid, pids in tqdm(biz_to_photos.items(), total=len(biz_to_photos)):
        # cap per business
        pids = pids[: args.max_images_per_business]
        embs = []
        for pid in pids:
            # Try common extensions
            img_path = None
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                p = images_dir / f"{pid}{ext}"
                if p.exists():
                    img_path = p
                    break
            if img_path is None:
                continue
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            x = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                e = model.encode_image(x)
                e = e / e.norm(dim=-1, keepdim=True)
            embs.append(e.squeeze(0).cpu().numpy().astype(np.float32))

        if not embs:
            continue
        m = np.mean(np.stack(embs, axis=0), axis=0)
        m = m / (np.linalg.norm(m) + 1e-12)
        out[bid] = m.astype(np.float32)

    # Write CSV
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    dim = next(iter(out.values())).shape[0] if out else 512
    header = ["business_id"] + [f"image_emb_{i}" for i in range(dim)]

    import csv
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for bid, vec in out.items():
            w.writerow([bid] + vec.tolist())

    print(f"Wrote {len(out)} businesses to {args.out_csv} (dim={dim}).")

if __name__ == "__main__":
    main()
