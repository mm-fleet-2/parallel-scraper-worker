"""CLIP embeddings for photo-gate training (no model calls, read-only on photos).
Manifest shard = jsonl lines {"k": key, "u": blob URL without SAS}. For each image: download, measure mean luminance,
brighten when round(lum, 3) < 0.36 (same _brighten as llm_batch_build / media_mirror enhance=auto), embed with
MobileCLIP2-S0 (image tower, reparameterized). Output <out-base>/<shard>.npz: k (keys), emb (fp16 N x 512, unnormalised),
lum, plus err (keys that failed). Labels never travel: keys only.
usage: python tools/clip_embed.py --manifest-url URL --out-base URL --shard NAME [--workers 32]
env: MEDIA_BLOB_READ_SAS, MEDIA_BLOB_WRITE_SAS (query strings)"""
import argparse, io, math, os, re, sys, time, concurrent.futures as cf
import numpy as np, requests, torch, open_clip
from PIL import Image, ImageFilter, ImageOps, ImageStat

Image.MAX_IMAGE_PIXELS = None
AUTO_DARK = 0.36
ZS_LABELS = [  # (label, prompt) for zero-shot image typing
    ("storefront", "a photo of a small shop front with a signboard"),
    ("shelves", "a photo of shop shelves and a counter stocked with packaged products"),
    ("product", "a close-up photo of a product package or snack packet"),
    ("food", "a photo of cooked food or a dish on a plate"),
    ("people", "a photo of people posing or a selfie"),
    ("logo", "a logo, poster, banner or advertisement graphic"),
    ("street", "a street scene with a road, vehicles and buildings"),
    ("empty", "a photo of a ceiling, wall, floor, sky or ground with nothing in it"),
    ("document", "a screenshot, document, bill or menu with text"),
]


def sas(name):
    return os.environ[name].lstrip("?")


def mask(e):  # never print a SAS
    return re.sub(r"sig=[^&\s]+", "sig=***", str(e))


def brighten(im):
    if ImageStat.Stat(im.convert("L")).mean[0] / 255 > 0.45:
        return Image.blend(im, ImageOps.autocontrast(im, cutoff=0.5, preserve_tone=True), 0.25)
    im = ImageOps.autocontrast(im, cutoff=1, preserve_tone=True)
    mean = min(max(ImageStat.Stat(im.convert("L")).mean[0] / 255, 0.01), 0.99)
    gamma = min(1.0, max(0.45, math.log(0.47) / math.log(mean)))
    im = im.point([round(255 * (i / 255) ** gamma) for i in range(256)] * 3)
    return im.filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=3))


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--manifest-url", required=True); a.add_argument("--out-base", required=True)
    a.add_argument("--shard", required=True); a.add_argument("--workers", type=int, default=32)
    a = a.parse_args()
    rs, ws = sas("MEDIA_BLOB_READ_SAS"), sas("MEDIA_BLOB_WRITE_SAS")
    s = requests.Session(); s.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=a.workers))
    try:
        r = s.get(f"{a.manifest_url}?{rs}", timeout=120); r.raise_for_status()
    except Exception as e:
        sys.exit(f"manifest read failed: {mask(e)}")
    import json
    rows = [json.loads(l) for l in r.text.splitlines() if l.strip()]
    m, _, pre = open_clip.create_model_and_transforms("hf-hub:timm/MobileCLIP2-S0-OpenCLIP")
    from timm.utils.model import reparameterize_model
    m = reparameterize_model(m.eval()); torch.set_num_threads(os.cpu_count())

    def get(row):
        for t in range(3):
            try:
                b = s.get(f"{row['u'].split('?')[0]}?{rs}", timeout=60); b.raise_for_status()
                im = Image.open(io.BytesIO(b.content)).convert("RGB")
                L = round(ImageStat.Stat(im.convert("L")).mean[0] / 255, 3)
                if L < AUTO_DARK:
                    im = brighten(im)
                return row["k"], L, pre(im)
            except Exception as e:
                err = mask(e); time.sleep(1 + t)
        return row["k"], None, err

    # Zero-shot labels (added 12 Nov 2026, Haldiram vision pilot): the same pass also scores every image
    # against fixed text prompts; saved as `labels` + `probs` in the npz. Old readers ignore the new arrays.
    tok = open_clip.get_tokenizer("hf-hub:timm/MobileCLIP2-S0-OpenCLIP")
    with torch.no_grad():
        T = m.encode_text(tok([p for _, p in ZS_LABELS])).float()
        T = T / T.norm(dim=-1, keepdim=True)

    t0 = time.time(); K, LUM, E, ERR, buf = [], [], [], [], []
    with cf.ThreadPoolExecutor(a.workers) as ex, torch.no_grad():
        for k, L, x in ex.map(get, rows):
            if L is None:
                ERR.append(k); continue
            K.append(k); LUM.append(L); buf.append(x)
            if len(buf) == 64:
                E.append(m.encode_image(torch.stack(buf)).numpy().astype(np.float16)); buf = []
        if buf:
            E.append(m.encode_image(torch.stack(buf)).numpy().astype(np.float16))
    if not K:  # never upload a result when every input failed (a rerun would skip the shard)
        sys.exit(f"{a.shard}: every image failed ({len(ERR)}); nothing uploaded")
    out = io.BytesIO()
    emb = np.concatenate(E) if E else np.zeros((0, 512), np.float16)
    I = torch.from_numpy(emb.astype(np.float32)); I = I / I.norm(dim=-1, keepdim=True)
    probs = (100.0 * I @ T.T).softmax(dim=-1).numpy().astype(np.float16)
    np.savez(out, k=np.array(K), emb=emb, lum=np.array(LUM, np.float32), err=np.array(ERR),
             labels=np.array([n for n, _ in ZS_LABELS]), probs=probs)
    try:
        p = s.put(f"{a.out_base.rstrip('/')}/{a.shard}.npz?{ws}", data=out.getvalue(),
                  headers={"x-ms-blob-type": "BlockBlob"}, timeout=300); p.raise_for_status()
    except Exception as e:
        sys.exit(f"upload failed: {mask(e)}")
    print(f"{a.shard}: {len(K)} ok, {len(ERR)} failed, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
