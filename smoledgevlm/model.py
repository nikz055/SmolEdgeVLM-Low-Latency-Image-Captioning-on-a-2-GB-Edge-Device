"""SmolEdgeVLM: frozen SigLIP-B/16@512 + token merging with restoration + pooling connector + frozen SmolLM2-135M.

    from smoledgevlm import SmolEdgeVLM
    model = SmolEdgeVLM('weights')
    print(model.caption(Image.open('photo.jpg'))['caption'])

mode='smoledgevlm' runs the proposed encoder (merge after layer 3, R, layers 4-12 on 256 tokens with the log n
bias, unmerge). mode='full' runs the same pipeline with the merge switched off, which is the baseline.
"""
import os
import time

import numpy as np
import torch
from PIL import Image

from . import config as C
from .merge import TokenMerger, size_bias, unmerge
from .modules import PoolConnector, Restore, load_state


def vision_core(model):
    """The module holding embeddings, encoder and post_layernorm (its location differs across transformers versions)."""
    for cand in (getattr(model, 'vision_model', None), model):
        if cand is not None and all(hasattr(cand, a) for a in ('embeddings', 'encoder', 'post_layernorm')):
            return cand
    raise AttributeError('SigLIP embeddings/encoder not found in this transformers version')


def run_layer(layer, h, bias=None):
    out = layer(h, attention_mask=bias)
    return out[0] if isinstance(out, (tuple, list)) else out


def siglip_pixels(img, size=512):
    """SigLIP preprocessing: bicubic resize to size x size, scale to [0, 1], normalise with mean 0.5 and std 0.5."""
    x = np.asarray(img.convert('RGB').resize((size, size), Image.BICUBIC), dtype=np.float32) / 255.0
    return torch.from_numpy((x - 0.5) / 0.5).permute(2, 0, 1)[None]


class SmolEdgeVLM:
    WEIGHTS = {'restore': 'restore_R.pt', 'smoledgevlm': 'connector_smoledgevlm.pt', 'full': 'connector_full_encoder.pt'}

    def __init__(self, weights_dir='weights', device=None, dtype=torch.float32):
        from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.vision = SiglipVisionModel.from_pretrained(C.VISION_ID).to(self.device, dtype).eval()
        self.lm = AutoModelForCausalLM.from_pretrained(C.LM_ID, torch_dtype=dtype).to(self.device).eval()
        self.tok = AutoTokenizer.from_pretrained(C.LM_ID)
        for m in (self.vision, self.lm):
            for p in m.parameters():
                p.requires_grad_(False)
        self.vm = vision_core(self.vision)
        self.layers = self.vm.encoder.layers
        cfg = self.vision.config
        self.image_size = cfg.image_size
        self.grid = cfg.image_size // cfg.patch_size
        self.d_v, self.d_t = cfg.hidden_size, self.lm.config.hidden_size
        self.merger = TokenMerger(self.grid, C.NUM_TOKENS, C.RADIUS, self.device)
        self.embed = self.lm.get_input_embeddings()
        ids = self.tok(C.PROMPT, return_tensors='pt').input_ids[0].to(self.device)
        self.prompt = self.embed(ids)[None]

        self.R = Restore(self.d_v, C.RESTORE_RANK).to(self.device, dtype).eval()
        self.connector = {}
        if weights_dir:
            self.R.load_state_dict(load_state(os.path.join(weights_dir, self.WEIGHTS['restore']), self.device))
            for mode in ('smoledgevlm', 'full'):
                path = os.path.join(weights_dir, self.WEIGHTS[mode])
                if os.path.exists(path):
                    c = PoolConnector(self.d_v, self.d_t, C.CONNECTOR_GRID).to(self.device, dtype)
                    c.load_state_dict(load_state(path, self.device))
                    self.connector[mode] = c.eval()

    # ------------------------------------------------------------------ encoder
    def pixels(self, img):
        return siglip_pixels(img, self.image_size).to(self.device, self.dtype)

    def layers_to_cut(self, px):
        h = self.vm.embeddings(px)
        for layer in self.layers[:C.CUT_LAYER]:
            h = run_layer(layer, h)
        return h

    def full_from_cut(self, h):
        for layer in self.layers[C.CUT_LAYER:]:
            h = run_layer(layer, h)
        return self.vm.post_layernorm(h)

    def merged_from_cut(self, h, restore=None):
        """Merge -> R -> layers 4-12 with the log n bias -> post-norm -> unmerge. Returns [1, 1024, 768]."""
        R = self.R if restore is None else restore
        with torch.no_grad():
            m, sizes, assign = self.merger.merge(h)
        if R is not False:
            m = R(m)
        bias = size_bias(sizes).expand(1, 1, m.shape[1], m.shape[1]).to(m.dtype)
        for layer in self.layers[C.CUT_LAYER:]:
            m = run_layer(layer, m, bias)
        return unmerge(self.vm.post_layernorm(m), assign)

    @torch.no_grad()
    def encode(self, px, mode='smoledgevlm'):
        """Patch features on the 32 x 32 grid, [1, 1024, 768]. mode: 'smoledgevlm', 'merge_no_R' or 'full'."""
        if mode == 'full':
            return self.vision(pixel_values=px).last_hidden_state
        h = self.layers_to_cut(px)
        return self.merged_from_cut(h, restore=False if mode == 'merge_no_R' else None)

    # ------------------------------------------------------------------ captioning
    def _sync(self):
        if self.device == 'cuda':
            torch.cuda.synchronize()

    @torch.no_grad()
    def caption(self, img, mode='smoledgevlm', max_new=C.MAX_NEW_TOKENS):
        """Greedy caption with a KV cache. Returns the caption, time to first token and total time in ms."""
        conn = self.connector['full' if mode == 'full' else 'smoledgevlm']
        px = self.pixels(img)
        self._sync()
        t0 = time.perf_counter()
        z = conn(self.encode(px, mode))
        out = self.lm(inputs_embeds=torch.cat([z, self.prompt], 1), use_cache=True)
        nxt = out.logits[:, -1].argmax(-1)
        self._sync()
        ttft = (time.perf_counter() - t0) * 1e3
        past, toks = out.past_key_values, []
        for _ in range(max_new):
            if int(nxt) == self.tok.eos_token_id:
                break
            toks.append(int(nxt))
            out = self.lm(inputs_embeds=self.embed(nxt)[:, None], past_key_values=past, use_cache=True)
            past, nxt = out.past_key_values, out.logits[:, -1].argmax(-1)
        self._sync()
        total = (time.perf_counter() - t0) * 1e3
        return {'caption': self.tok.decode(toks, skip_special_tokens=True).strip(),
                'ttft_ms': ttft, 'total_ms': total, 'tokens': len(toks)}

    @torch.no_grad()
    def merge_map(self, img, cell=16):
        """Picture of the merge: one colour per merged token, white dots on the anchors."""
        _, _, assign = self.merger.merge(self.layers_to_cut(self.pixels(img)))
        pal = np.random.default_rng(0).random((C.NUM_TOKENS, 3)) * 0.8 + 0.1
        over = np.kron(pal[assign.cpu().numpy()].reshape(self.grid, self.grid, 3), np.ones((cell, cell, 1)))
        base = np.asarray(img.convert('RGB').resize((cell * self.grid,) * 2)) / 255.0
        out = 0.45 * base + 0.55 * over
        for idx in self.merger.anchors.cpu().numpy():
            y, x = (idx // self.grid) * cell + cell // 2, (idx % self.grid) * cell + cell // 2
            out[y - 1:y + 2, x - 1:x + 2] = 1.0
        return Image.fromarray((out * 255).astype(np.uint8))
