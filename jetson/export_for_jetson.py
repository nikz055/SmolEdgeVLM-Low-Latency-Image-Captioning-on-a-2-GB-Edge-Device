"""Run once on a PC (or Kaggle) with transformers installed: pack everything the Jetson needs into two files.

    smoledgevlm_jetson_weights.npz   SigLIP vision encoder, SmolLM2-135M, both connectors and R, in float16
    smoledgevlm_jetson_meta.json     model sizes, prompt token ids, the tokenizer's byte table and
                                     (optionally) reference captions to check the Jetson output against

    python jetson/export_for_jetson.py --weights-dir weights --out jetson_package
    python jetson/export_for_jetson.py --weights-dir weights --out jetson_package --ref-images images --ref-count 10
"""
import argparse, json, os, sys
import numpy as np
import torch

VIS_ID, LM_ID = 'google/siglip-base-patch16-512', 'HuggingFaceTB/SmolLM2-135M'
PROMPT = 'Describe this image in one sentence.'
K_CUT, M_TOKENS, RADIUS = 3, 256, 3


def torch_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def weight(folder, name):
    p = os.path.join(folder, name)
    if not os.path.exists(p):
        raise FileNotFoundError('%s not found in %s' % (name, folder))
    sd = torch_load(p)
    return sd.get('state_dict', sd)


def vision_core(model):
    """transformers 4.x keeps the encoder under .vision_model; 5.x puts it on the model itself."""
    for cand in (getattr(model, 'vision_model', None), model):
        if cand is not None and all(hasattr(cand, a) for a in ('embeddings', 'encoder', 'post_layernorm')):
            return cand
    raise AttributeError('could not find the SigLIP embeddings/encoder in this transformers version')


EXPORT_DTYPE = np.float16          # float16 halves the size; the Jetson can still run it in float32 if needed


def arr(t):
    return t.detach().float().cpu().numpy().astype(EXPORT_DTYPE)


def collect_vision(vm, out):
    e = vm.embeddings
    out['v.patch.w'], out['v.patch.b'] = arr(e.patch_embedding.weight), arr(e.patch_embedding.bias)
    out['v.pos'] = arr(e.position_embedding.weight)
    for i, L in enumerate(vm.encoder.layers):
        p, a = 'v.L%d.' % i, L.self_attn
        out[p + 'ln1.w'], out[p + 'ln1.b'] = arr(L.layer_norm1.weight), arr(L.layer_norm1.bias)
        for nm, lin in (('q', a.q_proj), ('k', a.k_proj), ('v', a.v_proj), ('o', a.out_proj)):
            out[p + nm + '.w'], out[p + nm + '.b'] = arr(lin.weight), arr(lin.bias)
        out[p + 'ln2.w'], out[p + 'ln2.b'] = arr(L.layer_norm2.weight), arr(L.layer_norm2.bias)
        out[p + 'fc1.w'], out[p + 'fc1.b'] = arr(L.mlp.fc1.weight), arr(L.mlp.fc1.bias)
        out[p + 'fc2.w'], out[p + 'fc2.b'] = arr(L.mlp.fc2.weight), arr(L.mlp.fc2.bias)
    out['v.post.w'], out['v.post.b'] = arr(vm.post_layernorm.weight), arr(vm.post_layernorm.bias)
    return len(vm.encoder.layers)


def collect_lm(lm, out):
    base = lm.model
    out['lm.embed'] = arr(base.embed_tokens.weight)
    for i, L in enumerate(base.layers):
        p, a, m = 'lm.L%d.' % i, L.self_attn, L.mlp
        for nm, lin in (('q', a.q_proj), ('k', a.k_proj), ('v', a.v_proj), ('o', a.o_proj),
                        ('gate', m.gate_proj), ('up', m.up_proj), ('down', m.down_proj)):
            if getattr(lin, 'bias', None) is not None:
                raise ValueError('unexpected bias in %s%s: this script assumes the SmolLM2 (Llama) layout' % (p, nm))
            out[p + nm] = arr(lin.weight)
        out[p + 'ln_in'] = arr(L.input_layernorm.weight)
        out[p + 'ln_post'] = arr(L.post_attention_layernorm.weight)
    out['lm.norm'] = arr(base.norm.weight)
    tied = lm.lm_head.weight.data_ptr() == base.embed_tokens.weight.data_ptr()
    if not tied:
        out['lm.head'] = arr(lm.lm_head.weight)
    return len(base.layers), tied


def collect_connector(sd, prefix, out):
    """PoolConnector: proj.0 = Linear(768, 576), proj.2 = Linear(576, 576)."""
    out[prefix + 'fc1.w'], out[prefix + 'fc1.b'] = arr(sd['proj.0.weight']), arr(sd['proj.0.bias'])
    out[prefix + 'fc2.w'], out[prefix + 'fc2.b'] = arr(sd['proj.2.weight']), arr(sd['proj.2.bias'])


def collect_restore(sd, out):
    for k in ('norm.weight', 'norm.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'):
        out['r.' + k.replace('weight', 'w').replace('bias', 'b')] = arr(sd[k])
    if sd['fc1.weight'].shape[1] != sd['norm.weight'].shape[0]:
        raise ValueError('R must take the merged token only (fc1 input width = token width)')


def bytes_to_unicode():
    """GPT-2 byte-level BPE table (byte value -> printable character)."""
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('\xa1'), ord('\xac') + 1)) + list(range(ord('\xae'), ord('\xff') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def rope_theta(lc):
    """transformers 4.x: config.rope_theta; 5.x: config.rope_parameters['rope_theta']. Only plain RoPE is supported."""
    rp = getattr(lc, 'rope_parameters', None) or {}
    rs = getattr(lc, 'rope_scaling', None) or {}
    theta = getattr(lc, 'rope_theta', None) or rp.get('rope_theta')
    if theta is None:
        raise ValueError('cannot find rope_theta in the SmolLM2 config')
    rtype = rs.get('rope_type') or rs.get('type') or rp.get('rope_type') or 'default'
    if rtype != 'default':
        raise ValueError('rope type %s is not supported by the Jetson script' % rtype)
    return float(theta)


def resize_hw(proc, default):
    size = getattr(proc, 'size', None)
    for get in (lambda s: (s['height'], s['width']), lambda s: (s.height, s.width)):
        try:
            h, w = get(size)
            if h and w:
                return [int(h), int(w)]
        except Exception:
            pass
    return [default, default]


def clean_up_spaces(s):
    """The same replacements as transformers' clean_up_tokenization (used only if the tokenizer has it on)."""
    for a, b in ((' .', '.'), (' ?', '?'), (' !', '!'), (' ,', ','), (" ' ", "'"), (" n't", "n't"),
                 (" 'm", "'m"), (" 's", "'s"), (" 've", "'ve"), (" 're", "'re")):
        s = s.replace(a, b)
    return s


def tokenizer_tables(tok, vocab_size):
    """id -> raw bytes (hex), so the Jetson can turn generated ids into text without the tokenizers library."""
    clean = bool(getattr(tok, 'clean_up_tokenization_spaces', False))
    dec = {v: k for k, v in bytes_to_unicode().items()}
    special = set(int(i) for i in tok.all_special_ids)
    hexes, extra_special = [], []
    for i in range(vocab_size):
        t = tok.convert_ids_to_tokens(i)
        if t is None or i in special or any(ch not in dec for ch in t):
            hexes.append('')
            if i not in special:
                extra_special.append(i)
            continue
        hexes.append(bytes(dec[ch] for ch in t).hex())
    def our_decode(ids):
        s = b''.join(bytes.fromhex(hexes[i]) for i in ids).decode('utf-8', errors='replace')
        return (clean_up_spaces(s) if clean else s).strip()
    tests = ['A dog runs on the grass.', 'Two people ride bicycles near a lake, café, 3 kids!', PROMPT,
             "a dog , a cat . it's here !"]
    for s in tests:
        ids = tok(s, add_special_tokens=False).input_ids
        a, b = our_decode(ids), tok.decode(ids, skip_special_tokens=True).strip()
        if a != b:
            raise RuntimeError('byte decoder mismatch: %r vs %r' % (a, b))
    return hexes, sorted(special | set(extra_special)), clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights-dir', default='weights')
    ap.add_argument('--out', default='jetson_package')
    ap.add_argument('--ref-images', default='')
    ap.add_argument('--ref-count', type=int, default=10)
    ap.add_argument('--ref-device', default='cpu', help='device for the float32 reference captions (cpu or cuda)')
    a = ap.parse_args()
    from transformers import SiglipVisionModel, AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor
    import transformers
    os.makedirs(a.out, exist_ok=True)

    print('loading %s and %s (transformers %s) ...' % (VIS_ID, LM_ID, transformers.__version__))
    vision = SiglipVisionModel.from_pretrained(VIS_ID).float().eval()
    try:
        lm = AutoModelForCausalLM.from_pretrained(LM_ID, dtype=torch.float32).eval()
    except TypeError:
        lm = AutoModelForCausalLM.from_pretrained(LM_ID, torch_dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(LM_ID)
    proc = AutoImageProcessor.from_pretrained(VIS_ID)

    out = {}
    vm = vision_core(vision)
    n_vis = collect_vision(vm, out)
    n_lm, tied = collect_lm(lm, out)
    collect_connector(weight(a.weights_dir, 'connector_full_encoder.pt'), 'cf.', out)
    collect_connector(weight(a.weights_dir, 'connector_smoledgevlm.pt'), 'cm.', out)
    collect_restore(weight(a.weights_dir, 'restore_R.pt'), out)

    vc, lc = vision.config, lm.config
    hexes, special, clean = tokenizer_tables(tok, lc.vocab_size)
    meta = {
        'format': 1, 'prompt': PROMPT,
        'prompt_ids': [int(i) for i in tok(PROMPT, return_tensors='pt').input_ids[0].tolist()],
        'eos_id': int(tok.eos_token_id), 'special_ids': special, 'id_bytes_hex': hexes, 'clean_up_spaces': clean,
        'vision': {'hidden': vc.hidden_size, 'heads': vc.num_attention_heads, 'layers': n_vis,
                   'ln_eps': vc.layer_norm_eps, 'image_size': vc.image_size, 'patch': vc.patch_size,
                   'act': vc.hidden_act, 'mean': list(getattr(proc, 'image_mean', [0.5, 0.5, 0.5])),
                   'std': list(getattr(proc, 'image_std', [0.5, 0.5, 0.5])),
                   'resize': resize_hw(proc, vc.image_size)},
        'lm': {'hidden': lc.hidden_size, 'layers': n_lm, 'heads': lc.num_attention_heads,
               'kv_heads': lc.num_key_value_heads, 'rms_eps': lc.rms_norm_eps,
               'rope_theta': rope_theta(lc), 'vocab': lc.vocab_size, 'tied': tied},
        'merge': {'k': K_CUT, 'M': M_TOKENS, 'radius': RADIUS},
        'exported_with': {'transformers': transformers.__version__, 'torch': torch.__version__},
    }
    if vc.hidden_act not in ('gelu_pytorch_tanh', 'gelu_new'):
        raise ValueError('unexpected SigLIP activation %s' % vc.hidden_act)

    wpath = os.path.join(a.out, 'smoledgevlm_jetson_weights.npz')
    np.savez(wpath, **out)
    print('wrote %s (%.0f MB, %d tensors)' % (wpath, os.path.getsize(wpath) / 2 ** 20, len(out)))

    if a.ref_images:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
        from smoledgevlm.model import SmolEdgeVLM
        from PIL import Image
        fm = SmolEdgeVLM(a.weights_dir, device=a.ref_device)
        files = sorted(f for f in os.listdir(a.ref_images) if f.lower().endswith(('.jpg', '.jpeg', '.png')))[:a.ref_count]
        meta['reference'] = {}
        for f in files:
            img = Image.open(os.path.join(a.ref_images, f)).convert('RGB')
            meta['reference'][f] = {m: fm.caption(img, m, max_new=24)['caption'] for m in ('full', 'smoledgevlm')}
            print('  reference', f, meta['reference'][f])
    mpath = os.path.join(a.out, 'smoledgevlm_jetson_meta.json')
    json.dump(meta, open(mpath, 'w'))
    print('wrote %s' % mpath)
    print('Copy to the Jetson: smoledgevlm_jetson.py, benchmark.py, %s, %s and an images folder.'
          % (os.path.basename(wpath), os.path.basename(mpath)))


if __name__ == '__main__':
    main()
