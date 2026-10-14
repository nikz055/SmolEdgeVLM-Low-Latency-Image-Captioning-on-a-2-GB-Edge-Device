# Method details and design choices

All design decisions were made on 300 COCO development images (caption loss on held-out references, paired
bootstrap 95% CI). The 500 test images were used only for the final numbers.

## 1. Where to reduce tokens

Per-layer timing of SigLIP-B/16@512 on all 1,024 patches showed that **73% of the encoder's time is in layers 4 to
12**. Reducing tokens after layer 3 makes those nine layers run on 256 tokens.

| Option | Why not chosen / chosen |
|---|---|
| Shrink the image before the encoder | Removes detail before any feature exists |
| Reduce tokens after the encoder (PruMerge, VisionZip, DeCo) | Saves at most ~15 ms here: the encoder cost stays |
| Drop tokens inside the language model (FastV) | The encoder cost is paid in full |
| Replace the encoder (FastVLM) | Needs pretraining a new encoder |
| **Reduce tokens between encoder layers** | **Chosen**: the only placement that reaches the 73% |

| Cut after layer | TTFT (T4) | vs full encoder | CIDEr-D |
|---|---|---|---|
| 2 | 67.0 ms | −41% | 0.647 |
| **3** | **71.7 ms** | **−37%** | **0.746** |
| 4 | 76.6 ms | −32% | |

## 2. How to merge

| Choice | Options compared | Decision |
|---|---|---|
| Reduce by | pruning, fixed 2 × 2 pooling, global ToMe matching, local merge into anchors | **Local merge**: every patch still contributes; ToMe-style global matching gave a larger loss rise (+0.422 vs +0.366) |
| Anchors | learned relevance head, encoder attention, fixed 16 × 16 lattice | **Fixed lattice**: no scoring pass and full coverage; after repair, no significant quality difference from attention anchors, and faster (71.7 vs 74.6 ms) |
| Which anchor a patch joins | most similar anywhere, nearest by position, **most similar within 7 × 7** | Content-aware and local |

Group sizes vary from image to image. The size bias adds $\log n_j$ to every attention score that points at
merged token $j$, so after the softmax it carries $n_j$ times the weight, as if its $n_j$ patches were present
(proportional attention, from ToMe). This is exact only if the merged patches were identical.

## 3. The restoration module R

Merging alone raises the caption loss by **+0.366**: the averaged tokens no longer look like what the frozen layers
4 to 12 expect. R repairs each merged token before those layers:

$$m'_j = m_j + W_2\,\mathrm{GELU}\big(W_1\,\mathrm{LN}(m_j)\big), \qquad W_1 \in \mathbb{R}^{192\times768},\; W_2 \in \mathbb{R}^{768\times192}$$

$W_2$ starts at zero, so R begins as the identity (0.30 M parameters, under 3 ms on the Jetson).

| Question | Options compared | Decision |
|---|---|---|
| How to fix the damage | cut later, fine-tune SigLIP, training-free norm-preserving merge (ToFu-style), **small residual module** | R: CIDEr-D 0.746 vs 0.661 for the training-free merge |
| Where to place R | after the encoder, **at the cut before layer 4** | At the cut: removes 86% of the damage; after the encoder made quality worse |
| What R learns from | caption loss, **matching the full encoder** | No captions needed, and R is not tied to one language model |
| What R sees | **merged token only**, + group size, + group spread, a full transformer layer | Extra inputs gave no measurable gain; a full layer costs ~7 M parameters |

**Training (stage 1).** The same image goes through the full frozen encoder (target $F$) and through
merge → R → layers 4-12 → unmerge (output $U$):

$$\mathcal{L}_R = \mathrm{relMSE}(U, F) + \mathrm{relMSE}\big(\mathrm{pool}(U), \mathrm{pool}(F)\big), \qquad \mathrm{relMSE}(A,B) = \frac{\mathrm{mean}\big((A-B)^2\big)}{\mathrm{Var}(B)}$$

where pool is the connector's 2 × 2 average pooling. Gradients pass through the frozen layers but update only R.
3,000 unlabelled COCO train images, one pass, batch 8, AdamW (lr 1e-3, cosine), about 6 minutes on a T4.

Effect with R (3 seeds): caption-loss rise +0.366 → +0.053; cosine similarity to the full encoder's tokens
0.42 → 0.62.

## 4. Connector (stage 2)

The base SigLIP and SmolLM2 were never trained together, so a small connector maps the 32 × 32 grid of SigLIP
features to 256 embeddings for SmolLM2: 2 × 2 average pooling, then Linear(768 → 576) → GELU → Linear(576 → 576)
(0.78 M parameters). It is trained with next-token cross-entropy through the frozen language model on 5,000 COCO
images (1,500 steps, batch 16, AdamW lr 1e-3 with 40 warm-up steps and cosine decay). The full-encoder baseline,
merge-without-R and SmolEdgeVLM each get their own connector with this identical recipe.

## 5. Compute

| | Full encoder | SmolEdgeVLM |
|---|---|---|
| Encoder, one 512 × 512 image | 213.8 GFLOPs | 89.6 GFLOPs (−58%) |
| Layers 4-12 | 159.5 GFLOPs | 34.4 GFLOPs (4.6× less) |
| Layers 4-12, measured on the Jetson | 1,325 ms | 251 ms (5.3× faster) |

With $N$ tokens, a layer's linear work grows with $N$ and its attention-score work with $N^2$; going from 1,024 to
256 tokens cuts them 4× and 16×. Memory does not change: the weights are the same plus R.

## 6. Parameters

| Component | Parameters | Status |
|---|---|---|
| SigLIP-B/16@512 (as loaded, with its unused pooling head) | 93 M | frozen |
| SmolLM2-135M (tied input/output embeddings) | 134.5 M | frozen |
| Pooling connector | 0.78 M | trained |
| Restoration module R | 0.30 M | trained |
| **Total** | **≈ 229 M** (1.08 M trained, 0.47%) | |
