"""Fixed settings of the final SmolEdgeVLM model."""

VISION_ID = 'google/siglip-base-patch16-512'      # frozen vision encoder (12 layers, 512 x 512 input, 16 x 16 patches)
LM_ID = 'HuggingFaceTB/SmolLM2-135M'              # frozen language model
CLIP_ID = 'openai/clip-vit-base-patch32'          # used only for CLIPScore
PROMPT = 'Describe this image in one sentence.'

CUT_LAYER = 3          # merge after SigLIP layer 3; layers 4 to 12 run on merged tokens
NUM_TOKENS = 256       # 1,024 patch tokens -> 256 merged tokens
RADIUS = 3             # a patch may join an anchor at most 3 patches away (7 x 7 window)
RESTORE_RANK = 192     # bottleneck width of the restoration module R
CONNECTOR_GRID = 16    # the connector pools the 32 x 32 grid to 16 x 16 = 256 tokens

MAX_CAPTION_TOKENS = 32     # training captions are truncated to this length
MAX_NEW_TOKENS = 24         # generation limit used in the reported results
