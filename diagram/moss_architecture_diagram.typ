#import "lib.typ": *
#import "global.typ": global-structure
#import "block.typ": moss-block
#import "text_block.typ": text-block
#import "small.typ": *
#import "attention.typ": *
#import "pipeline.typ": *

#set document(title: "MOSS-Image architecture diagram", author: "Valérien")
#set page(width: auto, height: auto, margin: 1.5cm, fill: white)
#set text(font: font)
#set align(center)

#let fig(body, name, size: 15pt, caption: none) = box(align(center, stack(
  dir: ttb,
  spacing: 0.6cm,
  body,
  text(size: size, name),
  ..if caption != none { (text(size: 7pt, caption),) },
)))

#let row(..figs) = align(center + bottom, stack(dir: ltr, spacing: 2.2cm, ..figs.pos().map(f => align(bottom, f))))

#stack(
  dir: ttb,
  spacing: 2.4cm,
  fig(scale(135%, reflow: true, global-structure()), [MOSS-Image global architecture], size: 30pt),
  row(
    fig(moss-block(), [Block], caption: [same `mod` for all 42 blocks]),
    fig(text-block(), [Text Block]),
  ),
  row(
    fig(mlp-emb(), [MLP Emb.]),
    fig(modulation(), [Modulation], caption: [computed once from `vec`, shared by every block]),
    fig(last-layer(), [LastLayer], caption: [image tokens only]),
    fig(situ-glu(), [SiTU-GLU]),
    fig(softcap(), [Softcap], caption: [softcap#sub[β]\(x) = β tanh(x / β)]),
  ),
  row(
    fig(rope-attention(), [RoPE Attention]),
    fig(rope-apply(), [RoPE]),
    fig(stack(dir: ttb, spacing: 0.5cm, rope-4d(), position-table()), [4D RoPE], caption: [r = 0 for every token, so the r quarter of each head is never rotated]),
    fig(latent-space(), [FLUX.2 latent space]),
    fig(guidance(), [Sampling step], caption: [default: 50 Euler steps, shift 5, CFG 3]),
  ),
  align(left, text(size: 8pt)[
    References
    - source code: github.com/OpenMOSS/MOSS-Image (`pipeline_moss_image.py`) and ComfyUI `comfy/ldm/moss_image`
    - MOSI.AI, _MOSS-Image Technical Report_, 2026 (arXiv:2610.23374)
    - shared modulation: Black Forest Labs, FLUX, 2024
    - SiTU-GLU: Kimi Team, _Kimi K3: Open frontier intelligence_, 2026
    - H, W: latent token grid (image size / 16), L: number of prompt tokens
  ]),
)
