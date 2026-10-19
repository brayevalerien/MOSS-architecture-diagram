#import "lib.typ": *

// The two blocks of the text adapter, plain pre-norm transformer blocks, no modulation.
#let text-block() = cetz.canvas({
  import cetz.draw: *
  setup()

  let a-y = 2.6
  let q-y = a-y + 1.3
  let v-y = a-y - 1.3
  let pe-y = q-y + 1.4

  // residual stream
  io((-1.4, 0), "in", raw("txt"), side: "east")
  io((14.6, 0), "out", raw("txt"), side: "west")
  op((7.2, 0), "res-a", "+")
  op((13.2, 0), "res-f", "+")
  arrow("in.east", "res-a.west")
  arrow("res-a.east", "res-f.west")
  arrow("res-f.east", "out.west")

  // ATTENTION
  block((0.8, a-y), "norm-a", "RMSNorm")
  arrow((0, 0), (0, a-y), "norm-a.west")
  block((2.2, a-y), "qkv", "Linear", h: 3.4)
  arrow("norm-a.east", "qkv.west")
  block((3.6, q-y), "q-norm", "RMSNorm", h: 1.1, size: 6.5pt)
  block((3.6, a-y), "k-norm", "RMSNorm", h: 1.1, size: 6.5pt)
  block((5, a-y), "attn", [RoPE\ Attention], h: 3.4, w: 0.75)
  for (y, l, n) in ((q-y, "Q", "q-norm"), (a-y, "K", "k-norm")) {
    arrow((2.2 + block-w / 2, y), n + ".west")
    arrow(n + ".east", (5 - 0.375, y))
    note((2.75, y), l)
  }
  arrow((2.2 + block-w / 2, v-y), (5 - 0.375, v-y))
  note((2.75, v-y), "V")
  io((-1.4, pe-y), "pe", [P.E. txt], side: "east")
  arrow("pe.east", (5, pe-y), "attn.north")
  block((6.2, a-y), "o", "Linear")
  arrow("attn.east", "o.west")
  arrow("o.east", (7.2, a-y), "res-a.north")

  // FFN, separate up and gate projections here
  block((8.6, a-y), "norm-f", "RMSNorm")
  arrow((7.8, 0), (7.8, a-y), "norm-f.west")
  block((10, a-y + 0.9), "up", "Linear")
  block((10, a-y - 0.9), "gate", "Linear")
  wire("norm-f.east", (9.3, a-y))
  arrow((9.3, a-y + 0.9), "up.west")
  arrow((9.3, a-y - 0.9), "gate.west")
  wire((9.3, a-y + 0.9), (9.3, a-y - 0.9))
  block((11.3, a-y), "situ", "SiTU-GLU", h: 3.4)
  arrow("up.east", (11.3 - block-w / 2, a-y + 0.9))
  arrow("gate.east", (11.3 - block-w / 2, a-y - 0.9))
  note((10.75, a-y + 0.9), "u")
  note((10.75, a-y - 0.9), "g")
  block((12.4, a-y), "down", "Linear")
  arrow("situ.east", "down.west")
  arrow("down.east", (13.2, a-y), "res-f.north")
})
