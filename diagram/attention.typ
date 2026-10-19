#import "lib.typ": *

#let rope-attention() = cetz.canvas({
  import cetz.draw: *
  setup()
  io((0, 1.2), "q", raw("Q"), side: "east")
  io((0, 0), "k", raw("K"), side: "east")
  io((0, -1.4), "v", raw("V"), side: "east")
  block((1.4, 1.2), "rq", "RoPE", h: 1)
  block((1.4, -0.1), "rk", "RoPE", h: 1)
  arrow("q.east", "rq.west")
  arrow("k.east", (1.4 - block-w / 2, 0))
  io((0, 2.6), "pe", [P.E.], side: "east")
  wire("pe.east", (0.9, 2.6))
  wire((0.9, 2.6), (0.9, -0.85), (1.4, -0.85))
  arrow((0.9, 2.6), (1.4, 2.6), "rq.north")
  arrow((1.4, -0.85), "rk.south")
  op((2.8, 0.6), "qk", "·")
  arrow((1.4 + block-w / 2, 1.2), ((), "-|", "qk"), "qk.north")
  arrow((1.4 + block-w / 2, 0), ((), "-|", "qk"), "qk.south")
  note((2.2, 0), [Kᵀ], side: "south")
  op((3.9, 0.6), "scale", "÷")
  arrow("qk.east", "scale.west")
  arrow((3.9, -0.5), "scale.south")
  note((3.9, -0.55), [√128], side: "north")
  block((5.1, 0.6), "soft", "Softmax")
  arrow("scale.east", "soft.west")
  op((6.3, 0.6), "av", "·")
  arrow("soft.east", "av.west")
  arrow("v.east", ((), "-|", "av"), "av.south")
  arrow("av.east", (7.4, 0.6))
  note((5.1, -0.35), [per head], side: "north")
})

// Rotate-half RoPE as implemented: x cos + CAT(−x₂, x₁) sin, the halves pair dimension i with i + 64
#let rope-apply() = cetz.canvas({
  import cetz.draw: *
  setup()
  io((0, 0), "x", raw("x"), side: "east")
  block((1.5, -1.6), "chunk", "chunk", h: 1.4)
  arrow((0.7, 0), (0.7, -1.6), "chunk.west")
  let out = 1.5 + block-w / 2
  op((3.1, -0.8), "neg", "−", r: 0.2)
  arrow((out, -2), (2.5, -2), (2.5, -0.8), "neg.west")
  op((4, -1.7), "cat", "CAT")
  arrow("neg.east", ((), "-|", "cat"), "cat.north")
  arrow((out, -1.2), (2.2, -1.2), (2.2, -2.7), (4, -2.7), "cat.south")
  note((out + 0.05, -1.2), [x#sub[1]], side: "south-west")
  note((out + 0.05, -2), [x#sub[2]], side: "south-west")
  op((5.2, -1.7), "msin", "×")
  arrow("cat.east", "msin.west")
  op((5.2, 0), "mcos", "×")
  arrow("x.east", "mcos.west")
  op((6.4, 0), "add", "+")
  arrow("mcos.east", "add.west")
  arrow("msin.east", ((), "-|", "add"), "add.south")
  arrow("add.east", (7.4, 0))
  arrow((5.2, 1.1), "mcos.north")
  note((5.2, 1.15), [cos], side: "south")
  arrow((5.2, -2.8), "msin.south")
  note((5.2, -2.85), [sin], side: "north")
})

// Builds P.E. from the 4D position ids, 16 frequency bands per axis
#let rope-4d() = cetz.canvas({
  import cetz.draw: *
  setup()
  io((0, 0), "ids", [(r, t, h, w)], side: "east")
  op((2, 0), "mul", "×")
  arrow("ids.east", "mul.west")
  arrow((2, -1.1), "mul.south")
  note((2, -1.15), [ω#sub[j] = 10000#super[−j/16], j < 16], side: "north")
  block((3.6, 0.9), "sin", "sin", h: 1)
  block((3.6, -0.9), "cos", "cos", h: 1)
  wire("mul.east", (3, 0))
  wire((3, 0.9), (3, -0.9))
  arrow((3, 0.9), "sin.west")
  arrow((3, -0.9), "cos.west")
  op((5, 0), "cat", "CAT")
  arrow("sin.east", ((), "-|", "cat"), "cat.north")
  arrow("cos.east", ((), "-|", "cat"), "cat.south")
  arrow("cat.east", (6.3, 0))
  note((5.95, 0), [P.E.])
  note((2.6, -0.05), [4 × 16], side: "north")
})

#let position-table() = {
  set text(size: 7pt)
  table(
    columns: 5,
    stroke: none,
    align: center,
    inset: (x: 6pt, y: 3pt),
    table.hline(stroke: 0.6pt),
    [token], [r], [t], [h], [w],
    table.hline(stroke: 0.4pt),
    [image (i, j)], [0], [L], [i − (H − 1) / 2], [j − (W − 1) / 2],
    [text k], [0], [k], [0], [0],
    table.hline(stroke: 0.6pt),
  )
}
