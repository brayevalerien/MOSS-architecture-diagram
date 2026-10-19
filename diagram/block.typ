#import "lib.typ": *

// One of the 42 joint blocks. Residual stream on y = 0, attention and FFN branches above it,
// the shared modulation comes from below.
#let moss-block() = cetz.canvas({
  import cetz.draw: *
  setup()

  let a-y = 2.6
  let q-y = a-y + 1.3
  let v-y = a-y - 1.3
  let pe-y = q-y + 1.4

  // residual stream
  io((-1.4, 0), "in", raw("x"), side: "east")
  io((21.6, 0), "out", raw("x"), side: "west")
  op((10.2, 0), "res-a", "+")
  op((20.2, 0), "res-f", "+")
  arrow("in.east", "res-a.west")
  arrow("res-a.east", "res-f.west")
  arrow("res-f.east", "out.west")

  // ATTENTION
  block((1.8, a-y), "norm-a", "RMSNorm")
  arrow((0.8, 0), (0.8, a-y), "norm-a.west")
  op((3, a-y), "scale-a", "×")
  op((4, a-y), "shift-a", "+")
  arrow("norm-a.east", "scale-a.west")
  arrow("scale-a.east", "shift-a.west")
  block((5.2, a-y), "qkv", "Linear", h: 3.4)
  arrow("shift-a.east", "qkv.west")
  block((6.6, q-y), "q-norm", "RMSNorm", h: 1.1, size: 6.5pt)
  block((6.6, a-y), "k-norm", "RMSNorm", h: 1.1, size: 6.5pt)
  block((8, a-y), "attn", [RoPE\ Attention], h: 3.4, w: 0.75)
  for (y, l, n) in ((q-y, "Q", "q-norm"), (a-y, "K", "k-norm")) {
    arrow((5.2 + block-w / 2, y), n + ".west")
    arrow(n + ".east", (8 - 0.375, y))
    note((5.75, y), l)
  }
  arrow((5.2 + block-w / 2, v-y), (8 - 0.375, v-y))
  note((5.75, v-y), "V")
  io((-1.4, pe-y), "pe", [P.E.], side: "east")
  arrow("pe.east", (8, pe-y), "attn.north")
  note((8, v-y - 0.45), [24 heads × 128], side: "north")

  block((9.2, a-y), "o", "Linear")
  arrow("attn.east", "o.west")
  op((10.2, 1.2), "gate-a", "×")
  arrow("o.east", (10.2, a-y), "gate-a.north")
  arrow("gate-a.south", "res-a.north")

  // FFN
  block((11.8, a-y), "norm-f", "RMSNorm")
  arrow((11.35, 0), (11.35, a-y), "norm-f.west")
  op((13, a-y), "scale-f", "×")
  op((14, a-y), "shift-f", "+")
  arrow("norm-f.east", "scale-f.west")
  arrow("scale-f.east", "shift-f.west")
  block((15.2, a-y), "up", "Linear")
  arrow("shift-f.east", "up.west")
  block((16.4, a-y), "chunk-f", "chunk", h: 2.2)
  arrow("up.east", "chunk-f.west")
  note(("up.east", 50%, "chunk-f.west"), [16384])
  block((17.9, a-y), "situ", "SiTU-GLU", h: 2.2)
  arrow((16.4 + block-w / 2, a-y + 0.6), (17.9 - block-w / 2, a-y + 0.6))
  arrow((16.4 + block-w / 2, a-y - 0.6), (17.9 - block-w / 2, a-y - 0.6))
  note((17.15, a-y + 0.6), "u")
  note((17.15, a-y - 0.6), "g")
  block((19.2, a-y), "down", "Linear")
  arrow("situ.east", "down.west")
  note(("situ.east", 50%, "down.west"), [8192])
  op((20.2, 1.2), "gate-f", "×")
  arrow("down.east", (20.2, a-y), "gate-f.north")
  arrow("gate-f.south", "res-f.north")

  // SHARED MODULATION, the same six vectors for every block
  io((-1.4, -3), "mod", raw("mod"), side: "east")
  block((0, -3), "chunk-m", "chunk", h: 3.8)
  arrow("mod.east", "chunk-m.west")
  let one-y = 1.2
  // same order as the chunk in the code, so each scale wire hops over the shift wire above it
  let step = 0.62
  let hop = 0.15
  let outs = (
    ("shift", "a", 4, "shift-a"),
    ("scale", "a", 3, "scale-a"),
    ("gate", "a", 10.9, "gate-a"),
    ("shift", "f", 14, "shift-f"),
    ("scale", "f", 13, "scale-f"),
    ("gate", "f", 20.9, "gate-f"),
  )
  for (i, (kind, idx, x, target)) in outs.enumerate() {
    let y = -1.45 - i * step
    let lbl = [#kind#sub(idx)]
    note((block-w / 2 + 0.08, y + 0.02), lbl, side: "south-west")
    if kind == "scale" {
      op((x, one-y), "one-" + target, "+", r: 0.2)
      wire((block-w / 2, y), (x, y), (x, y + step - hop))
      arc((x, y + step - hop), start: -90deg, stop: 90deg, radius: hop)
      arrow((x, y + step + hop), "one-" + target + ".south")
      arrow("one-" + target + ".north", target + ".south")
      arrow((x - 0.8, one-y), "one-" + target + ".west")
      note((x - 0.85, one-y), "1", side: "east")
    } else if kind == "shift" {
      wire((block-w / 2, y), (x, y))
      arrow((x, y), target + ".south")
    } else {
      wire((block-w / 2, y), (x, y), (x, 1.2))
      arrow((x, 1.2), target + ".east")
    }
  }
})
