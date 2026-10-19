#import "lib.typ": *

#let global-structure() = cetz.canvas({
  import cetz.draw: *
  setup()

  let main-y = -5
  let vec-y = -2.5
  let mod-y = -3.5
  let txt-y = -6.5
  let pe-txt-y = -7.8
  let pe-y = -8.6
  let cat-x = 9.6
  let xs = (12.6, 15.6, 20.4)
  let dots-x = 18
  let tail-x = 22.3
  let last-x = 23.6

  // INPUTS
  io((0, 0), "in-img", raw("img"), side: "east")
  io((0, vec-y), "in-t", raw("timesteps"), side: "east")
  io((-6.2, txt-y), "in-prompt", raw("prompt"), side: "east")
  io((0, -11.6), "in-txt-ids", raw("txt_ids"), side: "east")
  io((0, -9.6), "in-img-ids", raw("img_ids"), side: "east")

  // image tokens
  block((2, 0), "img-in", "Linear")
  arrow("in-img.east", "img-in.west")
  note((1, 0), [HW × 128])
  note((cat-x - 1.3, 0), [HW × 3072])

  // timestep embedding
  block((2, vec-y), "t-sin", "Sinusoidal emb.", h: 2.1)
  block((3.6, vec-y), "t-mlp", "MLP Emb.")
  arrow("in-t.east", "t-sin.west")
  arrow("t-sin.east", "t-mlp.west")
  note(("t-sin.east", 50%, "t-mlp.west"), [256])

  // text: chat template, frozen Qwen3.5-9B, then the text adapter
  block((-4.6, txt-y), "tpl", "Chat template", h: 2.1)
  frozen((-3, txt-y), "qwen", [Qwen3.5-9B\ (frozen)], h: 2.1, w: 0.85)
  arrow("in-prompt.east", "tpl.west")
  arrow("tpl.east", "qwen.west")
  io((-1.1, txt-y), "txt", raw("txt"))
  arrow("qwen.east", "txt.west")
  note((-3, txt-y - 1.1), [last hidden state], side: "north")
  block((2, txt-y), "txt-in", "Linear")
  arrow("txt.east", "txt-in.west")
  note((0.7, txt-y), [L × 4096])
  stage((4.3, txt-y), "tb1", [Text\ Block], w: 1.6)
  stage((6.7, txt-y), "tb2", [Text\ Block], w: 1.6)
  arrow("txt-in.east", "tb1.west")
  arrow("tb1.east", "tb2.west")
  note((cat-x - 1.3, txt-y), [L × 3072])

  // shared modulation, computed once for the whole stack
  block((5.6, vec-y), "mod", "Modulation")
  arrow("t-mlp.east", "mod.west")
  wire("mod.east", (rel: (0.6, 0)), (rel: (0, mod-y - vec-y)), (xs.last(), mod-y))
  note((7.4, mod-y), [mod], side: "south-west")
  note((7.4, mod-y - 0.05), [6 × 3072], side: "north-west")

  // vec skips the stack and only conditions the last layer
  wire((4.05, vec-y), (4.05, vec-y + 1.2), (last-x, vec-y + 1.2))
  note((6, vec-y + 1.2), [vec])
  arrow((last-x, vec-y + 1.2), (last-x, main-y + block-h / 2))

  // positions, the text blocks only use the text part
  block((2, -11.6), "rope-txt", "4D RoPE")
  block((2, -9.6), "rope-img", "4D RoPE")
  arrow("in-txt-ids.east", "rope-txt.west")
  arrow("in-img-ids.east", "rope-img.west")
  op((cat-x, -10.6), "pe-cat", "CAT")
  arrow("rope-img.east", ((), "-|", "pe-cat"), "pe-cat.north")
  arrow("rope-txt.east", ((), "-|", "pe-cat"), "pe-cat.south")
  note((4, -11.6), [P.E. txt])
  note((4, -9.6), [P.E. img])
  wire((5.5, -11.6), (5.5, -9.75))
  arc((5.5, -9.75), start: -90deg, stop: 90deg, radius: 0.15)
  wire((5.5, -9.45), (5.5, pe-txt-y), (6.7, pe-txt-y))
  wire((4.3, pe-txt-y), (5.5, pe-txt-y))
  arrow((4.3, pe-txt-y), "tb1.south")
  arrow((6.7, pe-txt-y), "tb2.south")
  wire("pe-cat.east", (rel: (0.6, 0)), (rel: (0, pe-y + 10.6)), (xs.last(), pe-y))
  note((cat-x + 1.4, pe-y), [P.E.])

  // joint sequence
  op((cat-x, main-y), "cat", "CAT")
  arrow("img-in.east", ((), "-|", "cat"), "cat.north")
  arrow("tb2.east", ((), "-|", "cat"), "cat.south")

  // main stack
  for (i, x) in xs.enumerate() {
    stage((x, main-y), "b" + str(i), [Block])
    arrow((x, mod-y), "b" + str(i) + ".north")
    arrow((x, pe-y), "b" + str(i) + ".south")
  }
  dots((dots-x, main-y))
  arrow((dots-x, mod-y), (dots-x, main-y + 0.65))
  arrow((dots-x, pe-y), (dots-x, main-y - 0.65))
  arrow("cat.east", "b0.west")
  note(("cat.east", 50%, "b0.west"), [(HW + L) × 3072])
  arrow("b0.east", "b1.west")
  arrow("b1.east", (dots-x - 0.6, main-y))
  arrow((dots-x + 0.6, main-y), "b2.west")
  content(((xs.first() + xs.last()) / 2, vec-y + 0.5), text(size: 15pt)[N = 42])

  // tail, only the image tokens reach the last layer
  block((last-x, main-y), "last", "LastLayer")
  arrow("b2.east", "last.west")
  note((tail-x + 0.6, main-y), [img])
  arrow((tail-x, main-y), (tail-x, main-y - 1.3))
  content((tail-x, main-y - 1.55), text(size: 16pt)[×])
  note((tail-x - 0.08, main-y - 0.6), [txt], side: "east")
  io((25.6, main-y), "out", raw("v"), side: "west")
  arrow("last.east", "out.west")
  note((24.6, main-y), [HW × 128])
})
