#import "lib.typ": *

// How images map to the 128 channel tokens the DiT sees, and back
#let latent-space() = cetz.canvas({
  import cetz.draw: *
  setup()

  // encode
  io((0, 0), "in", raw("image"), side: "east")
  frozen((1.4, 0), "enc", [FLUX.2 VAE\ encoder], h: 2.1, w: 0.85)
  block((4.2, 0), "patch", [Patchify 2×2], h: 2.1)
  block((7, 0), "bn", "BatchNorm")
  block((8.3, 0), "flat", "Flatten")
  io((10.6, 0), "out", raw("img"), side: "west")
  arrow("in.east", "enc.west")
  arrow("enc.east", "patch.west")
  arrow("patch.east", "bn.west")
  arrow("bn.east", "flat.west")
  arrow("flat.east", "out.west")
  note(("enc.east", 50%, "patch.west"), [32 × H/8 × W/8])
  note(("patch.east", 50%, "bn.west"), [128 × H/16 × W/16])
  note(("flat.east", 50%, "out.west"), [HW × 128])

  // decode
  let y = -3.2
  io((0, y), "z", raw("x_0"), side: "east")
  block((1.4, y), "unflat", "Unflatten")
  op((2.7, y), "std", "×")
  op((3.7, y), "mean", "+")
  block((4.9, y), "unpatch", [Unpatchify 2×2], h: 2.1)
  frozen((7.7, y), "dec", [FLUX.2 VAE\ small decoder], h: 2.1, w: 0.85)
  io((9.2, y), "img", raw("image"), side: "west")
  arrow("z.east", "unflat.west")
  arrow("unflat.east", "std.west")
  arrow("std.east", "mean.west")
  arrow("mean.east", "unpatch.west")
  arrow("unpatch.east", "dec.west")
  arrow("dec.east", "img.west")
  arrow((2.7, y - 1), "std.south")
  arrow((3.7, y - 1), "mean.south")
  note((2.7, y - 1.05), [√(σ² + ε)], side: "north")
  note((3.7, y - 1.05), [μ], side: "north")
  note(("unpatch.east", 50%, "dec.west"), [32 × H/8 × W/8])
})

// One sampling step: CFG where the negative branch also skips block 22, then an Euler update
#let guidance() = cetz.canvas({
  import cetz.draw: *
  setup()

  io((0, 0), "x", raw("x_t"), side: "east")
  io((1.6, 2.2), "pos", raw("prompt"), side: "south")
  io((1.6, -3.2), "neg", raw("negative"), side: "north")
  stage((2.8, 1), "dit-c", [MOSS-Image], w: 2.2, h: 1)
  stage((2.8, -1.4), "dit-u", [MOSS-Image\ #small[block 22 skipped]], w: 2.2, h: 1.2)
  wire("x.east", (0.9, 0))
  wire((0.9, 1), (0.9, -1.4))
  arrow((0.9, 1), "dit-c.west")
  arrow((0.9, -1.4), "dit-u.west")
  arrow("pos.south", (1.6, 1.5))
  arrow("neg.north", (1.6, -2))

  op((5, 1), "sub", "−")
  arrow("dit-c.east", "sub.west")
  note((4.35, 1), [v#sub[c]])
  note((4.35, -1.4), [v#sub[u]])
  op((6.1, 1), "cfg", "×")
  arrow("sub.east", "cfg.west")
  arrow((6.1, 2.1), "cfg.north")
  note((6.1, 2.15), [s = 3], side: "south")
  op((7.2, -1.4), "add", "+")
  wire("dit-u.east", (5, -1.4))
  arrow((5, -1.4), "sub.south")
  arrow((5, -1.4), "add.west")
  arrow("cfg.east", ((), "-|", "add"), "add.north")

  op((8.6, -1.4), "dt", "×")
  arrow("add.east", "dt.west")
  note((7.9, -1.4), [v])
  arrow((8.6, -2.5), "dt.south")
  note((8.6, -2.55), [σ#sub[i+1] − σ#sub[i]], side: "north")
  op((9.8, 0), "step", "+")
  arrow("dt.east", ((), "-|", "step"), "step.south")
  wire((0.9, 0), (0.9, 3), (9.8, 3))
  arrow((9.8, 3), "step.north")
  io((11, 0), "next", raw("x_t+1"), side: "west")
  arrow("step.east", "next.west")
})
