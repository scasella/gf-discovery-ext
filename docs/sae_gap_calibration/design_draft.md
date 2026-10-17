DRAFT design low_variance (fields to be put into StructuredOutput)

title: Stage 4 v3-LV — identity-removed occurrence ICA on raw-residual PCs 65–256 at L12

layers: L12 only. Block A = PCs 65–256. Out of scope: L7 (rig fails), L24, PCs 257–512 (later prereg on spare 10,000–12,999).

representation: raw h at decoder-layer-12 output; bf16 model; e68_extract_raw (e60 minus SAE), 1038 zeroed, fp16; positions>=4, types>=8; type/doc/pos means x3; full PCA on every 3rd row, fitted on P3 (calib) and D3 (discovery) only; R3/S3 use D3 maps; block A unit-variance; second moment from {none, (256,128), (512,256)} by v2 Step-A rule; ICA-192 logcosh+skew, every 2nd row, max_iter 1000; reference ICA-64 on PCs 1–64 for shadow screen.

positive_control: paren (tuned) + quote (untuned); floor clip(0.5+0.6(probe_P3−0.5), 0.65, 0.85); null token-matched 1,000 perms p99.9; 4 looks; shadow self-test; planted-source detection limit (reported).

control_fallback: P3 ladder (Step-A choice -> other 2 settings -> block 65–192); else STOP before D3; optional planted-only continuation capped at exploratory geometric; D3 control fail -> STOP; anchor fail -> geometric-only.

data: P3 2,000–2,999; D3 3,000–4,999; S3 = D3 shuffled; R3 5,000–6,999; B5 7,000–7,999; B6 8,000–8,999; C3 9,000–9,999 sealed; spare 10,000–12,999, 15,000–15,695.

gates: G1 control; G2 S3 presence; G3 S3 yield >= 1.25x and >= +5; G4 R3; G5 coverage pre-steer; G6 B5 (rig anchor e64 #33, block anchor quote>paren); G7 B6; G8 C3.

p ≈ 0.04.
