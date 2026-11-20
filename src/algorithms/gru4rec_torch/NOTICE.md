# Vendored: official GRU4Rec PyTorch implementation

* Source: https://github.com/hidasib/GRU4Rec_PyTorch_Official  (commit d1fc31105577665d2f105de1c8d9a38d223a9a22)
* Files: `gru4rec_pytorch.py` is the official file with **two local changes** (see `local_changes.patch`): `.values` so it works
  with pandas 3, and `float(...)` around `sigma` so NumPy 2 keeps the initial weights float32 (as NumPy 1 did); nothing else differs; `official_paramfiles/` holds the authors' best parameters.
  Only `adapter.py` is ours (it wraps the official class for this repo's runner).
* License: the repository states "Final license is coming soon! Meanwhile, use the license of the original
  (https://github.com/hidasib/GRU4Rec/blob/master/license.txt), i.e. free for research and education, but contact me
  for commercial use." Keep this in mind before any commercial use or redistribution.
* The README asks to cite:
  [1] B. Hidasi, A. Karatzoglou, L. Baltrunas, D. Tikk, "Session-based Recommendations with Recurrent Neural Networks",
      ICLR 2016 (https://arxiv.org/abs/1511.06939)
  [2] B. Hidasi, A. Karatzoglou, "Recurrent Neural Networks with Top-k Gains for Session-based Recommendations",
      CIKM 2018 (https://arxiv.org/abs/1706.03847)
