"""Generate conf/ref_neural/<model>_seed<N>.yml from the nonneural-full config header."""
import re, os, sys
head = open('conf/ref_nonneural_full.yml').read().split('algorithms:\n')[0]
models = {
 'gru4rec': """- class: gru4rec.gru4rec.GRU4Rec
  params: { loss: 'top1-max', final_act: 'linear', dropout_p_hidden: 0.1, learning_rate: 0.08, momentum: 0.1, constrained_embedding: False }
  key: gru4rec""",
 'narm': """- class: narm.narm.NARM
  params: { epochs: 20, lr: 0.007, hidden_units: 100, factors: 100 }
  key: narm""",
 'stamp': """- class: STAMP.model.STAMP.Seq2SeqAttNN
  params: { init_lr: 0.003, n_epochs: 10, decay_rate: 0.4}
  key: stamp""",
 'sgnn': """- class: sgnn.gnn.GGNN
  params: { hidden_size: 100, out_size: 100, step: 1, nonhybrid: True, batch_size: 100, epoch_n: 10, batch_predict: True, lr: 0.006, l2: 3.00E-05, lr_dc: 0.36666667, lr_dc_step: 3}
  key: sgnn""",
}
os.makedirs('conf/ref_neural', exist_ok=True)
for seed in (1, 2, 3):
    for m, blk in models.items():
        h = head.replace('results/ref/nonneural-full/', f'results/ref/neural/{m}/seed{seed}/')
        open(f'conf/ref_neural/{m}_seed{seed}.yml', 'w').write(h + 'algorithms:\n' + blk + '\n')
