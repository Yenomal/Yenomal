manifest.yaml 是我们用于复现的主要环境配置，我们约束 manifest.yaml 的相关格式：

```yaml
experiment:
  name: simvla_baseline
  project: maevla
  description: "SimVLA baseline on RMBench without AutoGaze observation encoder."

tag:
  name: maevla-simvla-baseline-v1
  status: planned

third_party_refs:
  simvla_maevla:
    commit: b5c9a57371ae815ceb5b485101cfe0546a03ea3d
    tag: simvla-maevla-simvla-baseline-v1
    status: pending_snapshot
  rmbench:
    commit: ead77f635489228b4d621fe49af0390c33ecb4e1
    tag: rmbench-maevla-simvla-baseline-v1
    status: pending_snapshot

entrypoints:
  train: scripts/projects/maevla/simvla_baseline/train.sh
  eval: scripts/projects/maevla/simvla_baseline/eval.sh

third_party:
  - name: simvla_maevla
    path: third_party/simvla_maevla
  - name: rmbench
    path: third_party/rmbench

datasets:
  raw: datasets/raw/rmbench
  metas: datasets/metas/maevla/rmbench
  norm_stats: datasets/norm_stats/maevla/rmbench
  checkpoints: datasets/checkpoints/simvla

outputs:
  runs: outputs/runs/maevla/simvla_baseline
  eval: outputs/eval/rmbench
  logs: outputs/logs/maevla/simvla_baseline

notes:
  - "Experiment tag should be created only after the main repo snapshot is committed."
  - "Third-party tags should be created only after simvla_maevla and rmbench each have a clean committed snapshot."
```

我们复现的时候是按照exp为单位进行复现的，所以我建议 manifest 局限于各个 exp 文件夹下使用