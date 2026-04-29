# Born A Transformer, Always a Transformer — artifact

This repository contains **all code, configs and data used in our paper
“Born A Transformer -- Always a Transformer?”**

| section of paper | experiment category                                                   | folder                                                                |
| ---------------- | --------------------------------------------------------------------- | --------------------------------------------------------------------- |
| § 4.1            | **In-context learning** on synthetic *copy* / *retrieve* tasks        | [`copying/`](copying/README.md) · [`retrieval/`](retrieval/README.md) |
| § 4.2            | In-context learning on *real-world* tasks                             | [`realistic/`](realistic/README.md)                                   |
| § 4.3            | Supervised **fine-tuning** baselines                                  | [`finetuning/`](finetuning/README.md)                                 |
| § 4.4            | **Mechanistic probes** (induction-head patching, attention alignment) | [`mechanistic/`](mechanistic/README.md)                               |
| Appendix         | Training **from-scratch** ablations                                   | [`fromscratch/`](fromscratch/README.md)                               |

<sub>(Each link opens a dedicated README with setup & run instructions.)</sub>

---

## Repository map

```
copying/            # synthetic copy–ICL experiments (§4.1)
retrieval/          # retrieval–ICL experiments (§4.1)
realistic/          # ArXiv, lorem-ipsum, code-assist tasks (§4.2)
finetuning/         # supervised fine-tune baselines (§4.3)
mechanistic/        # induction-head probing & ablations (§4.4)
fromscratch/        # training tiny models from scratch (Appendix)
datasets/           # cached JSONL datasets
results/            # generated outputs (kept small in repo)
visualisations/     # plots for the paper
prompts/            # prompt templates
```

---

## Citation

```
@inproceedings{
    jobanputra2026born,
    title={Born a Transformer -- Always a Transformer? On the Effect of Pretraining on Architectural Abilities},
    author={Mayank Jobanputra and Yana Veitsman and Yash Sarrof and Aleksandra Bakalova and Vera Demberg and Ellie Pavlick and Michael Hahn},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2026},
    url={https://openreview.net/forum?id=Huw15LqglI}
}
```
