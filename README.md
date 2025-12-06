# Federated-Diabetes-Research

```
/Diabetes-Federated-MultiTask-Research
│
├── /data_loaders        # Scripts to load and preprocess data
│   ├── load_130US.py    # Shared by Comp 2 & 4
│   ├── load_retina.py   # For Comp 3 (Images)
│   └── load_iot.py      # For Comp 1 & 4
│
├── /models              # Deep Learning Architectures
│   ├── causal_graph.py  # Comp 2
│   ├── multimodal_net.py# Comp 3 (CNN + MLP)
│   └── multitask_net.py # Comp 4 (Shared layers + specific heads)
│
├── /strategies          # FL Aggregation Logic
│   ├── fed_avg.py       # Standard
│   ├── fed_prox.py      # For Non-IID (Comp 4)
│   └── secure_agg.py    # For Privacy/DP (Comp 1)
│
├── /experiments         # The main execution scripts
│   ├── run_comp1_privacy.py
│   ├── run_comp2_causal.py
│   ├── run_comp3_multimodal.py
│   └── run_comp4_personalization.py
│
└── requirements.txt
```
