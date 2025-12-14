# Federated-Diabetes-Research

THIS REPOSITORY DEFINES A SHARED FEDERATED LEARNING FRAMEWORK.
All four components use the same FL engine to ensure:
- consistency across experiments
- comparability of results
- unified system architecture


```
Federated-Diabetes-Research/
│
├── fl_core/                     # Shared FL engine (everyone uses this)
│   ├── server.py
│   └── client.py
│
├── components/
│   ├── component_1/             # Complication risk prediction (MTL optional)
│   ├── component_2/             # Causal discovery + FL explainability
│   ├── component_3/             # Multimodal FL model (CNN+MLP)
│   └──  component_4/             # YOUR MTFL comorbidity model
│
├── datasets/
│   ├── diabetes_130/            # Component 4 + Component 2
│   ├── complication_dataset/    # Component 1
│   └── multimodal_dataset/      # Component 3
│
├── experiments/
│   ├── comp1_experiments/
│   ├── comp2_experiments/
│   ├── comp3_experiments/
│   └── comp4_experiments/
│
├── results/
│   ├── comp1_results/
│   ├── comp2_results/
│   ├── comp3_results/
│   └── comp4_results/
│
├── dashboard.py
├── main_fl_runner.py
├── README.md                   # Top-level description of project
└── requirements.txt                  

```

## Development Flow For Everyone

1. Clone repo
2. Import fl_core
3. Prepare their dataset
4. Define their model
5. Plug model + data into FL runner
6. Run experiments & save results
