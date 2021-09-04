Data folder. 

[Physionet data](https://www.physionet.org/content/mimic-eicu-fiddle-feature/1.0.0/) can be downloaded with the following command:

```
wget -r -N -c -np --user <username> --ask-password https://physionet.org/files/mimic-eicu-fiddle-feature/1.0.0/
```

Total uncompressed size: 1.7 GB.

Project code expects data in the following structure:

```
└── data
    ├── FIDDLE_eicu
    │   ├── features
    │   │   ├── ARF_12h
    │   │   │   ├── s.feature_aliases.json
    │   │   │   ├── s.feature_names.json
    │   │   │   ├── s.npz
    │   │   │   ├── X.feature_aliases.json
    │   │   │   ├── X.feature_names.json
    │   │   │   └── X.npz
    │   │   ├── ARF_4h
    │   │   │   ├── s.feature_aliases.json
    │   │   │   ...
    │   │   │   └── X.npz
    │   │   ├── mortality_48h
    │   │   │   ├── s.feature_aliases.json
    │   │   │   ...
    │   │   │   └── X.npz
    │   │   ├── Shock_12h
    │   │   │   ├── s.feature_aliases.json
    │   │   │   ...
    │   │   │   └── X.npz
    │   │   └── Shock_4h
    │   │   │   ├── s.feature_aliases.json
    │   │   │   ...
    │   │   │   └── X.npz
    │   └── population
    │       ├── ARF_12h.csv
    │       ├── ARF_4h.csv
    │       ├── index.html
    │       ├── mortality_48h.csv
    │       ├── Shock_12h.csv
    │       └── Shock_4h.csv
    └── FIDDLE_mimic3
        ├── features
        │   ├── ARF_12h
        │   │   ├── s.feature_aliases.json
        │   │   ...
        │   │   └── X.npz
        │   ├── ARF_4h
        │   │   ├── s.feature_aliases.json
        │   │   ...
        │   │   └── X.npz
        │   ├── mortality_48h
        │   │   ├── s.feature_aliases.json
        │   │   ...
        │   │   └── X.npz
        │   ├── Shock_12h
        │   │   ├── s.feature_aliases.json
        │   │   ...
        │   │   └── X.npz
        │   └── Shock_4h
        │   │   ├── s.feature_aliases.json
        │   │   ...
        │   │   └── X.npz
        └── population
            ├── ARF_12h.csv
            ...
            └── Shock_4h.csv
```