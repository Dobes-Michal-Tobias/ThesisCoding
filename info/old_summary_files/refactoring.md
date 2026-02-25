
1. M1/S1
        import and setup
        data loading (fution from new modul load_data.py)
        datacheck (called from new module…)
        experimetnal loop (also from module?)
                regular saving to csv
                metrics:
                    f1, precision, recall, auprc
        load from csv -> result table (+ barplot for test and train, metrics from parametr)
        find best model -> fuction get_model(model=best_model, scenerio=best_scenerio) but also used for user choice.)
        retraining model (10-fold), detailed analysis
                pr_curve, plot_treshold_tuning (for M1), anomaly_hist (for M1), confusion_matrix… from visualiazation.py
                emebedding projection (umap, pca, tsne) for L0xL1 and TNxTPxFNxFP
        qualitative analysis
            make csv with concrete tokens are recognizable in model

2. similiar M1/S2 and M2/S1 or S2…

(5. Train and test Neural network)

6. Evaluate skills of general LLM.
 




## Modules



- models.py - wrapper for supervised and unsupervised model

- config.py

- load_process_data.py - fuction for process raw data to interim data and to vektors .pkl

- visualization.py - fuction for visualization in notebooks (pr_curve, barplots for models test adn train validation, embedding projection…)

- analysis (need a complete refactoring!)


### New modules:

- load_data.py (for loading fuction for every strategy, called in the start of every notebook.)

- eda.py (for  fuction to visulazite and explore statistic info from processd_dataset. )




--

## PROCESS DATA


Two version of get (contextual) embeddings 
- Word-level: Use contextual embeddings only on every (word) token separately. It may stop data leakage.
- Sentence-level: For sentence level make new emebddings (a) CLS (b) mean from contextual(?) emebeddign of every token/token

- I would like to use contextual embedding, bacuase they are better. But does it make sense yet? Instead of using static embeddings?




-> Keep in mind qualitative analysis from the start. For example:

pkl = {
    "X": embeddings,
    "token": tokens,
    "text_id": text_ids,       # odkud to je
    "vector_id": vector_id,     # nové id v pkl
    "position": positions,     # pozice v textu
    "label": y,
}

or 

df = pd.DataFrame({
    "token": tokens,
    "embedding": list(embeddings),
    "label": y,
})
df.to_pickle("data.pkl")
                                -> that would mean refactor from pkl to pd

--



