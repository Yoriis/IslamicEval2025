import pandas as pd
import os, json
from trectools import TrecEval, TrecRun, TrecQrel

def evaluate(input_run, qrels, depth):
    trec_run = TrecRun(input_run)
    trec_eval = TrecEval(trec_run, qrels)
    return round(trec_eval.get_map(depth=depth), 4)

def add_unscored_questions(df, qids):
    existing_ids = set(df['query'])
    missing_ids = [i for i in qids if i not in existing_ids]
    new_rows = pd.DataFrame({
        'query': missing_ids,
        "q0": "Q0",
        'docid': -1,
        'rank': 1,
        'score': 10,
        'system': "unretrieved_question"
    })
    df = pd.concat([df, new_rows], ignore_index=True)
    return df

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))  
reference_dir = os.path.join(current_dir, 'input', 'ref')
prediction_dir = os.path.join(current_dir, 'input', 'res')
score_dir = os.path.join(current_dir, 'output')

# Load qrels
quran_qrels = os.path.join(reference_dir, "quran_sample.qrels")
q_qrels = TrecQrel(quran_qrels)

# Unique query IDs
qids = list(set(q_qrels.qrels_data['query']))

all_scores = {}

for f in os.listdir(prediction_dir):
    print(f)
    if f.endswith(".tsv"):
        run_file = os.path.join(prediction_dir, f)
        
        # Load TSV into Pandas 
        rundata = pd.read_csv(run_file, sep='\t', header=None,
                              names=['query', 'q0', 'docid', 'rank', 'score', 'system'])
        
        # Clean up query/docid to ensure true duplicates are caught
        rundata['query'] = rundata['query'].astype(str).str.strip()
        rundata['docid'] = rundata['docid'].astype(str).str.strip()

        # Drop duplicate query-doc pairs, keeping the first occurrence
        rundata = rundata.drop_duplicates(subset=['query', 'docid'], keep='first')
        
        # Add missing queries
        rundata = add_unscored_questions(rundata, qids)
        
        # Save cleaned file
        updated_run_file = os.path.join(prediction_dir, f"updated_{f}")
        rundata.to_csv(updated_run_file, index=False, header=False, sep='\t')
        
        # load into TrecRun
        input_run = TrecRun(updated_run_file)
        
        # Evaluate
        map5 = evaluate(updated_run_file, q_qrels, depth=5)
        map10 = evaluate(updated_run_file, q_qrels, depth=10)
        
        all_scores[f] = {
            'MAP_Q@5': map5,
            'MAP_Q@10': map10
        }
        
        print(f"{f} → MAP_Q@5 = {map5}, MAP_Q@10 = {map10}")

# Save to JSON
with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
    score_file.write(json.dumps(all_scores, indent=2))
