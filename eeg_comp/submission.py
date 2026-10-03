"""Validate the official zero-based ID contract before writing a candidate."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED_ROWS = {'within_subject':680,'cross_subject':360}


def build_submission(predictions, metadata, competition):
    expected = metadata.loc[metadata.split.eq('test')].sort_values(['subject','run','trial_index']).copy()
    count = EXPECTED_ROWS[competition]
    if len(expected) != count or not expected.competition.eq(competition).all():
        raise ValueError('Wrong competition or test row count')
    if not np.array_equal(expected.test_order,np.arange(count)):
        raise ValueError('Test order differs from participant/session/epoch ordering')
    if predictions.epoch_index.duplicated().any() or len(predictions) != count:
        raise ValueError('Duplicate or missing prediction rows')
    aligned = predictions.set_index('epoch_index').reindex(expected.epoch_index)
    for column in ['competition','subject','run','file','trial_index','test_order']:
        if column not in aligned or not np.array_equal(aligned[column].to_numpy(),expected[column].to_numpy()):
            raise ValueError(f'Prediction mapping mismatch: {column}')
    probs = aligned.p_move.to_numpy(dtype=float)
    if not np.isfinite(probs).all() or ((probs<0)|(probs>1)).any():
        raise ValueError('Invalid or missing probabilities')
    return pd.DataFrame({'ID':np.arange(count),'TARGET':np.where(probs>=.5,'move','rest')})


def save_candidate(prediction_path, metadata_path, competition, category, output):
    prediction_path,metadata_path,output=map(Path,[prediction_path,metadata_path,output])
    if output.exists():
        raise FileExistsError(output)
    frame=build_submission(pd.read_csv(prediction_path),pd.read_csv(metadata_path),competition)
    output.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output,index=False)
    manifest={'competition':competition,'category':category,'rows':len(frame),
              'id_min':0,'id_max':len(frame)-1,'threshold':.5,
              'label_counts':frame.TARGET.value_counts().to_dict(),
              'prediction_file':str(prediction_path),'prediction_sha256':hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
              'metadata_sha256':hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
              'csv_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
              'status':'local candidate; not uploaded; Kaggle execution still required',
              'ordering_source':'Official competition evaluation pages; see docs/RULES_AND_SUBMISSION.md'}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
