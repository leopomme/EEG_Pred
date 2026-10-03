import unittest
import numpy as np
import pandas as pd
from eeg_comp.submission import build_submission

class SubmissionTests(unittest.TestCase):
    def frames(self):
        rows=[]
        for i in range(680):
            rows.append(dict(epoch_index=2000+i,competition='within_subject',split='test',
                subject=f'S{i//40+1:03}',run=3,trial_index=i%40,test_order=i,
                file=f'S{i//40+1:03}_test.csv',p_move=(i%2)*.8+.1))
        f=pd.DataFrame(rows)
        return f.drop(columns='p_move'),f.sample(frac=1,random_state=4)
    def test_prediction_order_restored_from_trial_mapping(self):
        meta,pred=self.frames(); result=build_submission(pred,meta,'within_subject')
        np.testing.assert_array_equal(result.ID,np.arange(680))
        np.testing.assert_array_equal(result.TARGET,np.where(np.arange(680)%2,'move','rest'))
    def test_wrong_mapping_duplicate_and_nan_rejected(self):
        meta,pred=self.frames()
        wrong=pred.copy();wrong.loc[0,'subject']='S999'
        with self.assertRaises(ValueError):build_submission(wrong,meta,'within_subject')
        wrong=pred.copy();wrong.loc[0,'p_move']=np.nan
        with self.assertRaises(ValueError):build_submission(wrong,meta,'within_subject')
        wrong=pd.concat([pred.iloc[:-1],pred.iloc[:1]])
        with self.assertRaises(ValueError):build_submission(wrong,meta,'within_subject')

if __name__=='__main__':unittest.main()
