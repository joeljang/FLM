import numpy as np
from dataclasses import dataclass
from transformers import DataCollatorForSeq2Seq

@dataclass
class TaskDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
   def check_uniqueness(self, samples):
      assert len(np.unique(samples)) == 1
   
   def __call__(self, features):
      #tasks = [d.pop('task') for d in features]
      labels_list_exist=False
      if 'labels_list' in features[0]:
         labels_list_exist=True
         labels_list = [d.pop('labels_list') for d in features]
      if 'source' in features[0]:
         sources = [d.pop('source') for d in features]
      if 'target' in features[0]:
         targets = [d.pop('target') for d in features]
      #self.check_uniqueness(tasks)
      output = super().__call__(features)
      if labels_list_exist:
         output["labels_list"] = labels_list
      return output