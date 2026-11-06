"""Compute independent tail folds while the serial driver handles the head.

Only fold enumeration and the completion-marker destination are adapted.
No fitting, encoding, prediction or selection function is changed.
The serial driver later reuses these completed folds, validating all results.
"""
from pathlib import Path
import torch
from diagnostics_v2 import rolling
from diagnostics_v2.common import config,save_json
from iclab.data import sha256


def main():
    cfg,v1=config();out=Path(cfg["output_dir"]);torch.set_num_threads(cfg["torch_threads"])
    original_load=rolling.load;original_save=rolling.save_json
    def tail_load(c):
        raw,y,meta,bars,folds=original_load(c)
        return raw,y,meta,bars,[folds[0]]+folds[17:]
    def tail_save(path,obj):
        if Path(path)==out/"rolling_complete.json":
            path=out/"parallel_tail_complete.json"
        return original_save(path,obj)
    save_json(out/"parallel_execution.json",{"tail_folds":list(range(17,23)),"head_driver":"serial driver reuses completed tail folds","helper_sha256":sha256(__file__),"core_training_and_selection_unchanged":True,"completion_marker":"parallel_tail_complete.json","is_llm_delegation":False})
    rolling.load=tail_load;rolling.save_json=tail_save
    rolling.run(cfg,v1)


if __name__=="__main__":main()
