"""Scientific identity, validation-only selection and physical metric checks."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1]/"experiments/synthetic/53_matched_reconstruction_report.py"
spec = importlib.util.spec_from_file_location("matched_report_under_test",SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def fixture(tmp_path, errors=None, *, classic_error=.4, surface=False):
    baseline=tmp_path/"previous_baselines"; baseline.mkdir()
    cohort=tmp_path/"cohort.nc"; cohort.write_bytes(b"fixed-raw-cohort")
    sha=report.sha256(cohort)
    levels=np.array([5,15,25,35,45,55,65,85,105,125,145,165.1,186.3,222.6,267.7,326.9,408.8,527.7,707.6,984.7])
    li=np.tile(np.arange(20),12)
    target=np.column_stack((np.sin(np.arange(240)/23),np.cos(np.arange(240)/31))).astype(np.float64)
    target[0]=np.nan
    canonical=report.canonical_target(target)
    mu=np.column_stack((np.linspace(.01,.02,20),np.linspace(-.01,.01,20)))
    sd=np.column_stack((np.linspace(.5,1,20),np.linspace(.05,.1,20)))
    reference={}
    exports={}
    for split in ("validation","development"):
        common={"month":np.repeat(np.arange(12)+(48 if split=="validation" else 60),20),
                "profile":np.arange(240)+(1000 if split=="development" else 0),"level":li,
                "target":target,"normalization_mean":mu,"normalization_std":sd,"levels":levels,
                "climatology_physical":np.column_stack((20-li/2,35+li/10))}
        reference[split]=common
        exports[split]={}
        for name,error in (("oi",classic_error),("nearest_profile",.8),("climatology",None)):
            mean=np.zeros_like(target) if error is None else np.nan_to_num(canonical).astype(float)+error
            path=baseline/f"{name}_{split}.npz"
            np.savez(path,**common,mean=mean,
                     metadata_json=json.dumps({"cohort_sha256":sha,"uncertainty_available":False}))
            exports[split][name]={"path":str(path),"sha256":report.sha256(path)}
    (baseline/"replay_manifest.json").write_text(json.dumps({"exports":exports,"cohort_sha256":sha}))
    fixed=tmp_path/"fixed_pointwise_mlp";fixed.mkdir()
    for split,common in reference.items():
        np.savez(fixed/f"{split}_seed1234.npz",**common,
                 mean=np.nan_to_num(canonical).astype(float)+.35,
                 baseline=np.nan_to_num(canonical).astype(float)+classic_error)
    (fixed/"model_seed1234.pt").write_bytes(b"fixed-MLP-checkpoint")
    (fixed/"summary_seed1234.json").write_text(json.dumps({"seed":1234,"completed_epochs":30,
        "source_sha256":{"54.py":"a"*64},"training_recipe":{"epochs":30,"lr":.001}}))
    prior=tmp_path/"prior_learned_oi";prior.mkdir()
    checkpoint=prior/"existing_model.pt";checkpoint.write_bytes(b"existing-learned-OI-checkpoint")
    oldsummary=prior/"historical_summary.json"
    oldsummary.write_text(json.dumps({"anomaly":"exact","steps":1500,"params":120,
        "first_guess":{"kind":"none"},"input_qc_z":None,
        "scores":{"validation":{"macro_z":.38}}}))
    prior_exports={}
    for split,common in reference.items():
        path=prior/f"{split}_seed1234.npz"
        np.savez(path,**common,mean=np.nan_to_num(canonical).astype(float)+.38,
                 baseline=np.nan_to_num(canonical).astype(float)+classic_error)
        prior_exports[split]={"path":str(path),"sha256":report.sha256(path)}
    (prior/"aux_registry.json").write_text(json.dumps({"parity_verified":True,
        "eligible_validation_candidate":True,"cohort_sha256":sha,"seed":1234,"training_steps":1500,
        "source_checkpoint":{"path":str(checkpoint),"sha256":report.sha256(checkpoint)},
        "historical_summary":{"path":str(oldsummary),"sha256":report.sha256(oldsummary)},"arrays":prior_exports}))
    jobs=[]
    families=("previous_token64","dense64","soft_moe192","soft_moe192_latent_off","soft_moe192_local_off")
    for mode in ([False,True] if surface else [False]):
        for fi,family in enumerate(families):
            for seed in report.SEEDS:
                tag=f"{family}_{'surface' if mode else 'argo'}_s{seed}"
                output=tmp_path/tag;output.mkdir()
                job={"tag":tag,"family":family,"seed":seed,"kind":"previous" if fi==0 else "latent",
                     "steps":15000,"output":str(output),"surface":mode,"command":["python","training.py"]}
                jobs.append(job)
                error=(errors or {}).get(family,.5 if fi==0 else .1+.1*(fi-1))
                if isinstance(error,tuple):
                    error=error[report.SEEDS.index(seed)]
                for split in ("validation","development"):
                    common=dict(reference[split])
                    # Mimic new46 float32 targets vs classical47 float64.
                    common["target"]=canonical if fi else target
                    mean=np.nan_to_num(canonical).astype(float)+error
                    payload={**common,"mean":mean,"baseline":np.nan_to_num(canonical).astype(float)+classic_error}
                    if fi:
                        payload["std"]=np.ones_like(mean)*.3
                    np.savez(output/f"{split}_seed{seed}.npz",**payload)
                score=float(abs(error))
                summary={"seed":seed,"best_step":1000,"params":100+fi,"config":{"variant":"dense",
                         "use_latent":not family.endswith("latent_off"),"use_local":not family.endswith("local_off"),
                         "n_sat_features":56 if mode else 0},
                         "training":{"steps":15000,"context_profiles":6080,"freeze_analysis":True},
                         "history":[{"step":15000}],"source_sha256":{"training.py":"a"*64},
                         "data":{"cohort_fingerprint":"fixed-cohort","surface":mode},
                         "validation":{"scores":{"macro_z":score}},
                         "development":{"scores":{"macro_z":0 if fi==2 else 100}}}
                (output/f"summary_seed{seed}.json").write_text(json.dumps(summary))
                (output/f"best_seed{seed}.pt").write_bytes(b"frozen-checkpoint"+str(seed).encode())
    campaign=tmp_path/"campaign.json";campaign.write_text(json.dumps({"jobs":jobs}))
    return campaign,baseline,cohort,{"validation":239,"development":239},jobs


def select(parts,destination):
    campaign,baseline,cohort,counts,_=parts
    return report.freeze_selection(campaign,baseline,destination,cohort_path=cohort,expected_counts=counts)


def overwrite(path,mutate):
    with np.load(path) as data:
        a={name:data[name] for name in data.files}
    mutate(a)
    np.savez(path,**a)


def test_freeze_reads_validation_only_and_ignores_development_champion(tmp_path,monkeypatch):
    parts=fixture(tmp_path)
    original=np.load;opened=[]
    def validation_only(path,*args,**kwargs):
        opened.append(str(path))
        assert "development" not in str(path)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(np,"load",validation_only)
    selected=select(parts,tmp_path/"selection.json")
    assert selected["selected"]["argo"]["family"]=="dense64"
    assert opened and all("validation" in path for path in opened)


def test_classical_oi_can_win_and_surface_choice_is_separate(tmp_path):
    parts=fixture(tmp_path,classic_error=.01,surface=True)
    selected=select(parts,tmp_path/"selection.json")
    assert selected["selected"]["argo"]["family"]=="oi"
    assert selected["selected"]["surface"]["family"]=="oi"
    assert selected["selected"]["surface"]["kind"]=="classical"


def test_ensemble_mean_is_selected_rather_than_mean_seed_error(tmp_path):
    parts=fixture(tmp_path,{"soft_moe192":(-.3,.3,0.)})
    selected=select(parts,tmp_path/"selection.json")
    assert selected["selected"]["argo"]["family"]=="soft_moe192"
    assert selected["selected"]["argo"]["validation_mean_standardized_rmse"]<1e-7


@pytest.mark.parametrize("key",["month","profile","level","target","normalization_std","normalization_mean"])
def test_any_scoring_or_normalization_change_refuses_selection(tmp_path,key):
    parts=fixture(tmp_path)
    job=parts[-1][4]
    path=Path(job["output"])/f"validation_seed{job['seed']}.npz"
    overwrite(path,lambda a:a[key].__setitem__((1,) if a[key].ndim==1 else (1,0),a[key][1] + 1 if a[key].ndim==1 else a[key][1,0]+1))
    with pytest.raises(ValueError,match="identity|target|normalization|depth"):
        select(parts,tmp_path/"selection.json")
    assert not (tmp_path/"selection.json").exists()


def test_float32_and_float64_targets_share_canonical_identity(tmp_path):
    parts=fixture(tmp_path)
    selected=select(parts,tmp_path/"selection.json")
    assert len(selected["runs"])==15


def test_nominal_steps_cannot_disguise_incomplete_run(tmp_path):
    parts=fixture(tmp_path)
    job=parts[-1][0];path=Path(job["output"])/f"summary_seed{job['seed']}.json"
    a=json.loads(path.read_text());a["history"]=[{"step":14000}];path.write_text(json.dumps(a))
    with pytest.raises(report.PendingReport,match="14000/15000"):
        select(parts,tmp_path/"selection.json")


def test_raw_cohort_sha_is_verified_before_selection(tmp_path):
    parts=fixture(tmp_path)
    parts[2].write_bytes(b"different-raw-cohort")
    with pytest.raises(ValueError,match="cohort"):
        select(parts,tmp_path/"selection.json")


def test_frozen_validation_prediction_change_is_rejected_before_development_read(tmp_path,monkeypatch):
    parts=fixture(tmp_path);selection=tmp_path/"selection.json";select(parts,selection)
    job=parts[-1][0];path=Path(job["output"])/f"validation_seed{job['seed']}.npz"
    overwrite(path,lambda a:a["mean"].__iadd__(.01))
    original=np.load
    def refuse_development(path,*args,**kwargs):
        assert "development" not in str(path)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(np,"load",refuse_development)
    with pytest.raises(ValueError,match="stored validation|frozen"):
        report.build_report(parts[0],parts[1],selection,bootstrap_draws=30)


def test_report_calibrates_oi_on_validation_only_and_exports_physical_metrics(tmp_path):
    parts=fixture(tmp_path,surface=True);selection=tmp_path/"selection.json";select(parts,selection)
    result=report.build_report(parts[0],parts[1],selection,bootstrap_draws=30)
    oi=next(row for row in result["rows"] if row["family"]=="oi")
    assert np.allclose(result["oi_calibration_std_z"],.4)
    assert "2004" in oi["uncertainty_method"]
    assert oi["physical_metrics"]["TEMP"]["n"]==239
    assert oi["physical_metrics"]["TEMP"]["mean_std"]<.4
    assert oi["physical_metrics"]["TEMP"]["absolute_r2"] is not None
    row=next(row for row in result["rows"] if row["family"]=="dense64" and row["mode"]=="argo")
    assert row["ensemble"]["physical_metrics"]["TEMP"]["rmse"]<oi["physical_metrics"]["TEMP"]["rmse"]
    assert row["seed_mean_sd"]["TEMP"]["rmse"]["sd"]==pytest.approx(0.)
    assert result["contrasts"]["argo:dense64"]["against_previous_token64"]["TEMP"]["delta_rmse"]<0
    assert result["component_contrasts"]["argo:full_minus_local_off"]["TEMP"]["delta_rmse"]<0
    assert "dense64" in result["multimodal_contrasts"]
    text=report.chinese_report(result)
    assert "RMSE" in text and "MAE" in text and "J score" not in text
    assert "三种子均值 ±" in text and "三种子集成" in text
    assert "不是精确 OI 后验" in text
    json.dumps(result,allow_nan=False)


def test_existing_frozen_choice_survives_summary_development_extension(tmp_path):
    parts=fixture(tmp_path);selection=tmp_path/"selection.json";first=select(parts,selection)
    for job in parts[-1]:
        path=Path(job["output"])/f"summary_seed{job['seed']}.json"
        summary=json.loads(path.read_text());summary["development"]={"new":"evaluation"}
        path.write_text(json.dumps(summary))
    second=select(parts,selection)
    assert first==second


def test_missing_freeze_refuses_report_before_opening_any_arrays(tmp_path,monkeypatch):
    monkeypatch.setattr(np,"load",lambda *args,**kwargs:pytest.fail("must not open arrays"))
    with pytest.raises(report.PendingReport,match="freeze"):
        report.build_report(tmp_path/"campaign",tmp_path/"baseline",tmp_path/"missing")


def test_classical_deterministic_rows_do_not_invent_uncertainty(tmp_path):
    parts=fixture(tmp_path);selection=tmp_path/"selection.json";select(parts,selection)
    result=report.build_report(parts[0],parts[1],selection,bootstrap_draws=10)
    nearest=next(row for row in result["rows"] if row["family"]=="nearest_profile")
    assert "nll" not in nearest["physical_metrics"]["TEMP"]
    assert nearest["uncertainty_method"]=="unavailable"


def test_fixed_pointwise_mlp_must_finish_before_freeze_and_is_not_selection_candidate(tmp_path):
    parts=fixture(tmp_path)
    path=tmp_path/"fixed_pointwise_mlp/summary_seed1234.json"
    summary=json.loads(path.read_text());summary["completed_epochs"]=29;path.write_text(json.dumps(summary))
    with pytest.raises(report.PendingReport,match="30 epochs"):
        select(parts,tmp_path/"selection.json")
    summary["completed_epochs"]=30;path.write_text(json.dumps(summary))
    selected=select(parts,tmp_path/"selection.json")
    assert selected["auxiliary_fixed_mlp"]["included_in_architecture_selection"] is False
    assert all(item["family"]!="fixed_pointwise_mlp" for item in selected["candidates"]["argo"])


def test_metrics_use_original_float64_truth_after_float32_identity_check(tmp_path):
    parts=fixture(tmp_path)
    ref=report.load_reference(parts[1],"validation",expected_counts=parts[3])
    assert ref["target"].dtype==np.float64 and ref["target32"].dtype==np.float32
    assert np.any(ref["target"][1:]!=ref["target32"][1:].astype(np.float64))
    prediction={"mean":np.nan_to_num(ref["target32"]).astype(float)}
    metrics=report.physical_metrics(prediction,ref)
    assert metrics["physical_metrics"]["TEMP"]["rmse"]>0
    assert metrics["physical_metrics"]["TEMP"]["rmse"]<1e-7


def test_frozen_fixed_mlp_validation_change_is_rejected(tmp_path):
    parts=fixture(tmp_path);selection=tmp_path/"selection.json";select(parts,selection)
    overwrite(tmp_path/"fixed_pointwise_mlp/validation_seed1234.npz",lambda a:a["mean"].__iadd__(.01))
    with pytest.raises(ValueError,match="fixed pointwise MLP changed"):
        select(parts,selection)


def test_hidden_oi_retraining_or_latent_context_reduction_is_rejected(tmp_path):
    parts=fixture(tmp_path)
    job=next(job for job in parts[-1] if job["kind"]=="latent")
    path=Path(job["output"])/f"summary_seed{job['seed']}.json"
    summary=json.loads(path.read_text());summary["training"]["freeze_analysis"]=False
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError,match="OI must remain frozen"):
        select(parts,tmp_path/"selection.json")
    summary["training"]["freeze_analysis"]=True;summary["training"]["context_profiles"]=768
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError,match="complete source pool"):
        select(parts,tmp_path/"selection.json")


def test_stronger_existing_learned_oi_can_win_without_new_training(tmp_path):
    errors={name:.6 for name in ("previous_token64","dense64","soft_moe192","soft_moe192_latent_off","soft_moe192_local_off")}
    parts=fixture(tmp_path,errors)
    selection=tmp_path/"selection.json";chosen=select(parts,selection)
    assert chosen["selected"]["argo"]["family"]=="prior_learned_oi"
    assert chosen["selected"]["argo"]["training_steps"]==1500
    result=report.build_report(parts[0],parts[1],selection,bootstrap_draws=10)
    assert result["selected_against_previous_best"]["argo"]["selected_family"]=="prior_learned_oi"
    assert result["selected_against_previous_best"]["argo"]["against_prior_learned_oi"]["TEMP"]["delta_rmse"]==0
    assert "保留既有 learned OI" in report.chinese_report(result)


def test_missing_or_unverified_strong_prior_is_not_silently_omitted(tmp_path):
    parts=fixture(tmp_path)
    path=tmp_path/"prior_learned_oi/aux_registry.json"
    stored=json.loads(path.read_text());path.unlink()
    with pytest.raises(report.PendingReport,match="prior learned-OI"):
        select(parts,tmp_path/"selection.json")
    stored["parity_verified"]=False;path.write_text(json.dumps(stored))
    with pytest.raises(ValueError,match="strongest earlier comparator"):
        select(parts,tmp_path/"selection.json")
