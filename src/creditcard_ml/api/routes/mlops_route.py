import json
import subprocess
import glob
import os
from fastapi import APIRouter

router = APIRouter(prefix="/mlops", tags=["MLOps"])

def find_dvc_bin():
    base_path = "/root/.cache/pypoetry/virtualenvs"
    matches = glob.glob(f"{base_path}/*/bin/dvc")

    if not matches:
        return "dvc"

    return matches[0]


DVC_BIN = "/root/.cache/pypoetry/virtualenvs/creditcard-ml-9TtSrW0h-py3.10/bin/dvc"
print(f"[MLOPS] Using DVC: {DVC_BIN}")

def run(cmd: str):
    process = subprocess.Popen(
        f"{DVC_BIN} {cmd}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        text=True
    )
    out, err = process.communicate()
    return {"output": out, "error": err}


# --- ROTAS MLOPS ---
@router.post("/download")
def mlops_download():
    return run("repro --force download")


@router.post("/preprocess")
def mlops_preprocess():
    return run("repro --force preprocess")


@router.post("/train")
def mlops_train():
    return run("repro --force train")


@router.post("/evaluate")
def mlops_evaluate():
    return run("repro --force evaluate")


@router.post("/retrain")
def mlops_retrain():
    """Pipeline completo"""
    return run("repro --force")


@router.get("/metrics")
def mlops_metrics():
    with open("metrics.json") as f:
        return json.load(f)


@router.get("/eval")
def mlops_eval():
    with open("eval.json") as f:
        return json.load(f)


@router.get("/model/info")
def mlops_model_info():
    path = "src/creditcard_ml/model/model.pkl"
    size = os.path.getsize(path)

    return {
        "path": path,
        "size_kb": round(size / 1024, 2)
    }


@router.get("/pipeline")
def mlops_pipeline():
    return run("dag")
