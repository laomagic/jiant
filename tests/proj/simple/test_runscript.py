import os
import pytest

import jiant.utils.python.io as py_io
from jiant.proj.simple import runscript as run
import jiant.scripts.download_data.runscript as downloader


@pytest.mark.parametrize("task_name", ["copa"])
@pytest.mark.parametrize("model_type", ["bert-base-cased"])
def test_simple_runscript(tmpdir, task_name, model_type):
    RUN_NAME = f"{test_simple_runscript.__name__}_{task_name}_{model_type}"
    data_dir = str(tmpdir.mkdir("data"))
    exp_dir = str(tmpdir.mkdir("exp"))

    downloader.download_data([task_name], data_dir)

    args = run.RunConfiguration(
        run_name=RUN_NAME,
        exp_dir=exp_dir,
        data_dir=data_dir,
        model_type=model_type,
        tasks=task_name,
        train_examples_cap=16,
        train_batch_size=16,
        no_cuda=True,
    )
    run.run_simple(args)

    val_metrics = py_io.read_json(os.path.join(exp_dir, "runs", RUN_NAME, "val_metrics.json"))
    assert val_metrics["aggregated"] > 0


@pytest.mark.gpu
@pytest.mark.parametrize("task_name", ["ccg"])
@pytest.mark.parametrize("model_type", ["roberta-large"])
def test_simple_runscript_save(tmpdir, task_name, model_type):
    run_name = f"{test_simple_runscript.__name__}_{task_name}_{model_type}_save"
    data_dir = str(tmpdir.mkdir("data"))
    exp_dir = str(tmpdir.mkdir("exp"))

    downloader.download_data([task_name], data_dir)

    args = run.RunConfiguration(
        run_name=RUN_NAME,
        exp_dir=exp_dir,
        data_dir=data_dir,
        model_type=model_type,
        tasks=task_name,
        train_examples_cap=16,
        train_batch_size=16,
        do_save=True
    )
    run.run_simple(args)

    # check output best model and last model
    assert os.path.exists(os.path.join(exp_dir, "runs", RUN_NAME, "best_model.p"))
    assert os.path.exists(os.path.join(exp_dir, "runs", RUN_NAME, "best_model.metadata.json"))
    assert os.path.exists(os.path.join(exp_dir, "runs", RUN_NAME, "last_model.p"))
    assert os.path.exists(os.path.join(exp_dir, "runs", RUN_NAME, "last_model.metadata.json"))

    # assert best_model not equal to last_model
    best_model_weights = torch.load(os.path.join(exp_dir, "runs", RUN_NAME, "best_model.p"))
    last_model_weights = torch.load(os.path.join(exp_dir, "runs", RUN_NAME, "last_model.p"))
    assert best_model_weights != last_model_weights

    run_name = f"{test_simple_runscript.__name__}_{task_name}_{model_type}_save_best"
    args = run.RunConfiguration(
        run_name=RUN_NAME,
        exp_dir=exp_dir,
        data_dir=data_dir,
        model_type=model_type,
        tasks=task_name,
        train_examples_cap=16,
        train_batch_size=16,
        do_save_best=True
    )
    run.run_simple(args)

    # check output best model
    assert os.path.exists(os.path.join(exp_dir, "runs", RUN_NAME, "best_model.p"))
    assert os.path.exists(os.path.join(exp_dir, "runs", RUN_NAME, "best_model.metadata.json"))
    assert not os.path.exists(os.path.join(exp_dir, "runs", RUN_NAME, "last_model.p"))
    assert not os.path.exists(os.path.join(exp_dir, "runs", RUN_NAME, "last_model.metadata.json"))

    # check output last model
    run_name = f"{test_simple_runscript.__name__}_{task_name}_{model_type}_save_last"
    args = run.RunConfiguration(
        run_name=RUN_NAME,
        exp_dir=exp_dir,
        data_dir=data_dir,
        model_type=model_type,
        tasks=task_name,
        train_examples_cap=16,
        train_batch_size=16,
        do_save_last=True
    )
    run.run_simple(args)

    # check output best model
    assert not os.path.exists(os.path.join(exp_dir, "runs", RUN_NAME, "best_model.p"))
    assert not os.path.exists(os.path.join(exp_dir, "runs", RUN_NAME, "best_model.metadata.json"))
    assert os.path.exists(os.path.join(exp_dir, "runs", RUN_NAME, "last_model.p"))
    assert os.path.exists(os.path.join(exp_dir, "runs", RUN_NAME, "last_model.metadata.json"))
