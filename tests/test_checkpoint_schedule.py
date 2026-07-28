from contextlib import nullcontext
import importlib
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest
import deepspec

try:
    import tensorboard  # noqa: F401
except ModuleNotFoundError:
    tensorboard_module = types.ModuleType("torch.utils.tensorboard")
    tensorboard_module.SummaryWriter = object
    sys.modules["torch.utils.tensorboard"] = tensorboard_module

trainer_package = types.ModuleType("deepspec.trainer")
trainer_package.__path__ = [
    str(Path(deepspec.__file__).resolve().parent / "trainer")
]
sys.modules["deepspec.trainer"] = trainer_package
base_trainer = importlib.import_module("deepspec.trainer.base_trainer")


class _FakeLoss:
    def __truediv__(self, _divisor):
        return self

    def backward(self):
        return None


class _FakeModel:
    def train(self):
        return None

    def no_sync(self):
        return nullcontext()


class _FakeOptimizer:
    def step(self):
        return None

    def get_learning_rate(self):
        return 1e-4


class _FakeSuspendController:
    def monitoring(self):
        return nullcontext()

    def requested(self):
        return False


class _FakeGradNorm:
    def item(self):
        return 1.0


@pytest.mark.parametrize(
    ("max_train_steps", "checkpointing_steps", "expected_saved_steps"),
    [
        (2, 1, [1, 2]),
        (3, 2, [2, 3]),
    ],
)
def test_train_does_not_resave_periodic_checkpoint_at_final_step(
    monkeypatch,
    max_train_steps,
    checkpointing_steps,
    expected_saved_steps,
):
    trainer = object.__new__(base_trainer.BaseTrainer)
    trainer.model = _FakeModel()
    trainer.optimizer = _FakeOptimizer()
    trainer.suspend_controller = _FakeSuspendController()
    trainer.device = "cuda"
    trainer.next_micro_step = 0
    trainer.gradient_accumulation_steps = 1
    trainer.micro_batches_per_epoch = max_train_steps
    trainer.max_train_steps = max_train_steps
    trainer.args = SimpleNamespace(
        train=SimpleNamespace(local_batch_size=1, max_grad_norm=1.0),
        logging=SimpleNamespace(checkpointing_steps=checkpointing_steps),
    )
    trainer._build_train_dataloader = lambda **_kwargs: [
        object()
        for _ in range(max_train_steps)
    ]
    trainer.run_batch = lambda _batch: _FakeLoss()

    saved_steps = []
    trainer.save_and_eval_checkpoint = lambda: saved_steps.append(
        trainer.global_step
    )

    monkeypatch.setattr(
        base_trainer,
        "CUDAPrefetcher",
        lambda dataloader, _device: dataloader,
    )
    monkeypatch.setattr(
        base_trainer.FSDP,
        "clip_grad_norm_",
        lambda _model, _max_norm: _FakeGradNorm(),
    )
    monkeypatch.setattr(
        base_trainer.training_logger,
        "start_session",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        base_trainer.training_logger,
        "on_optimizer_step",
        lambda **_kwargs: None,
    )

    trainer.train()

    assert saved_steps == expected_saved_steps
