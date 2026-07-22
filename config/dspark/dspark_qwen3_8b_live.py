import os

from deepspec.trainer.live_dspark_trainer import LiveQwen3DSparkTrainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR, QWEN_3_8B


project_name = "deepspec-live"
exp_name = "dspark_block7_qwen3_8b_live"
seed = 42

model = dict(
    target_model_name_or_path=QWEN_3_8B,
    target_revision="b968826d9c46dd6066d109eabc6255188de91218",
    init_draft_name_or_path="deepseek-ai/dspark_qwen3_8b_block7",
    init_draft_revision="03326e5043815da1f81b109078b2889737c26017",
    block_size=7,
    num_draft_layers=5,
    target_layer_ids=[1, 9, 17, 25, 33],
    mask_token_id=151669,
    num_anchors=512,
    markov_rank=256,
    markov_head_type="vanilla",
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,
    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,
)

train = dict(
    trainer_cls=LiveQwen3DSparkTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=32,
    num_train_epochs=3,
    max_train_steps=150,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=True,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=50,
)

data = dict(
    source_manifest_path=None,
    producer_manifests_path=None,
    ledger_dir=None,
    vllm_endpoint=None,
    vllm_model=QWEN_3_8B,
    selection_policy_digest=None,
    run_id=None,
    consumer_boot_id=None,
    request_timeout_s=60.0,
    max_retries=2,
    num_workers=4,
    parity_cache_path=None,
    hidden_cosine_min=0.999,
    hidden_relative_rmse_max=0.01,
    loss_symmetric_relative_error_max=0.01,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    logging_cfg["checkpoint_dir"] = os.path.join(
        BASE_CKPT_DIR,
        str(cfg["project_name"]),
        str(cfg["exp_name"]),
    )
    logging_cfg["tensorboard_dir"] = os.path.join(
        BASE_TB_DIR,
        str(cfg["project_name"]),
        str(cfg["exp_name"]),
    )
    cfg["logging"] = logging_cfg
    return cfg
