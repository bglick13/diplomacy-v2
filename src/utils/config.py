# src/utils/config.py
"""
Centralized configuration for Diplomacy GRPO experiments.

This module provides:
- ExperimentConfig: Pydantic model for all training/experiment settings
- EvalConfig: Pydantic model for evaluation settings
- add_config_args: Helper to auto-generate argparse arguments from Pydantic models
- config_from_args: Helper to create config from parsed arguments
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any, Literal, get_args, get_origin

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    pass

ProfilingMode = Literal["rollout", "trainer", "e2e"]


class ExperimentConfig(BaseModel):
    """
    Global configuration for a GRPO training run.
    Passed between Trainer, Inference, and Rollout workers.

    All fields have sensible defaults and can be overridden via CLI arguments.
    Use `add_config_args()` to auto-generate argparse arguments from this model.
    """

    # =========================================================================
    # Experiment Metadata
    # =========================================================================
    run_name: str = Field(
        default="diplomacy-grpo-v1",
        description="Name for this training run (used in WandB and checkpoints)",
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")
    experiment_tag: str | None = Field(
        default=None,
        description="Tag for grouping related runs in WandB (e.g., 'power-laws')",
    )
    wandb_project: str = Field(default="diplomacy-grpo", description="WandB project name")

    # =========================================================================
    # Loss Type (GRPO vs GSPO)
    # =========================================================================
    loss_type: Literal["grpo", "gspo"] = Field(
        default="grpo",
        description=(
            "Loss type for policy optimization. "
            "'grpo' uses token-level importance sampling (standard). "
            "'gspo' uses sequence-level importance sampling (geometric mean of token ratios)."
        ),
    )
    min_reward_variance: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Minimum reward variance within a group to include in training (DAPO-style dynamic sampling). "
            "Groups with lower variance are rejected (no gradient signal). "
            "Set to 0.0 to disable (use advantage_min_std instead). "
            "Recommended: 0.01 for GSPO."
        ),
    )

    # =========================================================================
    # Model Settings
    # =========================================================================
    base_model_id: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct",
        description="HuggingFace model ID for base model",
    )
    lora_rank: int = Field(
        default=32, description="LoRA adapter rank. Higher = more capacity. 16-64 typical for RLHF."
    )
    lora_alpha: int | None = Field(
        default=None,
        description=(
            "LoRA alpha (scaling factor). If None, defaults to 2 * lora_rank. "
            "Higher alpha = stronger LoRA effect. Typical: 1-2x rank."
        ),
    )
    lora_target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",  # Attention
            "gate_proj",
            "up_proj",
            "down_proj",  # MLP
        ],
        description=(
            "Which modules to apply LoRA to. Default includes attention + MLP layers. "
            "Attention only: ['q_proj', 'k_proj', 'v_proj', 'o_proj']. "
            "More modules = more trainable params but better capacity."
        ),
    )
    lora_dropout: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Dropout probability for LoRA layers. 0.0-0.1 typical.",
    )

    # =========================================================================
    # Rollout/Environment Settings
    # =========================================================================
    rollout_horizon_years: int = Field(
        default=4, description="Default/short rollout horizon in game years"
    )
    rollout_long_horizon_years: int = Field(
        default=6, description="Long horizon for a subset of rollouts"
    )
    rollout_long_horizon_chance: float = Field(
        default=0.2,
        description="Probability of using long horizon (0.0-1.0). 0.2 = 20% long, 80% short",
    )
    rollout_visualize_chance: float = Field(
        default=0.0,
        description="Probability of generating visualization for a rollout (0.0-1.0)",
    )
    trajectory_sample_rate: float = Field(
        default=0.01,
        description="Fraction of trajectories to log to Weave for debugging (0.01 = 1%)",
    )
    rollout_no_warmup_chance: float = Field(
        default=0.2,
        description="Probability of skipping warmup phase in rollout (0.0-1.0)",
    )

    # =========================================================================
    # Reward / Scoring Settings
    # =========================================================================
    win_bonus: float = Field(
        default=50.0,
        description=(
            "Bonus reward for the sole leader when they have >= winner_threshold_sc supply centers. "
            "Creates pressure to WIN outright, breaking cooperative stalemate equilibria. "
            "Note: Scaled by final_reward_weight (0.2), so 50 * 0.2 = 10 effective bonus."
        ),
    )
    winner_threshold_sc: int = Field(
        default=5,
        description="Minimum supply centers required to be eligible for win bonus",
    )
    solo_victory_sc: int = Field(
        default=18,
        description="Supply centers required for solo victory (full win bonus)",
    )
    use_position_based_scoring: bool = Field(
        default=True,
        description=(
            "Use position-based final scoring with bonuses for each rank (1st-7th). "
            "When enabled, all finishing positions receive appropriate bonuses/penalties, "
            "not just the winner. Inspired by webDiplomacy scoring systems."
        ),
    )
    position_bonus_1st: float = Field(
        default=50.0,
        description="Bonus for 1st place (leader or tied for lead)",
    )
    position_bonus_2nd: float = Field(
        default=25.0,
        description="Bonus for 2nd place",
    )
    position_bonus_3rd: float = Field(
        default=15.0,
        description="Bonus for 3rd place",
    )
    position_bonus_4th: float = Field(
        default=10.0,
        description="Bonus for 4th place",
    )
    position_bonus_5th: float = Field(
        default=5.0,
        description="Bonus for 5th place",
    )
    position_bonus_6th: float = Field(
        default=2.0,
        description="Bonus for 6th place",
    )
    position_bonus_7th: float = Field(
        default=0.0,
        description="Bonus for 7th place (surviving but last)",
    )
    elimination_penalty: float = Field(
        default=-30.0,
        description="Penalty for being eliminated (0 SCs, 0 units)",
    )
    step_reward_weight: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for per-step reward shaping (board state delta). "
            "Higher values emphasize immediate consequences of each decision."
        ),
    )
    final_reward_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for final game outcome in reward. "
            "Higher values emphasize winning the overall game vs immediate gains."
        ),
    )
    step_dislodgment_weight: float = Field(
        default=0.5,
        ge=0.0,
        description="Weight for dislodgment signals in step scoring (+/- per dislodged unit)",
    )
    step_territory_weight: float = Field(
        default=0.2,
        ge=0.0,
        description="Weight for territory expansion in step scoring (+ per new province)",
    )
    step_threat_weight: float = Field(
        default=0.3,
        ge=0.0,
        description="Weight for SC threat penalty in step scoring (- per threatened SC)",
    )
    step_forward_weight: float = Field(
        default=0.1,
        ge=0.0,
        description="Weight for forward unit positioning in step scoring (+ per unit outside home)",
    )
    reward_discount_gamma: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Discount factor for future step rewards (gamma). "
            "Each step t receives discounted sum of future deltas: Σ γ^(k-t) × delta_k. "
            "gamma=1.0 means no discounting (all future steps weighted equally). "
            "gamma=0.95 gives ~35% weight to rewards 10 steps ahead. "
            "gamma=0.0 means only immediate step delta (no temporal credit). "
            "WARNING: gamma > 0 creates correlated gradients that can destabilize training."
        ),
    )
    use_trajectory_level_rewards: bool = Field(
        default=False,
        description=(
            "If True, emit ONE sample per trajectory (using first step's prompt/completion) "
            "with final score as reward. Groups by trajectory instead of by step. "
            "This provides sparse but theoretically sound temporal credit assignment: "
            "GRPO compares entire trajectories ('which fork's strategy was better?') "
            "rather than individual steps. Avoids the gradient correlation issue of gamma > 0."
        ),
    )

    # Strategic awareness reward shaping
    leader_gap_penalty_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for penalizing large gaps to the leader. "
            "Encourages stopping runaway leaders instead of fighting weak neighbors."
        ),
    )
    balance_bonus_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for balance-of-power bonus. "
            "Non-leaders get rewarded when game is balanced (encourages coalition behavior)."
        ),
    )
    leader_gap_threshold: int = Field(
        default=3,
        ge=1,
        description="SC gap above which leader_gap_penalty starts applying.",
    )
    use_strategic_step_scoring: bool = Field(
        default=False,
        description=(
            "Use strategic step scoring (position-based) instead of SC-based. "
            "Rewards relative position and balance, not absolute SC accumulation. "
            "When enabled, leader_gap_penalty and balance_bonus are integrated into "
            "step scoring rather than applied as separate shaping."
        ),
    )

    # =========================================================================
    # League Training Settings
    # =========================================================================
    # NOTE: League training collects trajectories only for the "hero" power,
    # reducing trajectories per rollout by 7x compared to self-play.
    #
    # Recommended compensations:
    #   - Increase num_groups_per_step from 8 to 48-56 (6-7x more rollouts)
    #   - OR lower learning_rate from 1e-5 to 2e-6 (5x lower)
    #   - OR increase samples_per_group from 8 to 24 (3x more forks)
    # =========================================================================
    league_training: bool = Field(
        default=True,
        description="Enable league training with PFSP opponent sampling",
    )
    league_registry_path: str | None = Field(
        default=None,
        description="Path to league registry JSON file. Defaults to /data/league_{run_name}.json",
    )
    league_inherit_from: str | None = Field(
        default=None,
        description="Run name to inherit league opponents from (for curriculum). Copies checkpoints as opponents.",
    )
    checkpoint_every_n_steps: int = Field(
        default=10,
        description="Save checkpoint to league every N steps (for recent curriculum)",
    )
    # =========================================================================
    # TrueSkill Rating Settings
    # =========================================================================
    trueskill_mu_init: float = Field(
        default=25.0,
        description="Initial TrueSkill mu (skill mean). Standard default is 25.",
    )
    trueskill_sigma_init: float = Field(
        default=8.333,
        description="Initial TrueSkill sigma (uncertainty). Standard default is 25/3 ≈ 8.333.",
    )
    trueskill_beta: float = Field(
        default=4.166,
        description="TrueSkill beta (performance variance). Standard is sigma/2 ≈ 4.166.",
    )
    trueskill_tau: float = Field(
        default=0.0833,
        description="TrueSkill tau (skill drift per game). Standard is sigma/100 ≈ 0.0833.",
    )

    # =========================================================================
    # League Evaluation Settings (updates TrueSkill ratings)
    # =========================================================================
    league_eval_every_n_steps: int = Field(
        default=0,
        description="Run async league evaluation every N steps for rating updates (0 to disable). "
        "Primary TrueSkill updates come from rollouts; this is for additional validation.",
    )
    league_eval_games_per_opponent: int = Field(
        default=2,
        description="Games per league opponent during evaluation",
    )

    # =========================================================================
    # Benchmark Evaluation Settings (frozen checkpoints, absolute skill)
    # =========================================================================
    benchmark_eval_every_n_steps: int = Field(
        default=50,
        description="Run benchmark evaluation against frozen checkpoints every N steps (0 to disable).",
    )
    benchmark_games_per_opponent: int = Field(
        default=3,
        description="Number of games to play against each frozen benchmark checkpoint.",
    )
    benchmark_max_years: int = Field(
        default=5,
        description="Maximum game length in years for benchmark evaluation games.",
    )
    pfsp_self_play_weight: float = Field(
        default=0.30,
        description="PFSP: Weight for self-play (current policy)",
    )
    pfsp_peer_weight: float = Field(
        default=0.30,
        description="PFSP: Weight for peer opponents (similar Elo)",
    )
    pfsp_exploitable_weight: float = Field(
        default=0.35,
        description="PFSP: Weight for exploitable opponents (weaker)",
    )
    pfsp_baseline_weight: float = Field(
        default=0.05,
        description="PFSP: Weight for baseline opponents (bots)",
    )
    dumbbot_game_probability: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description=(
            "Probability that a rollout uses all DumbBot opponents (1v6 benchmark). "
            "These games measure absolute skill vs a fixed baseline, similar to DipNet evaluation. "
            "Win rate against 6 DumbBots is tracked separately in WandB. 0.0 disables."
        ),
    )

    # =========================================================================
    # Checkpointing & Resume
    # =========================================================================
    save_state_every_n_steps: int = Field(
        default=10,
        description="Save full training state (optimizer, step) every N steps for resume capability",
    )
    resume_from_run: str | None = Field(
        default=None,
        description="Run name to resume from (e.g., 'grpo-20251206-133229'). Will load latest checkpoint.",
    )
    resume_from_step: int | None = Field(
        default=None,
        description="Specific step to resume from (defaults to latest if not specified)",
    )
    disable_auto_resume: bool = Field(
        default=False,
        description="Disable auto-resume from crash. If True, always starts fresh even if checkpoints exist.",
    )

    # =========================================================================
    # Training Loop
    # =========================================================================
    total_steps: int = Field(
        default=200,
        description=(
            "Total number of training steps. "
            "Empirically, Elo gains plateau around step 100-150 with LR=1e-5. "
            "Recommendations: 100 for sweeps/ablations, 200 for standard runs, "
            "300+ for best quality (diminishing returns but more lottery tickets)."
        ),
    )
    num_groups_per_step: int = Field(
        default=16,
        description="Number of rollout groups per step (G in GRPO). I.e., how many different game states to start from",
    )
    samples_per_group: int = Field(
        default=8,
        description="Number of trajectory samples per group (N in GRPO). I.e., how many different forked games to simulate for each starting state",
    )
    buffer_depth: int = Field(
        default=3,
        description="Number of rollout batches to keep in flight ahead of trainer (1-2 recommended after fork sync)",
    )
    max_concurrent_containers: int = Field(
        default=100,
        description=(
            "Maximum number of concurrent Modal containers for rollouts. "
            "Modal has hard limits (~250-500 depending on plan). "
            "Set this below the limit to leave headroom for inference containers."
        ),
    )

    # =========================================================================
    # Optimizer Settings
    # =========================================================================
    learning_rate: float = Field(
        default=1e-5,
        description=(
            "Learning rate for AdamW optimizer. "
            "For GRPO with LoRA, 1e-5 to 2e-5 is typical. "
            "Scale up ~linearly with batch size (e.g., 2x batch → 2x LR)."
        ),
    )
    max_grad_norm: float = Field(
        default=10.0,
        description=(
            "Maximum gradient norm for clipping. "
            "10.0 recommended for GRPO (allows larger updates than default 5.0)."
        ),
    )
    chunk_size: int = Field(
        default=4,
        ge=1,
        description=(
            "Mini-batch size for gradient accumulation (must be >= 1). "
            "Reduced from 8 to 4 for Qwen2.5-7B's large vocab (~150k) to avoid OOM."
        ),
    )

    # =========================================================================
    # KL Penalty / GRPO Stability Settings
    # =========================================================================
    kl_beta: float = Field(
        default=0.0,
        description="KL penalty coefficient. Set to 0 to disable KL regularization entirely.",
    )
    kl_beta_warmup_steps: int = Field(
        default=20,
        ge=0,
        description=(
            "Number of steps to linearly warmup KL beta from 0 to kl_beta. "
            "Allows policy to explore freely in early training before constraining."
        ),
    )
    kl_target: float | None = Field(
        default=None,
        description=(
            "Target KL divergence for adaptive KL control. If set, beta is adjusted "
            "each step to achieve this target. Typical values: 0.01-0.1. "
            "None disables adaptive KL (uses fixed/warmup beta)."
        ),
    )
    kl_horizon: int = Field(
        default=10,
        ge=1,
        description="Number of steps over which to smooth KL for adaptive control.",
    )
    kl_beta_min: float = Field(
        default=0.001,
        gt=0,
        description="Minimum KL beta when using adaptive control.",
    )
    kl_beta_max: float = Field(
        default=0.5,
        gt=0,
        description="Maximum KL beta when using adaptive control.",
    )

    # =========================================================================
    # PPO Clipping Settings (DAPO-style asymmetric clipping)
    # =========================================================================
    use_ppo_clipping: bool = Field(
        default=True,
        description=(
            "Enable PPO-style ratio clipping. When False, uses vanilla REINFORCE. "
            "Clipping prevents large policy updates that could destabilize training."
        ),
    )
    ppo_epsilon_low: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description=(
            "Lower bound for PPO ratio clipping: ratio >= 1 - epsilon_low. "
            "Limits how much we can decrease an action's probability."
        ),
    )
    ppo_epsilon_high: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Upper bound for PPO ratio clipping: ratio <= 1 + epsilon_high. "
            "Set to 0.5 to accommodate vLLM-HuggingFace logprobs mismatch (~1.49x ratio). "
            "DAPO recommends higher than epsilon_low to encourage exploration."
        ),
    )
    use_token_level_loss: bool = Field(
        default=True,
        description=(
            "Weight loss by token count instead of sample count. "
            "Longer sequences get proportional influence on gradients. "
            "Recommended for GRPO with variable-length completions."
        ),
    )

    # =========================================================================
    # Importance Sampling Correction (for vLLM-HuggingFace logprobs mismatch)
    # =========================================================================
    importance_sampling_correction: bool = Field(
        default=True,
        description=(
            "Apply importance sampling correction to account for numerical differences "
            "between vLLM inference logprobs and HuggingFace training logprobs. "
            "Recommended when using vLLM for rollout generation."
        ),
    )
    importance_sampling_mode: Literal["sequence_truncate", "sequence_mask"] = Field(
        default="sequence_truncate",
        description=(
            "How to handle large IS ratios. "
            "'sequence_truncate': clip ratios to [1/cap, cap]. "
            "'sequence_mask': zero out sequences where ratio > cap."
        ),
    )
    importance_sampling_cap: float = Field(
        default=3.0,
        ge=1.0,
        description=(
            "Cap for importance sampling ratios. TRL default is 3.0. "
            "Lower values (e.g., 2.0) are more aggressive in limiting mismatch effects."
        ),
    )

    # =========================================================================
    # Entropy Bonus Settings (prevents mode collapse)
    # =========================================================================
    entropy_coef: float = Field(
        default=0.01,
        ge=0.0,
        description=(
            "Coefficient for entropy bonus in loss. Higher = more exploration. "
            "Prevents policy from collapsing to deterministic outputs. "
            "Set to 0.0 to disable. Typical values: 0.001-0.01."
        ),
    )
    entropy_top_k: int = Field(
        default=100,
        ge=10,
        description=(
            "Number of top tokens to use for memory-efficient entropy approximation. "
            "Lower = faster but less accurate. 100 captures most probability mass."
        ),
    )

    # =========================================================================
    # Advantage Processing Settings
    # =========================================================================
    advantage_clip: float | None = Field(
        default=10,
        description=(
            "Clip advantages to [-clip, +clip] after normalization. "
            "Prevents extreme gradients from outlier rewards. "
            "Typical values: 4.0-10.0. None disables clipping."
        ),
    )
    advantage_min_std: float = Field(
        default=1e-8,
        gt=0,
        description=(
            "Minimum std for advantage normalization. Groups with lower std are skipped. "
            "Higher values (e.g., 0.1) skip more low-signal groups."
        ),
    )

    # =========================================================================
    # Inference Settings
    # =========================================================================
    max_new_tokens: int = Field(default=120, description="Maximum tokens to generate per inference")
    temperature: float = Field(default=0.8, description="Sampling temperature for generation")
    compact_prompts: bool = Field(
        default=True, description="Use compact prompt format (reduces token count)"
    )
    # Leave this off for now - it seems to tank extraction rates. TODO: investigate.
    prefix_cache_optimized: bool = Field(
        default=True,
        description="Optimize prompt structure for vLLM prefix caching",
    )
    show_valid_moves: bool = Field(
        default=False,
        description=(
            "Include valid moves list in prompts. When False, prompts only show unit "
            "positions and rely on the logits processor for move validity. "
            "Dramatically reduces tokens late-game (~94% savings with 10+ units) "
            "but may affect strategic decision quality. A/B test recommended."
        ),
    )
    show_board_context: bool = Field(
        default=True,
        description=(
            "Include board context (supply centers, opponent positions, power rankings) when "
            "show_valid_moves=False. Provides strategic awareness for decision making."
        ),
    )
    show_map_windows: bool = Field(
        default=True,
        description=(
            "Include compact per-unit map windows (adjacent tiles + nearby threats) when "
            "show_valid_moves=False. Adds a few tokens per unit to provide strategic context."
        ),
    )
    show_action_counts: bool = Field(
        default=True,
        description=(
            "Show action count and types per unit (e.g., '15 moves | move,support') when "
            "show_valid_moves=False. Helps model understand action space size without listing all moves. "
            "Adds ~10-15 tokens per unit."
        ),
    )
    compute_ref_logprobs_in_rollout: bool = Field(
        default=False,
        description=(
            "Compute reference (base model) logprobs during rollouts. "
            "If True: Rollouts slower, training faster (no ref forward pass). "
            "If False: Rollouts faster, training does ref forward pass. "
            "Default False is optimal when rollouts are the bottleneck (trainer data-starved)."
        ),
    )

    # =========================================================================
    # Profiling / Instrumentation
    # =========================================================================
    profiling_mode: ProfilingMode | None = Field(
        default=None,
        description="Enable profiling: 'rollout', 'trainer', or 'e2e'",
    )
    profile_run_name: str | None = Field(
        default=None, description="Custom name for profiling output files"
    )
    profiling_trace_steps: int = Field(
        default=3, description="Number of steps to capture in profiler trace"
    )

    # =========================================================================
    # Computed Properties
    # =========================================================================
    @property
    def batch_size(self) -> int:
        """Total batch size per training step."""
        return self.num_groups_per_step * self.samples_per_group

    @property
    def expected_horizon_years(self) -> float:
        """Expected rollout horizon accounting for variable horizons."""
        return (
            self.rollout_horizon_years * (1 - self.rollout_long_horizon_chance)
            + self.rollout_long_horizon_years * self.rollout_long_horizon_chance
        )

    @property
    def simulated_years_per_step(self) -> int:
        """Calculate expected simulated years per training step."""
        return int(self.num_groups_per_step * self.samples_per_group * self.expected_horizon_years)

    @property
    def total_simulated_years(self) -> int:
        """Calculate total simulated years for the full training run."""
        return self.simulated_years_per_step * self.total_steps

    @model_validator(mode="after")
    def validate_pfsp_weights(self) -> ExperimentConfig:
        """Validate that PFSP weights sum to 1.0."""
        total = (
            self.pfsp_self_play_weight
            + self.pfsp_peer_weight
            + self.pfsp_exploitable_weight
            + self.pfsp_baseline_weight
        )
        if not (0.99 <= total <= 1.01):  # Allow small floating point tolerance
            raise ValueError(
                f"PFSP weights must sum to 1.0, got {total:.4f}. "
                f"(self_play={self.pfsp_self_play_weight}, peer={self.pfsp_peer_weight}, "
                f"exploitable={self.pfsp_exploitable_weight}, baseline={self.pfsp_baseline_weight})"
            )
        return self

    @model_validator(mode="after")
    def validate_lora_rank(self) -> ExperimentConfig:
        """Validate lora_rank doesn't exceed inference engine max.

        This catches the bug where lora_rank in config exceeds the max_lora_rank
        configured in InferenceEngine, which causes silent inference timeouts.
        """
        # Must match src/apps/inference_engine/app.py line 490
        MAX_LORA_RANK = 32

        if self.lora_rank > MAX_LORA_RANK:
            raise ValueError(
                f"lora_rank={self.lora_rank} exceeds InferenceEngine max_lora_rank={MAX_LORA_RANK}. "
                f"Either:\n"
                f"  1. Set lora_rank <= {MAX_LORA_RANK} in your config, OR\n"
                f"  2. Update max_lora_rank in src/apps/inference_engine/app.py:490 "
                f"and src/utils/preflight.py, then redeploy"
            )
        return self


class EvalConfig(BaseModel):
    """
    Configuration for checkpoint evaluation.

    Separate from ExperimentConfig to keep concerns clear.
    """

    # =========================================================================
    # Checkpoint Settings
    # =========================================================================
    checkpoint_path: str = Field(..., description="Path to checkpoint relative to /data/models")
    base_model_id: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct",
        description="HuggingFace model ID for base model",
    )

    # =========================================================================
    # Evaluation Settings
    # =========================================================================
    opponents: list[str] = Field(
        default=["random", "chaos"],
        description="Opponent types to evaluate against",
    )
    games_per_opponent: int = Field(default=10, description="Number of games per opponent type")
    max_years: int = Field(default=10, description="Maximum game length in years")
    eval_powers: list[str] = Field(
        default=["FRANCE"],
        description="Which powers use the checkpoint",
    )

    # =========================================================================
    # Visualization Settings
    # =========================================================================
    visualize: bool = Field(default=True, description="Generate HTML visualizations of games")
    visualize_sample_rate: float = Field(
        default=0.3, description="Fraction of games to visualize (0.0-1.0)"
    )

    # =========================================================================
    # Logging Settings
    # =========================================================================
    log_to_wandb: bool = Field(default=True, description="Log results to WandB")
    wandb_run_name: str | None = Field(default=None, description="Custom WandB run name")
    wandb_project: str = Field(default="diplomacy-grpo", description="WandB project name")


# =============================================================================
# CLI Argument Helpers
# =============================================================================


def _get_field_type_name(field_type: Any) -> str:
    """Get a human-readable name for a field type."""
    origin = get_origin(field_type)
    if origin is Literal:
        choices = get_args(field_type)
        return f"one of {choices}"
    if origin is list:
        inner = get_args(field_type)
        if inner:
            return f"list of {inner[0].__name__}"
        return "list"
    if hasattr(field_type, "__name__"):
        return field_type.__name__
    return str(field_type)


def add_config_args(
    parser: argparse.ArgumentParser,
    config_class: type[BaseModel],
    exclude: set[str] | None = None,
    prefix: str = "",
) -> None:
    """
    Auto-generate argparse arguments from a Pydantic model.

    This ensures CLI arguments stay in sync with the config model.
    Arguments are named with dashes (e.g., --learning-rate) but stored
    with underscores in the namespace to match Pydantic field names.

    Args:
        parser: ArgumentParser to add arguments to
        config_class: Pydantic model class to generate args from
        exclude: Set of field names to skip (e.g., computed properties)
        prefix: Optional prefix for argument names

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> add_config_args(parser, ExperimentConfig, exclude={"batch_size"})
        >>> args = parser.parse_args(["--total-steps", "100"])
        >>> cfg = config_from_args(args, ExperimentConfig)
    """
    exclude = exclude or set()

    for field_name, field_info in config_class.model_fields.items():
        if field_name in exclude:
            continue

        # Convert underscore to dash for CLI (e.g., learning_rate -> learning-rate)
        arg_name = f"--{prefix}{field_name.replace('_', '-')}"

        # Get field type and default
        field_type = field_info.annotation
        default = field_info.default
        description = field_info.description or ""

        # Handle Optional types (Union[X, None])
        origin = get_origin(field_type)
        is_optional = origin is type(None) or (
            hasattr(origin, "__origin__") and type(None) in get_args(field_type)
        )

        # Extract inner type from Optional
        if origin is type(str | None) or (is_optional and origin is not None):
            args = get_args(field_type)
            inner_types = [t for t in args if t is not type(None)]
            if inner_types:
                field_type = inner_types[0]
                origin = get_origin(field_type)

        # Build argument kwargs
        kwargs: dict[str, Any] = {
            "help": f"{description} (default: {default})",
            "dest": field_name,  # Store with underscore name
        }

        # Handle different types
        if field_type is bool:
            # Booleans get --flag and --no-flag
            if default is True:
                parser.add_argument(
                    f"--no-{prefix}{field_name.replace('_', '-')}",
                    action="store_false",
                    dest=field_name,
                    help=f"Disable: {description}",
                )
            else:
                parser.add_argument(
                    arg_name,
                    action="store_true",
                    dest=field_name,
                    help=description,
                )
            continue

        elif origin is Literal:
            # Literal types become choices
            choices = get_args(field_type)
            kwargs["choices"] = choices
            kwargs["type"] = str
            kwargs["default"] = default

        elif origin is list:
            # List types use nargs
            inner_types = get_args(field_type)
            inner_type = inner_types[0] if inner_types else str
            kwargs["nargs"] = "+"
            kwargs["type"] = inner_type
            kwargs["default"] = default

        elif field_type in (int, float, str):
            kwargs["type"] = field_type
            kwargs["default"] = default

        else:
            # Fallback for complex types - treat as string
            kwargs["type"] = str
            kwargs["default"] = str(default) if default is not None else None

        parser.add_argument(arg_name, **kwargs)


def config_from_args(
    args: argparse.Namespace,
    config_class: type[BaseModel],
    extra_values: dict[str, Any] | None = None,
) -> BaseModel:
    """
    Create a Pydantic config from parsed argparse namespace.

    Args:
        args: Parsed argument namespace
        config_class: Pydantic model class to instantiate
        extra_values: Additional values to include (override args)

    Returns:
        Instantiated config model

    Example:
        >>> args = parser.parse_args(["--total-steps", "100"])
        >>> cfg = config_from_args(args, ExperimentConfig)
        >>> print(cfg.total_steps)
        100
    """
    # Convert namespace to dict, filtering out None values for optional fields
    values = {
        k: v for k, v in vars(args).items() if v is not None and k in config_class.model_fields
    }

    # Apply extra values
    if extra_values:
        values.update(extra_values)

    return config_class(**values)


def create_experiment_parser(
    description: str = "Run Diplomacy GRPO experiment",
    exclude: set[str] | None = None,
) -> argparse.ArgumentParser:
    """
    Create a fully-configured argument parser for experiments.

    Convenience function that creates a parser with all ExperimentConfig fields.

    Args:
        description: Parser description
        exclude: Fields to exclude from CLI

    Returns:
        Configured ArgumentParser

    Example:
        >>> parser = create_experiment_parser("My training script")
        >>> parser.add_argument("--my-custom-arg", type=str)
        >>> args = parser.parse_args()
        >>> cfg = config_from_args(args, ExperimentConfig)
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_args(parser, ExperimentConfig, exclude=exclude)
    return parser


def create_eval_parser(
    description: str = "Evaluate Diplomacy GRPO checkpoint",
    exclude: set[str] | None = None,
) -> argparse.ArgumentParser:
    """
    Create a fully-configured argument parser for evaluation.

    Args:
        description: Parser description
        exclude: Fields to exclude from CLI

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_args(parser, EvalConfig, exclude=exclude)
    return parser
