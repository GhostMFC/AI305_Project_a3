"""Explicit, strict-load adapters. No ground-truth object poses enter either model."""
import json
from pathlib import Path

PI_PROMPT = ('grasp five yellow blocks together from the source bin, insert them upright into the marked '
             'slots of the 2 by 5 box, then return to the source bin, grasp five more, and fill all ten slots')
SMOL_PROMPT = 'transfer 10 cookies into target box'
CAMERAS = ('observation.images.front', 'observation.images.left_wrist', 'observation.images.right_wrist')


class TeacherPolicy:
    action_mode = 'joint_position'

    def __init__(self, kind, checkpoint, device, horizon, flow_steps, seed):
        import numpy as np
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.utils import prepare_observation_for_inference
        from a3_dual_arm_sim.learning.checkpoint_bundle import load_processors
        self.np, self.torch, self.prepare = np, torch, prepare_observation_for_inference
        self.kind, self.seed, self.device = kind, seed, torch.device(device)
        self.name = f'teacher_{kind}_h{horizon}_f{flow_steps}'
        checkpoint = Path(checkpoint).resolve()
        if kind == 'smolvla':
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            cls = SmolVLAPolicy
        else:
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy
            cls = PI05Policy
        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        if cfg.type != kind or not 1 <= horizon <= cfg.chunk_size:
            raise ValueError('Wrong checkpoint type or horizon')
        cfg.device, cfg.n_action_steps = device, horizon
        cfg.compile_model = False
        cfg.use_amp = False
        if kind == 'smolvla':
            cfg.num_steps = flow_steps
        else:
            cfg.num_inference_steps = flow_steps
            if cfg.use_relative_actions:
                raise ValueError('This adapter is for the published absolute-action checkpoint')
        # Bypass PI05.from_pretrained: version 0.5.1 catches weight errors and can return
        # a randomly initialized model. Strict loading here must fail loudly instead.
        self.model = cls(cfg)
        if kind == 'pi05':
            from safetensors.torch import load_file
            state = load_file(str(checkpoint / 'model.safetensors'), device='cpu')
            state = self.model._fix_pytorch_state_dict_keys(state, cfg)
            state = {k if k.startswith('model.') else 'model.' + k: v for k, v in state.items()}
            self.model.load_state_dict(state, strict=True)
            del state
        else:
            from safetensors.torch import load_model
            load_model(self.model, str(checkpoint / 'model.safetensors'), strict=True, device='cpu')
        self.model.to(self.device).eval()
        self.pre, self.post = load_processors(cfg, checkpoint, None, device)
        self.action_dim = cfg.output_features['action'].shape[0]
        expected_dim = 8 if kind == 'smolvla' else 16
        if self.action_dim != expected_dim:
            raise ValueError(f'Expected released {expected_dim}D actions, got {self.action_dim}')
        self.prompt = SMOL_PROMPT if kind == 'smolvla' else PI_PROMPT
        self.effective_config = __import__('dataclasses').asdict(cfg)
        self.last_raw_action = None
        self.right_hold = None
        self.timing = []
        print('STRICT_WEIGHTS_LOADED: all model parameters matched', flush=True)

    def reset(self, context=None):
        self.torch.manual_seed(self.seed)
        self.model.reset()
        for processor in (self.pre, self.post):
            if hasattr(processor, 'reset'):
                processor.reset()
        self.right_hold = None
        self.last_raw_action = None

    def act(self, observation, task=''):
        import time
        np, torch = self.np, self.torch
        state = np.asarray(observation['observation.state'], dtype=np.float32)
        if state.shape != (16,) or not np.isfinite(state).all():
            raise ValueError('Expected finite 16D A3 joint state')
        if self.right_hold is None:
            self.right_hold = state[8:].copy()
        inputs = {k: np.array(observation[k], copy=True) for k in CAMERAS}
        # PI05 stats are 16D despite config declaring a 32D maximum. Do not append
        # velocities or pad before the saved 16D quantile normalizer.
        inputs['observation.state'] = state[:self.action_dim].copy()
        batch = self.prepare(inputs, self.device, self.prompt, 'A3_dual_arm')
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        batch = self.pre(batch)
        with torch.inference_mode():
            action = self.post(self.model.select_action(batch))
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        self.timing.append(time.perf_counter() - start)
        a = action.detach().float().cpu().numpy().reshape(-1)
        if a.shape != (self.action_dim,) or not np.isfinite(a).all():
            raise ValueError(f'Invalid action shape or non-finite action: {a.shape}')
        if self.action_dim == 8:
            a = np.concatenate((a, self.right_hold))
        self.last_raw_action = a.copy()
        return a

    def close(self):
        self.model = None
