import re
import json
from pathlib import Path
import torch
from mmengine.hooks import Hook
from mmengine.dist import is_main_process, all_reduce
from mmengine.dist import is_distributed
from mmdet.registry import HOOKS

@HOOKS.register_module()
class MoELevelStatsHook(Hook):
    def __init__(self, num_levels=4, name_pattern=r"(?:^|\.)(router)(?:$|\.|_)",
                 out_file="router_stats.json"):
        """
        name_pattern: 用于匹配 model.named_modules() 的模块名
          默认匹配名字里带 router 的模块。
        """
        self.num_levels = num_levels
        # self.name_re = re.compile(name_pattern)
        self.name_re = re.compile(r"router", re.IGNORECASE)
        self.out_file = out_file

        self.level_counts = {}  # module_name -> Tensor[num_levels]
        self.pair_counts = {}   # module_name -> Tensor[num_levels, num_levels]

    def _ensure_buffers(self, module_name):
        if module_name not in self.level_counts:
            self.level_counts[module_name] = torch.zeros(self.num_levels, dtype=torch.long)
            self.pair_counts[module_name] = torch.zeros(self.num_levels, self.num_levels, dtype=torch.long)

    @torch.no_grad()
    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        model = runner.model.module if hasattr(runner.model, "module") else runner.model

        for name, module in model.named_modules():
            if not self.name_re.search(name):
                continue
            if "router" in name.lower():
                runner.logger.info(f"matched module: {name}, has_attr={hasattr(module, '_last_idx_selected')}")
            top2: torch.Tensor  = getattr(module, "_last_idx_selected", None)
            runner.logger.info(
                f"[router stat] name={name}, top2_shape={tuple(top2.shape)}, "
                f"min={top2.min().item() if top2.numel() else 'empty'}, "
                f"max={top2.max().item() if top2.numel() else 'empty'}"
            )
            if top2 is None:
                continue

            self._ensure_buffers(name)

            if not torch.is_tensor(top2):
                top2 = torch.as_tensor(top2)
            top2 = top2.detach().cpu()

            # top2: [K,2]，K 可以是 B 或 query 数等
            if top2.ndim != 2 or top2.size(-1) != 2:
                # 如果你的 router 输出不是 [*,2]，你需要在 router 里统一存成这种形状
                continue

            for i in range(top2.size(0)):
                a = int(top2[i, 0])
                b = int(top2[i, 1])
                if not (0 <= a < self.num_levels and 0 <= b < self.num_levels):
                    continue
                self.level_counts[name][a] += 1
                self.level_counts[name][b] += 1
                x, y = (a, b) if a <= b else (b, a)
                self.pair_counts[name][x, y] += 1

    def after_test(self, runner):
        # 1) 拿到真实 model（兼容 DDP）
        model = runner.model.module if hasattr(runner.model, "module") else runner.model

        # 2) 稳定拿 device：从任意参数取；没有参数就 cpu
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        out = {}

        for name in sorted(self.level_counts.keys()):
            lc = self.level_counts[name].to(device)
            pc = self.pair_counts[name].to(device)

            # 3) 分布式才做 all_reduce（单卡时跳过也没影响）
            if is_distributed():
                all_reduce(lc)
                all_reduce(pc)

            out[name] = {
                "level_counts": lc.cpu().tolist(),
                "pair_counts_uppertri": pc.cpu().tolist(),
            }

        if is_main_process():
            out_path = Path(runner.work_dir) / self.out_file
            payload = {
                "num_levels": self.num_levels,
                "routers": out
            }
            out_path.write_text(json.dumps(payload, indent=2))
            runner.logger.info(f"Saved router stats to {out_path}")