# hooks/finetune_token_hook.py

import torch
from mmengine.hooks import Hook
from mmengine.logging import MMLogger
from mmdet.registry import HOOKS

@HOOKS.register_module()
class FineTuneTokenHook(Hook):
    """
    一个专门用于在类BERT模型中仅微调"object"词元嵌入的Hook。
    """
    def __init__(self, token="object", language_model_path="language_model"):
        self.token = token
        self.language_model_path = language_model_path
        self._hook_registered = False

    def _get_language_model(self, runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        
        language_model = model
        for part in self.language_model_path.split('.'):
            language_model = getattr(language_model, part)
        return language_model


    def before_run(self, runner):
        if self._hook_registered:
            return

        logger: MMLogger = runner.logger
        logger.info(f"[{self.__class__.__name__}] 开始应用选择性微调逻辑。")

        # 获取语言模型
        language_model = self._get_language_model(runner)
        embedding_layer = language_model.language_backbone.body.model.embeddings.word_embeddings

        if not hasattr(language_model, 'tokenizer'):
            raise RuntimeError("语言模型没有 'tokenizer' 属性！")
        
        tokenizer = language_model.tokenizer

        logger.info(f"[{self.__class__.__name__}] 冻结所有模型参数。")
        for param in runner.model.parameters():
            param.requires_grad = False

        logger.info(f"[{self.__class__.__name__}] 解冻词嵌入权重张量。")
        embedding_layer.weight.requires_grad = True

        # 注册后向钩子以屏蔽梯度
        token_id = tokenizer.convert_tokens_to_ids(self.token)
        if isinstance(token_id, list):
            token_id = token_id[0]

        if token_id == tokenizer.unk_token_id:
            logger.warning(f"词元 '{self.token}' 不在分词器词汇表中。")
            return

        def grad_mask_hook(grad):
            mask = torch.zeros_like(grad)
            mask[token_id, :] = 1.0
            return grad * mask

        embedding_layer.weight.register_hook(grad_mask_hook)
        self._hook_registered = True
        
        logger.info(
            f"[{self.__class__.__name__}] 成功注册梯度钩子。"
            f"只有 '{self.token}' (ID: {token_id}) 对应的嵌入会被更新。"
        )