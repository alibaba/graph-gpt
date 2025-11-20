# coding=utf-8
# Copyright 2024 The Dream team, HKUNLP Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# copied and modified from https://huggingface.co/Dream-org/Dream-v0-Base-7B/blob/main/generation_utils.py
import torch
import torch.distributions as dists
from torch.nn import functional as F
from ..conf.generation import GenerationConfig


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(
    logits,
    temperature=0.0,
    top_p=None,
    top_k=None,
    margin_confidence=False,
    neg_entropy=False,
):
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            # TODO: implement Gumbel-softmax sampling!
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities; `...` ensure operating on tensors of arbitrary dims
        top1_probs = sorted_probs[..., 0]
        top2_probs = sorted_probs[..., 1]
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    return confidence, x0


@torch.no_grad()
def sample_per_batch(
    model,
    cfg: GenerationConfig,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    inputs_raw_embeds: torch.Tensor,
):
    model.eval()
    assert input_ids.dim() == 3, "expect [bz, seq, next_n]"
    bz, seq, next_n = input_ids.shape
    device = input_ids.device

    # 0. init
    eps = cfg.eps
    max_steps = cfg.steps
    mask_token_id = cfg.mask_token_id

    # 1. format inputs
    x = input_ids.clone()  # [bz, seq, next_n]
    x = x.view(bz, seq * next_n)  # [bz, seq*next_n]

    m = x == mask_token_id  # [bz, seq*next_n]  bool
    func = torch.max  # torch.mean|torch.max|torch.median
    steps = min(int(func(m.sum(dim=-1).float()).item()), max_steps)

    # 2. time steps
    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    histories = [] if cfg.output_history else None

    # 3. iterative decoding
    # compare to `for loop`, `while loop` can skip the useless inference when no token to unmask!!!
    i = 0
    while i < steps:
        # 3.1 convert to specific shape to infer, and then convert to shape compatible to diffusion LLM
        logits = model(
            input_ids=x.view(bz, seq, next_n),
            attention_mask=attention_mask,
            labels=None,
            inputs_raw_embeds=inputs_raw_embeds,
        ).head1_logits  # [bz*seq*next_n, vocab]

        vocab_size = logits.size(-1)
        logits = logits.view(bz, seq * next_n, vocab_size)

        # 3.2 apply un-mask
        # x = _batch_unmask_with_for_loop(x, logits, timesteps, i, cfg)
        x, i = _batch_unmask_without_for_loop(x, logits, timesteps, i, cfg)
        if histories is not None:
            histories.append(x.view(bz, seq, next_n).clone())
    return x, histories


def _batch_unmask_without_for_loop(
    x: torch.Tensor,
    logits: torch.Tensor,
    timesteps: torch.Tensor,
    i: int,
    cfg: GenerationConfig,
):
    steps = len(timesteps) - 1
    mask_token_id = cfg.mask_token_id
    device = x.device
    temperature = cfg.temperature
    top_p = cfg.top_p
    top_k = cfg.top_k
    alg = cfg.alg
    alg_temp = cfg.alg_temp

    mask_index = x == mask_token_id  # shape: [bz, total_tokens]

    if alg == "origin":
        t, s = timesteps[i], timesteps[i + 1]
        p_transfer = 1 - s / t if i < steps - 1 else 1.0
        # 为所有位置采样，而不仅仅是 mask 位置
        _, x0_candidates = sample_tokens(
            logits, temperature=temperature, top_p=top_p, top_k=top_k
        )  # x0_candidates shape: [bz, total_tokens]
        # 创建一个随机掩码来决定哪些 token 被更新
        transfer_mask = torch.rand(x.shape, device=x.device) < p_transfer
        # 只有在当前是 [MASK] 并且随机掩码为 True 的位置才更新
        update_positions = mask_index & transfer_mask
        # 使用 torch.where 高效地进行批处理更新
        x = torch.where(update_positions, x0_candidates, x)
        i += 1
    else:  # maskgit_plus, topk_margin, entropy
        k = 0
        # 1. Calculate how many tokens to unmask for each sample in the batch
        num_masked_per_sample = mask_index.sum(dim=1)  # shape: [bz]
        num_masked_all = num_masked_per_sample.sum().item()
        while (k == 0) and (num_masked_all > 0) and (i < steps):
            # if no tokens to unmask, move to the next step, i.e., i += 1
            t, s = timesteps[i], timesteps[i + 1]
            p_transfer = 1 - s / t if i < steps - 1 else 1.0
            # 每个样本需要 unmask 的数量，向下取整; num_transfer_per_sample -> shape: [bz]
            num_transfer_per_sample = torch.floor(
                num_masked_per_sample * p_transfer
            ).int()  # shape: [bz]
            k = num_transfer_per_sample.max().item()
            i += 1

        # 2. Get confidence scores and candidate tokens
        # `sample_tokens` 内部的 softmax 和 top_k/p 都是在最后一个维度上操作，天然支持批处理
        confidence, x0_candidates = sample_tokens(
            logits,  # [bz, seq*next_n, vocab]
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            margin_confidence=(alg == "topk_margin"),
            neg_entropy=(alg == "entropy"),
        )  # confidence, x0_candidates shapes: [bz, seq*next_n]

        # Mask out non-masked positions to ensure we only select from them
        confidence[~mask_index] = -torch.inf

        # 2.1 If alg_temp is specified, use the Gumbel-Max trick to enable stochastic sampling.
        if alg_temp is not None and alg_temp > 0:
            # Apply temperature to confidence scores (treating them as logits)
            confidence = confidence / alg_temp

            # Sample Gumbel noise and add it to the logits.
            # Adding noise before topk is equivalent to multinomial sampling.
            gumbel_noise = -torch.log(
                -torch.log(torch.rand_like(confidence) + 1e-9) + 1e-9
            )
            confidence = confidence + gumbel_noise
        # If alg_temp is 0 or None, we just use the original confidence,
        # which results in a deterministic top-k selection.

        # 3. Get the indices to unmask using our (potentially perturbed) confidence
        # `torch.topk` 需要一个固定的 k。一个常见的策略是使用批次中最大的 k 值，
        # 然后对不足 k 的样本进行处理。一个更简单的近似是使用一个平均的 k。
        # 这里我们采用更精确但略复杂的方法：对每个样本选择其对应的 top-k。
        # 这通常需要循环，但我们可以用一个技巧来向量化它。
        # 然而，最直接的批处理方法是使用一个固定的k。
        # 我们选择批次中最大的 num_transfer 作为 k。
        _, transfer_indices = torch.topk(confidence, k=k, dim=1)  # shape: [bz, k]

        # 4. 执行更新
        # 4.1. 首先，获取所有k个候选更新值: shape: [bz, k]
        updates = torch.gather(x0_candidates, 1, transfer_indices)
        # 4.2. 确定哪些位置是“多余”的更新，需要被变回 MASK
        # arange shape: [1, k], num_transfer_per_sample[:, None] shape: [bz, 1]
        # mask_out_mask shape: [bz, k]
        arange = torch.arange(k, device=device)[None, :]
        mask_out_mask = arange >= num_transfer_per_sample[:, None]
        # 4.3. 准备最终要写入的值：
        # 如果 mask_out_mask 为 True (是多余的)，则使用 mask_token_id
        # 否则，使用候选的 update token
        final_updates = torch.where(mask_out_mask, mask_token_id, updates)
        # 4.4. 执行一次性的、正确的 scatter 操作
        x.scatter_(1, transfer_indices, final_updates)
    return x, i


def _batch_unmask_with_for_loop(
    x: torch.Tensor,
    logits: torch.Tensor,
    timesteps: torch.Tensor,
    i: int,
    cfg: GenerationConfig,
):
    # even slower than sample_per_example in `circuit` dataset
    steps = len(timesteps) - 1
    bz = x.size(0)
    mask_token_id = cfg.mask_token_id
    device = x.device
    temperature = cfg.temperature
    top_p = cfg.top_p
    top_k = cfg.top_k
    alg = cfg.alg
    alg_temp = cfg.alg_temp

    mask_index = x == cfg.mask_token_id  # [bz, seq*next_n]  bool
    mask_logits = logits[mask_index]  # [M, V]  M=total # of mask tokens

    t, s = timesteps[i], timesteps[i + 1]

    if cfg.alg == "origin":
        p_transfer = 1 - s / t if i < steps - 1 else 1
        x0 = torch.full_like(x[mask_index], mask_token_id, device=device)
        transfer_index_t_s = torch.rand(*x0.shape, device=x.device) < p_transfer
        _, x0[transfer_index_t_s] = sample_tokens(
            mask_logits[transfer_index_t_s],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        x[mask_index] = x0.clone()
    else:
        # maskgit_plus / topk_margin / entropy  3 sampling algos
        confidence, sampled = sample_tokens(
            mask_logits,
            temperature,
            top_p,
            top_k,
            margin_confidence=(alg == "topk_margin"),
            neg_entropy=(alg == "entropy"),
        )  # [M]
        # 4. Calculate unmasked tokens per example
        num_mask_per_ex = mask_index.sum(dim=1)  # [bz]
        num_trans_per_ex = (
            (num_mask_per_ex * (1 - s / t)).long() if i < steps - 1 else num_mask_per_ex
        )  # [bz]

        # 4. Construct full_confidence matrix for top-k / sampling
        full_conf = torch.full_like(x, -torch.inf, dtype=logits.dtype)
        full_conf[mask_index] = confidence

        # 4.1 Apply per-sample-level top-k or softmax sampling
        x0 = torch.full_like(x, mask_token_id)  # [bz, seq*next_n]
        x0[mask_index] = sampled
        new_x = x.clone()  # [bz, seq*next_n]
        for b in range(bz):
            number_transfer_tokens = num_trans_per_ex[b].item()
            if number_transfer_tokens == 0:
                continue
            # all mask pos of current sample
            mask_idx_b = mask_index[b].nonzero(as_tuple=True)[0]  # [k]
            conf_b = full_conf[b, mask_idx_b]  # [k]
            if alg_temp is None or alg_temp == 0:
                _, sel = torch.topk(conf_b, number_transfer_tokens)
            else:
                conf_b = F.softmax(conf_b / alg_temp, dim=0)
                sel = torch.multinomial(conf_b, number_transfer_tokens)
            pos_to_fill = mask_idx_b[sel]
            new_x[b, pos_to_fill] = x0[b, pos_to_fill]
        x = new_x
    return x


@torch.no_grad()
def sample_per_example(
    model,
    cfg: GenerationConfig,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    inputs_raw_embeds: torch.Tensor,
):
    """
    :param model:
    :param cfg:
    :param input_ids: [seq, next_n]
    :param attention_mask: [bz, seq]
    :param inputs_raw_embeds: [bz, seq, dim] or None
    :return:
    """
    model.eval()
    assert input_ids.dim() == 2

    # 0. init
    alg = cfg.alg
    eps = cfg.eps
    max_steps = cfg.steps
    temperature = cfg.temperature
    top_p = cfg.top_p
    top_k = cfg.top_k
    alg_temp = cfg.alg_temp
    mask_token_id = cfg.mask_token_id

    # 1. format inputs
    x = input_ids.clone()
    x = x.unsqueeze(0)  # [max_length, next_n_token] -> [1, max_length, next_n_token]
    mask = x == mask_token_id  # bool
    steps = min(int(mask.sum().item()), max_steps)
    bz, seq, next_n_token = x.shape

    # 2. time steps
    timesteps = torch.linspace(1, eps, steps + 1, device=x.device)
    histories = [] if cfg.output_history else None

    # 3. iterative decoding
    for i in range(steps):
        # 3.1 convert to specific shape to infer, and then convert to shape compatible to diffusion LLM
        x = x.view(bz, seq, next_n_token)
        logits = model(
            input_ids=x,
            attention_mask=attention_mask,
            labels=None,
            inputs_raw_embeds=inputs_raw_embeds,
        ).head1_logits

        x = x.view(bz, seq * next_n_token)

        vocab_size = logits.shape[-1]
        logits = logits.view(bz, seq * next_n_token, vocab_size)
        # 3.2
        mask_index = x == mask_token_id

        mask_logits = logits[mask_index]
        t = timesteps[i]
        s = timesteps[i + 1]

        if alg == "origin":
            p_transfer = 1 - s / t if i < steps - 1 else 1
            x0 = (
                torch.zeros_like(x[mask_index], device=x.device, dtype=torch.long)
                + mask_token_id
            )
            transfer_index_t_s = torch.rand(*x0.shape, device=x.device) < p_transfer
            _, x0[transfer_index_t_s] = sample_tokens(
                mask_logits[transfer_index_t_s],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            x[mask_index] = x0.clone()
        else:
            confidence, x0 = sample_tokens(
                mask_logits,
                temperature,
                top_p,
                top_k,
                margin_confidence=(alg == "topk_margin"),
                neg_entropy=(alg == "entropy"),
            )  # [M]
            num_mask_token = mask_index.sum() / mask_index.shape[0]
            number_transfer_tokens = (
                int(num_mask_token * (1 - s / t))
                if i < steps - 1
                else int(num_mask_token)
            )
            full_confidence = torch.full_like(
                x, -torch.inf, device=x.device, dtype=logits.dtype
            )
            full_confidence[mask_index] = confidence
            if number_transfer_tokens > 0:
                if alg_temp is None or alg_temp == 0:
                    _, transfer_index = torch.topk(
                        full_confidence, number_transfer_tokens
                    )
                else:
                    full_confidence = full_confidence / alg_temp
                    full_confidence = F.softmax(full_confidence, dim=-1)
                    transfer_index = torch.multinomial(
                        full_confidence, num_samples=number_transfer_tokens
                    )
                x_ = (
                    torch.zeros_like(x, device=x.device, dtype=torch.long)
                    + mask_token_id
                )
                x_[mask_index] = x0.clone()
                row_indices = (
                    torch.arange(x.size(0), device=x.device)
                    .unsqueeze(1)
                    .expand_as(transfer_index)
                )
                x[row_indices, transfer_index] = x_[row_indices, transfer_index]
        if histories is not None:
            histories.append(x.clone())
    return x, histories


def cal_gen_acc_per_sample(gen_cfg: GenerationConfig, input_ids, labels, gen_res):
    # [seq*next_n_token] -> [seq, next_n_token]
    gen_res = gen_res.reshape(input_ids.shape)
    m = input_ids == gen_cfg.mask_token_id

    acc = (gen_res[m] == labels[m]).sum() / len(labels[m])
    return acc


def cal_gen_acc_batch(gen_cfg: GenerationConfig, input_ids, labels, gen_res):
    """
    :param gen_cfg:
    :param input_ids: [bz, seq, next_n]
    :param labels: [bz, seq, next_n]
    :param gen_res: [bz, seq*next_n]
    :return:
        A tensor of shape [batch_size], where each element is the accuracy
        for the corresponding sample in the batch.
    """
    bz, seq, next_n = input_ids.shape
    gen_res = gen_res.view(bz, seq, next_n)
    mask = input_ids == gen_cfg.mask_token_id  # [bz, seq, next_n]
    correct = (gen_res == labels) & mask
    acc_per_ex = correct.sum(dim=(1, 2)).float() / mask.sum(dim=(1, 2)).float()
    return acc_per_ex
