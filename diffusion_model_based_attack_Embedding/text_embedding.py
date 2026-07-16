from __future__ import annotations

from typing import Any

import torch
from torch import nn


def build_placeholder_tokens(base_token: str, num_vectors: int) -> list[str]:
    if num_vectors < 1:
        raise ValueError("--num-vectors must be >= 1.")
    if num_vectors == 1:
        return [base_token]
    if base_token.startswith("<") and base_token.endswith(">"):
        stem = base_token[:-1]
        return [base_token] + [f"{stem}-{index}>" for index in range(1, num_vectors)]
    return [base_token] + [f"{base_token}_{index}" for index in range(1, num_vectors)]


def expand_prompt(prompt: str, placeholder_token: str, placeholder_tokens: list[str]) -> str:
    if placeholder_token not in prompt:
        raise ValueError(f"Prompt must contain placeholder token '{placeholder_token}'.")
    return prompt.replace(placeholder_token, " ".join(placeholder_tokens))


class PlaceholderEmbeddingManager:
    """Adds textual-inversion-style placeholder rows and patches embedding lookup output."""

    def __init__(
        self,
        *,
        name: str,
        tokenizer: Any,
        text_encoder: nn.Module,
        placeholder_tokens: list[str],
        initializer_token: str,
        trainable: bool,
        device: torch.device,
    ) -> None:
        self.name = name
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.placeholder_tokens = placeholder_tokens
        self.initializer_token = initializer_token
        self.trainable = trainable
        self.device = device
        self.placeholder_ids: list[int] = []
        self.parameter: nn.Parameter | None = None
        self.initial_parameter: torch.Tensor | None = None
        self._install()

    def _install(self) -> None:
        added = self.tokenizer.add_tokens(self.placeholder_tokens)
        if added < 0:
            raise RuntimeError(f"Tokenizer for {self.name} returned an invalid add_tokens result: {added}.")
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        embedding = self.text_encoder.get_input_embeddings()
        embedding.weight.requires_grad_(False)

        initializer_ids = self.tokenizer.encode(self.initializer_token, add_special_tokens=False)
        if len(initializer_ids) == 0:
            raise ValueError(f"Initializer token '{self.initializer_token}' produced no token ids for {self.name}.")
        initializer_id = int(initializer_ids[0])
        if len(initializer_ids) > 1:
            print(
                f"Warning: initializer token '{self.initializer_token}' maps to multiple ids for {self.name}; "
                f"using the first id {initializer_id}."
            )

        for token in self.placeholder_tokens:
            token_id = int(self.tokenizer.convert_tokens_to_ids(token))
            if token_id == self.tokenizer.unk_token_id:
                raise ValueError(f"Could not add placeholder token '{token}' to tokenizer for {self.name}.")
            self.placeholder_ids.append(token_id)

        with torch.no_grad():
            init_vector = embedding.weight.data[initializer_id].detach().clone()
            for token_id in self.placeholder_ids:
                embedding.weight.data[token_id].copy_(init_vector)

        initial = embedding.weight.detach()[self.placeholder_ids].clone().float().to(self.device)
        if self.trainable:
            self.parameter = nn.Parameter(initial.clone())
            self.initial_parameter = initial.clone().detach()
            self._patch_embedding_forward(embedding)
        else:
            self.initial_parameter = initial.clone().detach()

    def _patch_embedding_forward(self, embedding: nn.Embedding) -> None:
        if self.parameter is None:
            return
        original_forward = embedding.forward
        token_ids = torch.tensor(self.placeholder_ids, device=self.device, dtype=torch.long)
        parameter = self.parameter

        def patched_forward(input_ids: torch.Tensor) -> torch.Tensor:
            output = original_forward(input_ids)
            learned = parameter.to(device=output.device, dtype=output.dtype)
            local_token_ids = token_ids.to(device=input_ids.device)
            view_shape = [1] * input_ids.ndim + [output.shape[-1]]
            for row_index, token_id in enumerate(local_token_ids):
                replacement = learned[row_index].view(*view_shape)
                mask = (input_ids == token_id).unsqueeze(-1)
                output = torch.where(mask, replacement, output)
            return output

        embedding.forward = patched_forward  # type: ignore[method-assign]

    def parameters(self) -> list[nn.Parameter]:
        return [self.parameter] if self.parameter is not None else []

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "placeholder_tokens": self.placeholder_tokens,
            "placeholder_ids": self.placeholder_ids,
            "initializer_token": self.initializer_token,
            "trainable": self.trainable,
            "embedding": self.current_embedding().detach().cpu(),
        }

    def load_embedding(self, tensor: torch.Tensor) -> None:
        if tuple(tensor.shape) != tuple(self.current_embedding().shape):
            raise ValueError(
                f"Checkpoint embedding for {self.name} has shape {tuple(tensor.shape)}, "
                f"expected {tuple(self.current_embedding().shape)}."
            )
        tensor = tensor.to(device=self.device, dtype=torch.float32)
        if self.parameter is not None:
            with torch.no_grad():
                self.parameter.copy_(tensor)
        else:
            embedding = self.text_encoder.get_input_embeddings()
            with torch.no_grad():
                for row, token_id in enumerate(self.placeholder_ids):
                    embedding.weight.data[token_id].copy_(tensor[row].to(embedding.weight.dtype))

    def current_embedding(self) -> torch.Tensor:
        if self.parameter is not None:
            return self.parameter
        embedding = self.text_encoder.get_input_embeddings()
        return embedding.weight.detach()[self.placeholder_ids].float()

    def regularization(self) -> torch.Tensor:
        if self.parameter is None or self.initial_parameter is None:
            return torch.zeros((), device=self.device)
        return torch.mean((self.parameter - self.initial_parameter.to(self.parameter.device)).pow(2))

    def norm(self) -> torch.Tensor:
        return torch.linalg.vector_norm(self.current_embedding().reshape(-1))

    def displacement(self) -> torch.Tensor:
        if self.initial_parameter is None:
            return torch.zeros((), device=self.device)
        current = self.current_embedding()
        return torch.linalg.vector_norm((current - self.initial_parameter.to(current.device)).reshape(-1))
