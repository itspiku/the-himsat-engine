"""Minimal OpenAI-compatible chat server for open-weight models via Hugging Face transformers.

For single-box deployments without Ollama or vLLM:

    himsat llm-serve --model Qwen/Qwen2.5-7B-Instruct --port 8081
    HIMSAT_LLM_BASE_URL=http://localhost:8081/v1 HIMSAT_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct

It serves ``GET /v1/models`` and ``POST /v1/chat/completions`` (non-streaming), one request
at a time. For higher throughput use vLLM or Ollama, which speak the same API.
"""

from __future__ import annotations

import threading
import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class Msg(BaseModel):
    role: str
    content: str


class ChatReq(BaseModel):
    model: str | None = None
    messages: list[Msg]
    temperature: float = 0.2
    max_tokens: int = 1500
    response_format: dict | None = None


def create_app(model_id: str, device: str = "auto", dtype: str = "auto") -> FastAPI:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    torch_dtype = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype,
                                                 device_map=None if device == "cpu" else device)
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    lock = threading.Lock()
    app = FastAPI(title=f"HimSat local LLM ({model_id})")

    @app.get("/v1/models")
    def models() -> dict:
        return {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    def chat(req: ChatReq) -> dict:
        msgs = [m.model_dump() for m in req.messages]
        if req.response_format and req.response_format.get("type") == "json_object":
            msgs[-1]["content"] += "\n\nRespond with a single valid JSON object only."
        try:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception as e:  # pragma: no cover
            raise HTTPException(400, f"chat template failed: {e}") from e
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with lock, torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=req.max_tokens, do_sample=req.temperature > 0,
                                 temperature=max(req.temperature, 1e-3), top_p=0.9,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        new = out[0, inputs["input_ids"].shape[1]:]
        text = tok.decode(new, skip_special_tokens=True)
        return {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()),
                "model": model_id, "choices": [{"index": 0, "finish_reason": "stop",
                                                "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": int(inputs["input_ids"].shape[1]), "completion_tokens": int(new.shape[0])}}

    return app
