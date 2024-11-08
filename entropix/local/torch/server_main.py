import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, List, Literal, Optional, Tuple

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from entropix.local.config import MODEL_CONFIGS, get_model_params
from entropix.local.torch.torch_weights import load_weights_torch
from entropix.local.tokenizer import Tokenizer 
from entropix.local.torch.prompts import ChatCompletionRequest, Message, generate_chat_prompt
from entropix.local.torch.dslider_config import DEFAULT_DS_CONFIG
from entropix.local.torch.dslider import initialize_state, adaptive_dirichlet_step
from entropix.local.torch.model import xfmr
from entropix.local.torch.kvcache import KVCache
from entropix.local.torch.utils import precompute_freqs_cis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Metadata:
    def __init__(self):
        self.start_time = time.time()

class ModelRequest:
    def __init__(self, tokens: torch.Tensor, max_tokens: int, metadata: Metadata):
        self.tokens = tokens
        self.max_tokens = max_tokens
        self.metadata = metadata
        self.is_client_side_tokenization = False

class ModelManager:
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._is_ready = False
        self._warmup_lock = asyncio.Lock()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    async def initialize(
        self,
        model_size: str = "1B",
        model_path: Optional[Path] = None,
        tokenizer_path: str = "entropix/data/tokenizer.model",
    ):
        if self._is_ready:
            return

        async with self._warmup_lock:
            if self._is_ready:
                return

            logger.info(f"Initializing {model_size} model...")

            if model_size not in MODEL_CONFIGS:
                raise ValueError(f"Unknown model: {model_size}")

            self.config = MODEL_CONFIGS[model_size]
            self.model_params = get_model_params(self.config)
            self._xfmr_weights = load_weights_torch(model_size)
            self._tokenizer = Tokenizer(tokenizer_path)
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            await self._warmup()
            self._is_ready = True
            logger.info("Model initialization and warmup complete")

    async def _warmup(self):
        logger.info("Starting model warmup...")
        warmup_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>Hello<|eot_id|>"
        warmup_request = ModelRequest(
            tokens=self._tokenizer.encode(warmup_prompt, bos=False, eos=False, allowed_special='all'),
            max_tokens=10,
            metadata=Metadata()
        )

        try:
            async for _ in self._generate_response(warmup_request):
                pass
            logger.info("Warmup complete")
        except Exception as e:
            logger.error(f"Warmup failed: {e}")
            raise

    async def _generate_response(self, request: ModelRequest) -> AsyncGenerator[List[Tuple[str, List[int]]], None]:
        with torch.inference_mode():
            tokens = torch.tensor([request.tokens], dtype=torch.long).to(self.device)
            bsz, seqlen = tokens.shape
            cur_pos = 0
            
            # Initialize generation components
            freqs_cis = precompute_freqs_cis(
                self.model_params.head_dim,
                self.model_params.max_seq_len,
                self.model_params.rope_theta,
                self.model_params.use_scaled_rope
            )
            kvcache = KVCache.new(
                self.model_params.n_layers,
                bsz,
                self.model_params.max_seq_len,
                self.model_params.n_local_kv_heads,
                self.model_params.head_dim
            ).to(self.device)
            
            cfg = DEFAULT_DS_CONFIG.to(self.device)
            state = initialize_state(bsz, 128256, cfg, device=self.device)

            # First token generation
            logits, kvcache, scores, _ = xfmr(
                self._xfmr_weights,
                self.model_params,
                tokens,
                cur_pos,
                freqs_cis[:seqlen],
                kvcache
            )
            
            state, next_token, _ = adaptive_dirichlet_step(state, logits[:, -1], cfg)
            token_text = self._tokenizer.decode([next_token.item()])
            yield [(token_text, [next_token.item()])]

            gen_tokens = next_token
            cur_pos = seqlen
            stop = torch.tensor([128001, 128008, 128009], device=self.device, dtype=torch.int32)

            # Generate remaining tokens
            while cur_pos < request.max_tokens:
                cur_pos += 1
                logits, kvcache, scores, _ = xfmr(
                    self._xfmr_weights,
                    self.model_params,
                    next_token,
                    cur_pos,
                    freqs_cis[cur_pos:cur_pos+1],
                    kvcache
                )
                
                state, next_token, _ = adaptive_dirichlet_step(state, logits[:, -1], cfg)
                token_text = self._tokenizer.decode([next_token.item()])
                yield [(token_text, [next_token.item()])]

                if torch.isin(next_token, stop).any():
                    break

                gen_tokens = torch.cat((gen_tokens, next_token), dim=0)

app = FastAPI(title="Entropix Model Server")
model_manager = ModelManager()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def stream_response(request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    request_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    # Send initial response
    yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created, 'model': request.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

    prompt = generate_chat_prompt(request)
    tokens = model_manager._tokenizer.encode(prompt, bos=False, eos=False, allowed_special='all')
    model_request = ModelRequest(tokens=tokens, max_tokens=request.max_tokens, metadata=Metadata())

    try:
        async for token_batch in model_manager._generate_response(model_request):
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {"index": idx, "delta": {"content": text}, "finish_reason": None}
                    for idx, (text, _) in enumerate(token_batch)
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        # Send final chunk
        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    if not model_manager._is_ready:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return StreamingResponse(stream_response(request), media_type="text/event-stream")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model_manager._is_ready else "initializing",
        "model_initialized": model_manager._is_ready,
    }

@app.on_event("startup")
async def startup_event():
    await model_manager.initialize()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server_main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        workers=2
    )