import os
import json
import time
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


ENGINE = None
TOKENIZER = None
LOCK = threading.Lock()


def make_sampling_params_from_openai(obj):
    temperature = float(obj.get("temperature", 1.0))
    max_tokens = int(obj.get("max_tokens", 64))
    ignore_eos = bool(obj.get("ignore_eos", False))
    return SamplingParams(temperature=temperature, max_tokens=max_tokens, ignore_eos=ignore_eos)


def messages_to_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        p = urlparse(self.path)
        if p.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        p = urlparse(self.path)
        if p.path != "/v1/chat/completions":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            req = json.loads(body.decode("utf-8"))
        except Exception:
            self._send_json(400, {"error": "invalid json"})
            return
        messages = req.get("messages")
        if not isinstance(messages, list):
            self._send_json(400, {"error": "messages must be a list"})
            return
        stream = bool(req.get("stream", False))
        model = req.get("model", "")
        sp = make_sampling_params_from_openai(req)
        created = int(time.time())
        comp_id = "chatcmpl-" + uuid.uuid4().hex
        if not stream:
            try:
                with LOCK:
                    prompt = messages_to_prompt(TOKENIZER, messages)
                    outputs = ENGINE.generate([prompt], sp, use_tqdm=False)
                text = outputs[0]["text"]
                prompt_len = len(TOKENIZER.encode(prompt))
                completion_len = len(ENGINE.tokenizer.encode(text))
                resp = {
                    "id": comp_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_len,
                        "completion_tokens": completion_len,
                        "total_tokens": prompt_len + completion_len,
                    },
                }
                self._send_json(200, resp)
            except Exception as e:
                self._send_json(500, {"error": str(e)})
        else:
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                with LOCK:
                    prompt = messages_to_prompt(TOKENIZER, messages)
                    ENGINE.add_request(prompt, sp)
                    last_count = 0
                    while not ENGINE.is_finished():
                        output, _ = ENGINE.step()
                        running = list(ENGINE.scheduler.running)
                        if running:
                            seq = running[0]
                            cur = seq.num_completion_tokens
                            if cur > last_count:
                                delta_tokens = seq.completion_token_ids[last_count:cur]
                                delta_text = ENGINE.tokenizer.decode(delta_tokens)
                                chunk = {
                                    "id": comp_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model,
                                    "choices": [
                                        {"index": 0, "delta": {"role": "assistant", "content": delta_text}, "finish_reason": None}
                                    ],
                                }
                                self.wfile.write(b"data: " + json.dumps(chunk, ensure_ascii=False).encode("utf-8") + b"\n\n")
                                self.wfile.flush()
                                last_count = cur
                        for sid, toks in output:
                            chunk = {
                                "id": comp_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [
                                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                                ],
                            }
                            self.wfile.write(b"data: " + json.dumps(chunk, ensure_ascii=False).encode("utf-8") + b"\n\n")
                            self.wfile.flush()
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except Exception:
                    pass
            except Exception as e:
                try:
                    self._send_json(500, {"error": str(e)})
                except Exception:
                    pass

    def log_message(self, format, *args):
        return


def run_server(host=None, port=None):
    global ENGINE, TOKENIZER
    model_path = os.path.expanduser(os.environ.get("MODEL_PATH", "~/huggingface/Qwen3-0.6B/"))
    tp_size = int(os.environ.get("TP_SIZE", "1"))
    enforce_eager = os.environ.get("ENFORCE_EAGER", "1") not in ("0", "false", "False")
    host = host or os.environ.get("HOST", "0.0.0.0")
    port = int(port or os.environ.get("PORT", "8001"))
    TOKENIZER = AutoTokenizer.from_pretrained(model_path)
    ENGINE = LLM(model_path, enforce_eager=enforce_eager, tensor_parallel_size=tp_size)
    server = ThreadingHTTPServer((host, port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    run_server()
