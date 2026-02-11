import os
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


ENGINE = None
TOKENIZER = None
LOCK = threading.Lock()


def make_sampling_params(obj):
    if isinstance(obj, list):
        return [make_sampling_params(x) for x in obj]
    if isinstance(obj, dict):
        return SamplingParams(
            temperature=float(obj.get("temperature", 1.0)),
            max_tokens=int(obj.get("max_tokens", 64)),
            ignore_eos=bool(obj.get("ignore_eos", False)),
        )
    return SamplingParams()


def format_prompts(tokenizer, prompts):
    if isinstance(prompts, str):
        prompts = [prompts]
    out = []
    for prompt in prompts:
        if isinstance(prompt, str):
            s = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            out.append(s)
        elif isinstance(prompt, list):
            out.append(prompt)
        else:
            out.append(str(prompt))
    return out


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status, obj):
        data = json.dumps(obj).encode("utf-8")
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
        if p.path not in ("/generate", "/generate_stream"):
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            req = json.loads(body.decode("utf-8"))
        except Exception:
            self._send_json(400, {"error": "invalid json"})
            return
        prompts = req.get("prompts")
        if prompts is None and "prompt" in req:
            prompts = req.get("prompt")
        if prompts is None:
            self._send_json(400, {"error": "missing prompts"})
            return
        sp = req.get("sampling_params", {})
        sp_obj = make_sampling_params(sp)
        if p.path == "/generate":
            try:
                with LOCK:
                    formatted = format_prompts(TOKENIZER, prompts)
                    outputs = ENGINE.generate(formatted, sp_obj, use_tqdm=False)
                for o in outputs:
                    o.pop("token_ids", None)
                self._send_json(200, {"outputs": outputs})
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
                    formatted = format_prompts(TOKENIZER, prompts)
                    sp_list = sp_obj if isinstance(sp_obj, list) else [sp_obj] * len(formatted)
                    for prompt, sp_i in zip(formatted, sp_list):
                        ENGINE.add_request(prompt, sp_i)
                    last_counts = {}
                    while not ENGINE.is_finished():
                        output, _ = ENGINE.step()
                        running = list(ENGINE.scheduler.running)
                        for seq in running:
                            sid = seq.seq_id
                            prev = last_counts.get(sid, 0)
                            cur = seq.num_completion_tokens
                            if cur > prev:
                                delta_tokens = seq.completion_token_ids[prev:cur]
                                delta_text = ENGINE.tokenizer.decode(delta_tokens)
                                evt = json.dumps({"seq_id": sid, "text": delta_text, "finished": False}).encode("utf-8")
                                self.wfile.write(b"data: " + evt + b"\n\n")
                                self.wfile.flush()
                                last_counts[sid] = cur
                        for sid, toks in output:
                            text = ENGINE.tokenizer.decode(toks)
                            evt = json.dumps({"seq_id": sid, "text": text, "finished": True}).encode("utf-8")
                            self.wfile.write(b"data: " + evt + b"\n\n")
                            self.wfile.flush()
                try:
                    self.wfile.write(b"data: " + json.dumps({"done": True}).encode("utf-8") + b"\n\n")
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
    port = int(port or os.environ.get("PORT", "8000"))
    TOKENIZER = AutoTokenizer.from_pretrained(model_path)
    ENGINE = LLM(model_path, enforce_eager=enforce_eager, tensor_parallel_size=tp_size)
    server = ThreadingHTTPServer((host, port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    run_server()
