#!/usr/bin/env python3
"""
LLM Inference Server - HTTP server for local model inference.

Loads a HuggingFace model once, serves requests via HTTP.
Supports LoRA adapters, tool/function calling, and extensible hooks.

Usage:
    python llm_server.py --model path/to/model --port 8765
    python llm_server.py --model path/to/model --adapter path/to/lora

API:
    POST /generate
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 256,
        "temperature": 0.7,
        "tools": [...]  // optional
        "system_prompt": "...",  // optional
        "system_context": "..."  // optional, injected into system prompt
    }

    GET /health
    Returns {"status": "ok", "model": "..."}

Extensibility:
    The server accepts system_context in the request body, which is appended
    to the system prompt. This enables stateless operation - the caller gathers
    context and passes it with each request.
"""

import argparse
import json
import re
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Callable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional: LoRA support
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class LLMServer:
    """
    HTTP-served LLM inference with support for LoRA and tools.

    Pure stateless inference server. The caller provides:
    - system_prompt: Base system prompt
    - system_context: Dynamic context to append (e.g., system facts)

    Both are optional - defaults to a basic assistant prompt.
    """

    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: str = "auto",
        default_system_prompt: Optional[str] = None,
        default_tools: Optional[list] = None,
    ):
        print(f"Loading model from {model_path}...", file=sys.stderr)

        # Determine dtype
        if dtype == "auto":
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device or "auto"
        )

        # Load LoRA adapter if specified
        if adapter_path and Path(adapter_path).exists():
            if not PEFT_AVAILABLE:
                print("Warning: peft not installed, cannot load LoRA adapter", file=sys.stderr)
            else:
                print(f"Loading LoRA adapter from {adapter_path}...", file=sys.stderr)
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
                print("LoRA adapter loaded.", file=sys.stderr)

        self.device = next(self.model.parameters()).device
        self.model_path = model_path
        self.default_system_prompt = default_system_prompt or "You are a helpful assistant."
        self.default_tools = default_tools
        print(f"Model loaded on {self.device}.", file=sys.stderr)

        # Ensure padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def extract_response(self, raw_output: str) -> dict:
        """
        Parse model output into structured response.

        Handles:
        - <think>...</think> blocks (reasoning)
        - <tool_call>...</tool_call> (function calls)
        - Plain text responses
        """
        # Extract thinking content if present
        thinking = None
        think_match = re.search(r'<think>(.*?)</think>', raw_output, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            if not thinking:
                thinking = None

        # Check for XML-style tool call: <tool_call>{"name": ..., "arguments": ...}</tool_call>
        tool_call_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', raw_output, re.DOTALL)
        if tool_call_match:
            try:
                tool_data = json.loads(tool_call_match.group(1))
                tool_name = tool_data.get("name", "")
                arguments = tool_data.get("arguments", {})

                result = {
                    "success": True,
                    "type": "tool_call",
                    "tool_name": tool_name,
                    "arguments": arguments,
                }
                if thinking:
                    result["thinking"] = thinking
                return result
            except json.JSONDecodeError:
                pass

        # Check for SmolLM3-style: call:function_name{param:value}
        smol_match = re.search(r'call:(\w+)\{([^}]+)\}', raw_output, re.DOTALL)
        if smol_match:
            func_name = smol_match.group(1)
            args_str = smol_match.group(2)

            # Parse simple key:value format
            arguments = {}
            for match in re.finditer(r'(\w+):<escape>(.*?)<escape>', args_str, re.DOTALL):
                arguments[match.group(1)] = match.group(2).strip()

            result = {
                "success": True,
                "type": "tool_call",
                "tool_name": func_name,
                "arguments": arguments,
            }
            if thinking:
                result["thinking"] = thinking
            return result

        # Natural language response - strip XML tags but preserve content
        text = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', '', text).strip()

        result = {"success": True, "type": "text", "response": text}
        if thinking:
            result["thinking"] = thinking
        return result

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: Optional[list] = None,
        system_prompt: Optional[str] = None,
        system_context: Optional[str] = None,
    ) -> dict:
        """Generate response for a conversation.

        Args:
            messages: Conversation history
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            tools: Tool definitions for function calling
            system_prompt: Base system prompt
            system_context: Dynamic context to append to system prompt
        """
        try:
            # Build system prompt with optional context
            base_prompt = system_prompt or self.default_system_prompt
            if system_context:
                full_system_prompt = f"{base_prompt}\n\n{system_context}"
            else:
                full_system_prompt = base_prompt

            # Build message list
            full_messages = [{"role": "system", "content": full_system_prompt}]
            full_messages.extend(messages)

            # Apply chat template
            template_kwargs = {
                "add_generation_prompt": True,
                "return_dict": True,
                "return_tensors": "pt"
            }

            active_tools = tools or self.default_tools
            if active_tools:
                template_kwargs["tools"] = active_tools

            inputs = self.tokenizer.apply_chat_template(
                full_messages,
                **template_kwargs
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode only the generated tokens
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

            # Parse response
            result = self.extract_response(raw_output)

            return result

        except Exception as e:
            return {"success": False, "error": str(e)}


# Global server instance
llm_server: Optional[LLMServer] = None


class RequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_POST(self):
        if self.path == "/health":
            self._send_json({
                "status": "ok",
                "model": llm_server.model_path if llm_server else None
            })
            return

        if self.path not in ["/generate", "/query"]:  # Support both endpoints
            self._send_json({"error": "Not found"}, 404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)

            messages = data.get("messages", [])
            max_tokens = data.get("max_tokens", 256)
            temperature = data.get("temperature", 0.7)
            top_p = data.get("top_p", 0.9)
            tools = data.get("tools")
            system_prompt = data.get("system_prompt")
            system_context = data.get("system_context")

            result = llm_server.generate(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=tools,
                system_prompt=system_prompt,
                system_context=system_context,
            )

            self._send_json(result)

        except json.JSONDecodeError as e:
            self._send_json({"success": False, "error": f"Invalid JSON: {e}"}, 400)
        except Exception as e:
            self._send_json({"success": False, "error": str(e)}, 500)

    def do_GET(self):
        if self.path == "/health":
            self._send_json({
                "status": "ok",
                "model": llm_server.model_path if llm_server else None
            })
        else:
            self._send_json({"error": "Use POST /generate"}, 405)

    def log_message(self, format, *args):
        print(f"[{self.address_string()}] {args[0]}", file=sys.stderr)


def run_server(
    server_instance: LLMServer,
    host: str = "127.0.0.1",
    port: int = 8765,
):
    """Run the HTTP server with the given LLMServer instance."""
    global llm_server
    llm_server = server_instance

    server = HTTPServer((host, port), RequestHandler)
    print(f"Server listening on http://{host}:{port}", file=sys.stderr)
    print(f"  POST /generate - Generate text", file=sys.stderr)
    print(f"  POST /query    - Alias for /generate", file=sys.stderr)
    print(f"  GET  /health   - Health check", file=sys.stderr)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", file=sys.stderr)
        server.shutdown()


def main():
    global llm_server

    parser = argparse.ArgumentParser(description="LLM HTTP Inference Server")
    parser.add_argument("--model", "-m", required=True, help="Model path or HuggingFace ID")
    parser.add_argument("--adapter", "-a", default=None, help="LoRA adapter path (optional)")
    parser.add_argument("--port", "-p", type=int, default=8765, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--device", "-d", default=None, help="Device (cuda, cpu, auto)")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"],
                        help="Model dtype")
    parser.add_argument("--system-prompt", "-s", default=None, help="Default system prompt")
    parser.add_argument("--tools-json", "-t", default=None, help="JSON file with default tool definitions")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists() and "/" not in args.model:
        print(f"Error: Model not found at {args.model}", file=sys.stderr)
        sys.exit(1)

    # Load tools if specified
    default_tools = None
    if args.tools_json:
        with open(args.tools_json) as f:
            default_tools = json.load(f)
        print(f"Loaded {len(default_tools)} tool definitions", file=sys.stderr)

    llm_server = LLMServer(
        args.model,
        adapter_path=args.adapter,
        device=args.device,
        dtype=args.dtype,
        default_system_prompt=args.system_prompt,
        default_tools=default_tools,
    )

    run_server(llm_server, args.host, args.port)


if __name__ == "__main__":
    main()
