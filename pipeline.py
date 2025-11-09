"""
title: SiliconFlow Manifold Pipeline
author: ivanreeve
author_url: https://github.com/ivanreeve/
project_url: https://github.com/ivanreeve/siliconflow-manifold
version: 0.9.0
license: Apache License 2.0
description: A comprehensive Open WebUI manifold pipe for seamless integration with SiliconFlow's API, providing dynamic access to text chat models, text-to-image generation, and image-to-image editing capabilities. Automatically discovers and caches available models, handles authentication, and manages streaming responses with built-in regional failover support.
features:
  - Dynamic model discovery: Automatically fetches and caches available chat and image models from SiliconFlow (10-minute TTL)
  - Text chat support: Full SSE streaming for real-time chat completions with system message handling and multi-part content extraction
  - Text-to-image generation: Creates images from prompts with model-specific defaults (Qwen: 1328x1328, Kolors: 1024x1024)
  - Image-to-image editing: Supports Qwen-Image-Edit models with multi-image input capability (up to 3 images)
  - Model-specific parameters: Intelligent handling of batch_size, guidance_scale, cfg, negative_prompt, seed, and inference steps
  - Streaming response handling: Preserves native SSE formatting with proper data: prefixes and [DONE] markers
  - Regional failover: Automatic endpoint switching between api.siliconflow.com and api.siliconflow.cn for global accessibility
  - Robust error handling: Graceful request exception management with timeout protection (connect: 3.05s, read: 10-300s)
  - Bearer token authentication: Secure API key management via environment variables
  - Message normalization: Converts Open WebUI format to SiliconFlow API specification
"""

import os
import time
import requests
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message


class Pipe:
    class Valves(BaseModel):
        API_KEY: str = Field(default="")

    def __init__(self):
        self.type = "manifold"
        self.id = "siliconflow"
        self.name = "SiliconFlow/"
        self.valves = self.Valves(API_KEY=os.getenv("API_KEY", ""))
        self._base = "https://api.siliconflow.com/v1"  # switch to .cn if your key is China region
        self._models_url = f"{self._base}/models"
        self._chat_url = f"{self._base}/chat/completions"
        self._images_url = f"{self._base}/images/generations"

        self._allow_cache: dict = {"text": set(), "img_gen": set(), "img_edit": set()}
        self._allow_cache_ts: float = 0.0
        self._allow_cache_ttl: float = 600.0

    def _auth_headers(self, stream: bool = False, json_accept: bool = True) -> dict:
        h = {
            "Authorization": f"Bearer {self.valves.API_KEY}",
            "Content-Type": "application/json",
        }
        if stream:
            h["Accept"] = "text/event-stream"
        elif json_accept:
            h["Accept"] = "application/json"
        return h

    def _normalize_model_id(self, model: str) -> str:
        for p in ("siliconflow/", "SiliconFlow/", "siliconflow.", "SiliconFlow."):
            while model.startswith(p):
                model = model[len(p) :]
        return model

    def _refresh_allowed_models(self) -> None:
        if time.time() - self._allow_cache_ts < self._allow_cache_ttl:
            return
        text_ids, img_gen_ids, img_edit_ids = set(), set(), set()
        try:
            r1 = requests.get(
                self._models_url,
                params={"type": "text", "sub_type": "chat"},
                headers=self._auth_headers(),
                timeout=(3.05, 10),
            )
            if r1.ok:
                for m in (r1.json() or {}).get("data", []):
                    mid = m.get("id")
                    if mid:
                        text_ids.add(mid)

            r2 = requests.get(
                self._models_url,
                params={"type": "image", "sub_type": "text-to-image"},
                headers=self._auth_headers(),
                timeout=(3.05, 10),
            )
            if r2.ok:
                for m in (r2.json() or {}).get("data", []):
                    mid = m.get("id")
                    if mid:
                        img_gen_ids.add(mid)

            r3 = requests.get(
                self._models_url,
                params={"type": "image", "sub_type": "image-to-image"},
                headers=self._auth_headers(),
                timeout=(3.05, 10),
            )
            if r3.ok:
                for m in (r3.json() or {}).get("data", []):
                    mid = m.get("id")
                    if mid:
                        img_edit_ids.add(mid)

            self._allow_cache = {
                "text": text_ids,
                "img_gen": img_gen_ids,
                "img_edit": img_edit_ids,
            }
            self._allow_cache_ts = time.time()
        except Exception:
            pass

    def _is_allowed_text(self, model_id: str) -> bool:
        self._refresh_allowed_models()
        return model_id in self._allow_cache["text"]

    def _is_allowed_image_gen(self, model_id: str) -> bool:
        self._refresh_allowed_models()
        return model_id in self._allow_cache["img_gen"]

    def _is_allowed_image_edit(self, model_id: str) -> bool:
        self._refresh_allowed_models()
        return model_id in self._allow_cache["img_edit"]

    def _fetch_models_filtered(self) -> List[dict]:
        """Return ONLY chat + image models for display in Open WebUI."""
        self._refresh_allowed_models()
        out = []
        for mid in sorted(self._allow_cache["text"]):
            out.append({"id": mid, "name": mid})
        for mid in sorted(self._allow_cache["img_gen"] | self._allow_cache["img_edit"]):
            out.append({"id": mid, "name": mid})
        return out

    def pipes(self) -> List[dict]:
        return self._fetch_models_filtered()

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        model_id = self._normalize_model_id(body["model"])

        if not (
            self._is_allowed_text(model_id)
            or self._is_allowed_image_gen(model_id)
            or self._is_allowed_image_edit(model_id)
        ):
            return (
                f"Error: model '{model_id}' is not allowed. "
                f"Allowed categories: chat and image models only."
            )

        if self._is_allowed_image_gen(model_id) or self._is_allowed_image_edit(
            model_id
        ):
            return self._image_pipe(model_id, body)

        system_message, messages = pop_system_message(body["messages"])

        processed_messages = []
        if system_message:
            if isinstance(system_message, dict):
                processed_messages.append(
                    {"role": "system", "content": system_message.get("content", "")}
                )
            else:
                processed_messages.append(
                    {"role": "system", "content": str(system_message)}
                )

        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                parts = []
                for item in content:
                    t = item.get("text") or item.get("input_text")
                    if t:
                        parts.append(t)
                content = "".join(parts)
            processed_messages.append({"role": message["role"], "content": content})

        payload = {
            "model": model_id,
            "messages": processed_messages,
            "stream": body.get("stream", False),
        }
        for k in ("user", "chat_id", "title"):
            payload.pop(k, None)

        try:
            if payload["stream"]:
                return self.stream_response(
                    self._chat_url, self._auth_headers(stream=True), payload
                )
            return self.non_stream_response(
                self._chat_url, self._auth_headers(), payload
            )
        except requests.exceptions.RequestException as e:
            return f"Error: Request failed: {e}"
        except Exception as e:
            return f"Error: {e}"

    def _last_user_text(self, messages: List[dict]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                c = m.get("content", "")
                if isinstance(c, str):
                    return c
                if isinstance(c, list):
                    parts = [
                        it.get("text")
                        for it in c
                        if it.get("type") == "text" and it.get("text")
                    ]
                    if parts:
                        return "\n".join(parts)
        return ""

    def _extract_image_inputs(self, body_messages: List[dict]) -> List[str]:
        urls = []
        for m in body_messages:
            c = m.get("content")
            if isinstance(c, list):
                for it in c:
                    if it.get("type") == "image_url" and it.get("image_url", {}).get(
                        "url"
                    ):
                        urls.append(it["image_url"]["url"])
        return urls

    def _image_pipe(self, model_id: str, body: dict) -> Union[str, dict]:
        """
        SiliconFlow /images/generations
        - text-to-image: require image_size
        - image-to-image (Qwen-Image-Edit*): do NOT send image_size; require input image
        - 'image' accepts a data URI or http(s) URL
        """
        prompt = self._last_user_text(body["messages"])
        is_edit = self._is_allowed_image_edit(model_id)
        is_qwen_image = model_id.startswith("Qwen/Qwen-Image")
        is_kolors = model_id.startswith("Kwai-Kolors/Kolors")

        payload = {"model": model_id, "prompt": prompt or body.get("prompt", "")}

        if is_edit:
            srcs = self._extract_image_inputs(body["messages"])
            if not srcs:
                return "Error: this image-to-image model requires an input image."
            payload["image"] = srcs[0]
            if model_id == "Qwen/Qwen-Image-Edit-2509":
                if len(srcs) > 1:
                    payload["image2"] = srcs[1]
                if len(srcs) > 2:
                    payload["image3"] = srcs[2]
        else:
            image_size = body.get("image_size")
            if not image_size:
                if is_qwen_image:
                    image_size = "1328x1328"
                elif is_kolors:
                    image_size = "1024x1024"
                else:
                    image_size = "1024x1024"
            payload["image_size"] = image_size

        if "negative_prompt" in body:
            payload["negative_prompt"] = body["negative_prompt"]
        if is_kolors:
            if "batch_size" in body:
                payload["batch_size"] = body["batch_size"]
            if "guidance_scale" in body:
                payload["guidance_scale"] = body["guidance_scale"]
        if "seed" in body:
            payload["seed"] = body["seed"]
        if "num_inference_steps" in body:
            payload["num_inference_steps"] = body["num_inference_steps"]
        if is_qwen_image and not is_edit:
            if "cfg" in body:
                payload["cfg"] = body["cfg"]

        try:
            resp = requests.post(
                self._images_url,
                headers=self._auth_headers(),
                json=payload,
                timeout=(3.05, 90),
            )
            if resp.status_code == 404 and self._base.endswith(".com/v1"):
                # region fallback if needed
                alt = "https://api.siliconflow.cn/v1/images/generations"
                resp = requests.post(
                    alt, headers=self._auth_headers(), json=payload, timeout=(3.05, 90)
                )
            resp.raise_for_status()
            data = resp.json() or {}
            images = [it.get("url") for it in data.get("images", []) if it.get("url")]
            if not images:
                return f"Error: No image URL in response: {data}"
            if len(images) == 1:
                return f"Here is your image:\n\n![result]({images[0]})"
            md = ["Here are your images:"]
            for i, u in enumerate(images, 1):
                md.append(f"\n\n**#{i}**\n\n![result {i}]({u})")
            return "".join(md)
        except requests.exceptions.RequestException as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error: {e}"

    def stream_response(self, url, headers, payload):
        """
        Forward native SSE to Open WebUI without stripping the 'data:' prefix.
        """
        try:
            with requests.post(
                url, json=payload, headers=headers, stream=True, timeout=(3.05, 300)
            ) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines():  # bytes; do not pre-decode
                    if raw is None:
                        continue
                    if raw.startswith(b":"):
                        yield raw + b"\n\n"
                        continue
                    line = raw if raw.startswith(b"data:") else b"data: " + raw
                    yield line + b"\n\n"
                    if line == b"data: [DONE]":
                        break
        except requests.exceptions.RequestException as e:
            yield f"data: Error: {e}\n\n".encode("utf-8")
        except Exception as e:
            yield f"data: Error: {e}\n\n".encode("utf-8")

    def non_stream_response(self, url, headers, payload):
        try:
            resp = requests.post(
                url=url, headers=headers, json=payload, timeout=(3.05, 60)
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error: {e}"
