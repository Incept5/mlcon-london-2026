#!/usr/bin/env python3
import os
import sys
import numpy as np
import gradio as gr
from llama_cpp import Llama
import plotly.graph_objects as go
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

os.environ.setdefault("GGML_METAL_LOG_LEVEL", "1")

DEFAULT_MODEL_PATH = "/Users/jdavies/.lmstudio/models/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf"
DEFAULT_GPU_LAYERS = -1
DEFAULT_PROMPT = "The capital of France is"


@dataclass
class InferenceConfig:
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    num_tokens: int = 10


class SuppressStderr:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr


class TokenProbabilityAnalyzer:
    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = 0):
        with SuppressStderr():
            self.model = Llama(
                model_path=model_path,
                logits_all=True,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
                chat_format="chatml"
            )
        self.debug_info = ""

    def analyze_next_tokens(self, system_prompt: str, user_prompt: str, config: InferenceConfig) -> List[Tuple[str, float, float]]:
        # Qwen3 thinking suppression: `/no_think` in system + pre-seeded empty <think></think> block
        formatted_prompt = "<|im_start|>system\n"
        if system_prompt.strip():
            formatted_prompt += system_prompt.strip()
        else:
            formatted_prompt += "You are a helpful assistant."
        formatted_prompt += " /no_think<|im_end|>\n"
        formatted_prompt += f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        formatted_prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n"

        self.debug_info = f"Chat mode - formatted prompt length: {len(formatted_prompt)} chars\n"

        output = self.model.create_completion(
            prompt=formatted_prompt, max_tokens=1, logprobs=20,
            temperature=1.0, top_p=1.0, top_k=-1,
            repeat_penalty=config.repeat_penalty, echo=False
        )

        result = self._extract_probabilities(output, config)

        # Workaround for intermittent llama-cpp bug: top_logprobs returns empty on finish_reason="length".
        # Retrying with a perturbed prompt (trailing space) and slightly different repeat_penalty usually succeeds.
        if not result and "choices" in output and output["choices"]:
            choice = output["choices"][0]
            if choice.get("finish_reason") == "length":
                self.debug_info += "\nAutomatically retrying due to empty logprobs bug...\n"
                output2 = self.model.create_completion(
                    prompt=formatted_prompt + " ", max_tokens=1, logprobs=20,
                    temperature=1.0, top_p=1.0, top_k=-1,
                    repeat_penalty=config.repeat_penalty * 0.99, echo=False
                )
                result = self._extract_probabilities(output2, config)
                if result:
                    self.debug_info += f"SUCCESS: Retry worked! Got {len(result)} tokens\n"
                else:
                    self.debug_info += "Retry also returned empty logprobs\n"

        return result

    def analyze_continuation(self, prompt: str, config: InferenceConfig) -> List[Tuple[str, float, float]]:
        self.debug_info = f"Analyzing continuation of: {repr(prompt[-50:])}\n"
        self.debug_info += f"Full prompt length: {len(prompt)} chars\n"

        output = self.model.create_completion(
            prompt=prompt, max_tokens=1, logprobs=20,
            temperature=1.0, top_p=1.0, top_k=-1,
            repeat_penalty=config.repeat_penalty, echo=False
        )

        if "choices" in output and output["choices"]:
            choice = output["choices"][0]
            self.debug_info += f"\nGenerated text: {repr(choice.get('text', ''))}\n"
            self.debug_info += f"Finish reason: {choice.get('finish_reason', 'None')}\n"
            logprobs = choice.get("logprobs", {})
            if isinstance(logprobs, dict):
                top_logprobs = logprobs.get("top_logprobs", [])
                self.debug_info += f"Top_logprobs length: {len(top_logprobs)}\n"
                if not top_logprobs:
                    self.debug_info += "WARNING: top_logprobs is empty! (common llama-cpp bug)\n"

        result = self._extract_probabilities(output, config)

        # Same empty-logprobs retry as Chat mode
        if not result and choice.get("finish_reason") == "length":
            self.debug_info += "\nAutomatically retrying due to empty logprobs bug...\n"
            output2 = self.model.create_completion(
                prompt=prompt + " ", max_tokens=1, logprobs=20,
                temperature=1.0, top_p=1.0, top_k=-1,
                repeat_penalty=config.repeat_penalty * 0.99, echo=False
            )
            if "choices" in output2 and output2["choices"]:
                choice2 = output2["choices"][0]
                logprobs2 = choice2.get("logprobs", {})
                if isinstance(logprobs2, dict) and logprobs2.get("top_logprobs"):
                    self.debug_info += f"SUCCESS: Retry worked!\n"
                    result = self._extract_probabilities(output2, config)
                else:
                    self.debug_info += "Retry also returned empty logprobs\n"

        return result

    def _extract_probabilities(self, output: dict, config: InferenceConfig) -> List[Tuple[str, float, float]]:
        if "choices" not in output or not output["choices"]:
            self.debug_info += "ERROR: No choices in output\n"
            return []

        choice = output["choices"][0]
        if choice.get("finish_reason") == "stop":
            self.debug_info += "Model returned stop token - end of generation\n"
            return []

        logprobs_data = choice.get("logprobs", {})
        if not logprobs_data:
            self.debug_info += "ERROR: No logprobs data returned\n"
            return []

        top_logprobs = logprobs_data.get("top_logprobs", [])
        if not top_logprobs:
            self.debug_info += "ERROR: Empty top_logprobs list\n"
            return []

        first_position = top_logprobs[0]
        if not first_position:
            return []

        token_logprobs = list(first_position.items())
        original_logprobs = {token: logprob for token, logprob in token_logprobs}

        if config.temperature != 1.0:
            token_logprobs = [(token, logprob / config.temperature) for token, logprob in token_logprobs]

        token_probs = [(token, float(np.exp(logprob)), original_logprobs[token]) for token, logprob in token_logprobs]
        token_probs.sort(key=lambda x: x[1], reverse=True)

        if config.top_k > 0:
            token_probs = token_probs[:config.top_k]

        if config.top_p < 1.0:
            token_probs = self._apply_top_p_filtering(token_probs, config.top_p)

        total_prob = sum(prob for _, prob, _ in token_probs)
        if total_prob > 0:
            token_probs = [(token, prob / total_prob, logprob) for token, prob, logprob in token_probs]

        return token_probs[:config.num_tokens]

    def _apply_top_p_filtering(self, token_probs: List[Tuple[str, float, float]], top_p: float) -> List[Tuple[str, float, float]]:
        token_probs.sort(key=lambda x: x[1], reverse=True)
        cumulative_prob = 0.0
        filtered_tokens = []
        for token, prob, logprob in token_probs:
            cumulative_prob += prob
            filtered_tokens.append((token, prob, logprob))
            if cumulative_prob >= top_p:
                break
        return filtered_tokens


current_analyzer: Optional[TokenProbabilityAnalyzer] = None
last_analysis_results: Optional[List[Tuple[str, float, float]]] = None


def load_model(model_path: str, context_size: int, gpu_layers: int) -> str:
    global current_analyzer, last_analysis_results

    if not model_path.strip():
        return "Error: Model path cannot be empty"

    try:
        current_analyzer = TokenProbabilityAnalyzer(
            model_path=model_path, n_ctx=context_size, n_gpu_layers=gpu_layers
        )
        last_analysis_results = None
        return f"✅ Model loaded: {Path(model_path).name}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def is_stop_token(token: str) -> bool:
    stop_tokens = {'</s>', '<|endoftext|>', '<|im_end|>', '<|end|>', '<|eot_id|>', '<eos>'}
    if token in stop_tokens:
        return True
    if len(token) == 1 and ord(token) < 32:
        return True
    return False


def analyze_tokens(mode: str, system_prompt: str, user_prompt: str, generate_prompt: str,
                   temperature: float, top_p: float, top_k: int, repeat_penalty: float, num_tokens: int) -> tuple:
    global current_analyzer, last_analysis_results

    if current_analyzer is None:
        return "Error: No model loaded", None, gr.Button(interactive=False)

    active_prompt = user_prompt if mode == "Chat" else generate_prompt
    if not active_prompt or not active_prompt.strip():
        return "Error: prompt is empty — type something before clicking Analyze", None, gr.Button(interactive=False)

    config = InferenceConfig(temperature, top_p, top_k, repeat_penalty, num_tokens)

    try:
        current_analyzer.debug_info = ""

        if mode == "Chat":
            token_probs = current_analyzer.analyze_next_tokens(system_prompt, user_prompt, config)
        else:
            token_probs = current_analyzer.analyze_continuation(generate_prompt, config)

        last_analysis_results = token_probs

        if not token_probs:
            debug_msg = "**No tokens found**\n\n"
            debug_msg += "**Debug Information:**\n```\n"
            debug_msg += current_analyzer.debug_info
            debug_msg += "```\n\n"
            debug_msg += "**Try clicking Analyze again** - this issue is intermittent and often works on retry."
            return debug_msg, None, gr.Button(interactive=False)

        formatted_results = format_results(token_probs)

        if current_analyzer.debug_info and "SUCCESS: Retry worked!" in current_analyzer.debug_info:
            formatted_results += "\n\n✅ _Auto-retry: Successfully recovered from empty logprobs bug_"

        pie_chart = create_pie_chart(token_probs)

        if token_probs and is_stop_token(token_probs[0][0]):
            button_state = gr.Button(interactive=False)
            formatted_results += "\n\n🔒 'Add Top Token' disabled - top token is a stop token"
        else:
            button_state = gr.Button(interactive=True)

        return formatted_results, pie_chart, button_state

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return error_msg, None, gr.Button(interactive=False)


def add_top_token(mode: str, system_prompt: str, user_prompt: str, generate_prompt: str,
                  chat_response: str, temperature: float, top_p: float, top_k: int,
                  repeat_penalty: float, num_tokens: int) -> tuple:
    global current_analyzer, last_analysis_results

    if not last_analysis_results:
        return "No previous analysis to use", None, gr.Button(interactive=False), generate_prompt, chat_response

    top_token = last_analysis_results[0][0]
    config = InferenceConfig(temperature, top_p, top_k, repeat_penalty, num_tokens)

    try:
        current_analyzer.debug_info = ""

        if mode == "Chat":
            new_chat_response = chat_response + top_token

            formatted_prompt = "<|im_start|>system\n"
            if system_prompt.strip():
                formatted_prompt += system_prompt.strip()
            else:
                formatted_prompt += "You are a helpful assistant."
            formatted_prompt += " /no_think<|im_end|>\n"
            formatted_prompt += f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            formatted_prompt += f"<|im_start|>assistant\n<think>\n\n</think>\n\n{new_chat_response}"

            output = current_analyzer.model.create_completion(
                prompt=formatted_prompt, max_tokens=1, logprobs=20,
                temperature=1.0, top_p=1.0, top_k=-1,
                repeat_penalty=config.repeat_penalty, echo=False
            )

            token_probs = current_analyzer._extract_probabilities(output, config)

            if not token_probs and "choices" in output and output["choices"]:
                choice = output["choices"][0]
                if choice.get("finish_reason") == "length":
                    output2 = current_analyzer.model.create_completion(
                        prompt=formatted_prompt + " ", max_tokens=1, logprobs=20,
                        temperature=1.0, top_p=1.0, top_k=-1,
                        repeat_penalty=config.repeat_penalty * 0.99, echo=False
                    )
                    token_probs = current_analyzer._extract_probabilities(output2, config)

            return_generate_prompt = generate_prompt
            return_chat_response = new_chat_response

        else:
            new_generate_prompt = generate_prompt + top_token
            token_probs = current_analyzer.analyze_continuation(new_generate_prompt, config)
            return_generate_prompt = new_generate_prompt
            return_chat_response = chat_response

        last_analysis_results = token_probs

        if not token_probs:
            formatted_results = "**No more tokens available**\n\nTry clicking 'Add Top Token' again."
            pie_chart = None
            button_state = gr.Button(interactive=False)
        else:
            formatted_results = format_results(token_probs)
            pie_chart = create_pie_chart(token_probs)
            if is_stop_token(token_probs[0][0]):
                button_state = gr.Button(interactive=False)
                formatted_results += "\n\n🔒 'Add Top Token' disabled - top token is a stop token"
            else:
                button_state = gr.Button(interactive=True)

        return formatted_results, pie_chart, button_state, return_generate_prompt, return_chat_response

    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return error_msg, None, gr.Button(interactive=False), generate_prompt, chat_response


def format_results(token_probs: List[Tuple[str, float, float]]) -> str:
    result = f"**Top {len(token_probs)} next token probabilities:**\n"
    result += "─" * 60 + "\n"

    for idx, (token, prob, logit) in enumerate(token_probs):
        token_repr = repr(token)
        percentage = prob * 100
        if is_stop_token(token):
            result += f"{idx + 1:2d}. {token_repr:<20} {percentage:6.2f}% 🛑 STOP (logit: {logit:7.3f})\n"
        else:
            result += f"{idx + 1:2d}. {token_repr:<20} {percentage:6.2f}%          (logit: {logit:7.3f})\n"

    total_prob = sum(prob for _, prob, _ in token_probs) * 100
    result += f"\n**Total probability:** {total_prob:.2f}%"
    return result


def create_pie_chart(token_probs: List[Tuple[str, float, float]]):
    if not token_probs:
        return None

    tokens = []
    probabilities = []
    colors = []

    for token, prob, logit in token_probs:
        token_display = repr(token).replace("'", "").replace('"', '')
        if len(token_display) > 15:
            token_display = token_display[:12] + "..."

        percentage = prob * 100
        if is_stop_token(token):
            token_display = f"🛑 {token_display} ({percentage:.1f}%)"
            colors.append('#ff4444')
        else:
            token_display = f"{token_display} ({percentage:.1f}%)"
            colors.append(None)

        tokens.append(token_display)
        probabilities.append(prob)

    fig = go.Figure(data=[go.Pie(
        labels=tokens, values=probabilities, textinfo='none',
        marker=dict(colors=colors) if any(colors) else None
    )])

    fig.update_layout(
        title="Token Probability Distribution",
        height=400, font=dict(size=12)
    )
    return fig


def create_gradio_interface():
    with gr.Blocks(title="Token Probability Analyzer") as app:
        with gr.Row():
            with gr.Column():
                gr.Markdown("# 🔍 Token Probability Analyzer")
                gr.Markdown("Analyze token probabilities from language models.")
            with gr.Column(scale=0):
                mode = gr.Radio(
                    choices=["Chat", "Generate"], value="Chat", label="Mode",
                    info="Chat: separate response window | Generate: add to existing text"
                )

        with gr.Tab("Token Analysis"):
            with gr.Row():
                with gr.Column():
                    system_prompt = gr.Textbox(label="System Prompt (Optional)", value="", lines=2,
                                                placeholder="System instructions...", visible=True)
                    user_prompt = gr.Textbox(label="User Prompt", value=DEFAULT_PROMPT, lines=2, visible=True)
                    chat_response = gr.Textbox(label="Assistant Response", value="", lines=3,
                                                placeholder="Response will appear here as you add tokens...", visible=True)
                    generate_prompt = gr.Textbox(label="Text to Continue", value=DEFAULT_PROMPT, lines=6,
                                                  placeholder="Text will grow here as you add tokens...", visible=False)

                    with gr.Accordion("Parameters", open=True):
                        with gr.Row():
                            temperature = gr.Slider(0.01, 2.0, value=0.8, label="Temperature")
                            num_tokens = gr.Slider(1, 20, value=10, step=1, label="Tokens to Show")
                        with gr.Row():
                            top_p = gr.Slider(0.01, 1.0, value=0.95, label="Top-p")
                            top_k = gr.Slider(1, 100, value=40, step=1, label="Top-k")
                        repeat_penalty = gr.Slider(0.5, 2.0, value=1.1, label="Repeat Penalty")

                    with gr.Row():
                        analyze_btn = gr.Button("🚀 Analyze", variant="primary")
                        add_token_btn = gr.Button("🔄 Add Top Token", variant="secondary")

                with gr.Column():
                    results = gr.Markdown("Click 'Analyze' to see results...")
                    pie_chart = gr.Plot()

        with gr.Tab("Model Configuration"):
            with gr.Row():
                with gr.Column():
                    model_path = gr.Textbox(label="Model Path",
                                             value=os.environ.get('GGUF_MODEL_PATH', DEFAULT_MODEL_PATH),
                                             placeholder="Path to .gguf file")
                    with gr.Row():
                        context_size = gr.Number(label="Context Size", value=2048)
                        gpu_layers = gr.Number(label="GPU Layers", value=DEFAULT_GPU_LAYERS)
                    load_btn = gr.Button("🔄 Load Model", variant="primary")
                with gr.Column():
                    load_status = gr.Textbox(label="Status", lines=4, interactive=False)

        def toggle_mode(mode):
            if mode == "Chat":
                return (gr.update(visible=True), gr.update(visible=True),
                        gr.update(visible=True), gr.update(visible=False))
            else:
                return (gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=False), gr.update(visible=True))

        mode.change(fn=toggle_mode, inputs=[mode],
                    outputs=[system_prompt, user_prompt, chat_response, generate_prompt])

        load_btn.click(fn=load_model, inputs=[model_path, context_size, gpu_layers],
                       outputs=[load_status])

        analyze_btn.click(
            fn=analyze_tokens,
            inputs=[mode, system_prompt, user_prompt, generate_prompt, temperature, top_p, top_k, repeat_penalty, num_tokens],
            outputs=[results, pie_chart, add_token_btn]
        )

        add_token_btn.click(
            fn=add_top_token,
            inputs=[mode, system_prompt, user_prompt, generate_prompt, chat_response, temperature, top_p, top_k, repeat_penalty, num_tokens],
            outputs=[results, pie_chart, add_token_btn, generate_prompt, chat_response]
        )

    return app


def main():
    print("Starting Token Probability Analyzer...")

    model_path = os.environ.get('GGUF_MODEL_PATH', DEFAULT_MODEL_PATH)

    if model_path and Path(model_path).exists():
        try:
            load_model(model_path, 2048, DEFAULT_GPU_LAYERS)
            print(f"✅ Model loaded from: {model_path}")
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}")
    else:
        print("⚠️ No model loaded - configure in the Model Configuration tab")

    port = os.environ.get('GRADIO_SERVER_PORT')

    app = create_gradio_interface()
    if port:
        app.launch(server_name="127.0.0.1", server_port=int(port), share=False)
    else:
        app.launch(server_name="127.0.0.1", share=False)


if __name__ == "__main__":
    main()
