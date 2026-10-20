"""Local browser demo for the trained multi-domain AI image detector."""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
from dataclasses import dataclass
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("GRADIO_TEMP_DIR", str(ROOT / ".gradio_tmp"))

import gradio as gr
from PIL import Image

from inference import DemoPredictor, PredictionResult, normalize_image


DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "dil_on_best.pt"
EVALUATION_PATH = ROOT / "metadata" / "evaluation.json"
EXAMPLES_PATH = ROOT / "examples" / "examples.json"
SHOWCASE_PATH = ROOT / "examples" / "showcase.json"


@dataclass(frozen=True)
class ModelSpec:
    """A selectable demo checkpoint and its matching benchmark record."""

    display_name: str
    checkpoint: Path
    evaluation: Path | None


MODEL_SPECS = {
    "gated": ModelSpec(
        display_name="Gated fusion",
        checkpoint=DEFAULT_CHECKPOINT,
        evaluation=EVALUATION_PATH,
    ),
    "concatenation": ModelSpec(
        display_name="Concatenation fusion",
        checkpoint=ROOT / "checkpoints" / "concatenation_best.pt",
        evaluation=ROOT / "metadata" / "evaluation_concatenation.json",
    ),
}

CSS = """
.gradio-container { max-width: 1180px !important; margin: 0 auto; }
.hero { text-align: center; margin: 1rem 0 1.5rem; }
.hero h1 { font-size: 2.15rem; margin-bottom: .3rem; }
.result-card { border-radius: 16px; padding: 1.2rem 1.4rem; text-align: center; }
.result-real { background: #ecfdf5; border: 1px solid #86efac; color: #14532d; }
.result-ai { background: #fff7ed; border: 1px solid #fdba74; color: #7c2d12; }
.result-card h2 { margin: 0 0 .35rem; font-size: 1.8rem; }
.result-card p { margin: .2rem 0; }
.caveat { border-left: 4px solid #f59e0b; padding: .75rem 1rem; background: #fffbeb; }
.domain-card { text-align: center; }
.domain-card h3 { margin: 0 0 .6rem; }
.domain-card img { width: 100%; max-height: 280px; object-fit: contain; border-radius: 12px; background: #111827; }
"""


class PredictorProtocol(Protocol):
    device: str

    def predict(self, image: Image.Image) -> PredictionResult: ...


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def result_html(result: PredictionResult) -> str:
    css_class = "result-ai" if result.predicted_label else "result-real"
    safe_prediction = html.escape(result.prediction)
    return (
        f'<div class="result-card {css_class}">'
        f"<h2>{safe_prediction}</h2>"
        f"<p><strong>{result.confidence:.1%} confidence</strong></p>"
        f"<p>Analyzed in {result.elapsed_ms:.0f} ms · threshold {result.threshold:.2f}</p>"
        "</div>"
    )


def image_html(image: Image.Image, title: str) -> str:
    """Embed an image directly so Gradio does not need temp-file serving."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        '<div class="domain-card">'
        f"<h3>{html.escape(title)}</h3>"
        f'<img alt="{html.escape(title)}" src="data:image/png;base64,{encoded}">'
        "</div>"
    )


def analyze_image(image: Image.Image | None, predictor: PredictorProtocol):
    if image is None:
        raise gr.Error("Please upload or select an image first.")
    try:
        normalized = normalize_image(image)
        result = predictor.predict(image)
    except (OSError, TypeError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc
    except RuntimeError as exc:
        message = str(exc)
        if "out of memory" in message.lower():
            message = "GPU memory is exhausted. Close other GPU applications and try again."
        raise gr.Error(f"The image could not be analyzed: {message}") from exc

    probabilities = {
        "Real": result.probability_real,
        "AI-generated": result.probability_ai,
    }
    return (
        result_html(result),
        probabilities,
        image_html(normalized, "Original image"),
        image_html(result.frequency_image, "Frequency spectrum"),
        image_html(result.noise_image, "Noise residuals"),
    )


def analyze_example(path: Path, predictor: PredictorProtocol):
    with Image.open(path) as image:
        return analyze_image(image.copy(), predictor)


def resolve_model_selection(model_name: str, checkpoint: Path | None) -> ModelSpec:
    """Resolve a registered model or a custom checkpoint override."""
    selected = MODEL_SPECS[model_name]
    if checkpoint is None:
        return selected
    return ModelSpec(
        display_name=f"Custom checkpoint ({checkpoint.name})",
        checkpoint=checkpoint,
        evaluation=None,
    )


def benchmark_markdown(model_name: str, evaluation_path: Path | None) -> str:
    if evaluation_path is None:
        return f"""
### Recorded held-out benchmark

No benchmark metadata is attached to **{model_name}**. Registered models show
their matching saved evaluation here; custom checkpoints do not inherit a
possibly incompatible benchmark.
"""

    evaluation = load_json(evaluation_path)
    overall = evaluation["overall"]
    return f"""
### Recorded held-out benchmark — {model_name}

- Balanced accuracy: **{overall['balanced_accuracy']:.2%}**
- F1 score: **{overall['f1']:.2%}**
- AUC: **{overall['auc']:.2%}**
- Decision threshold: **{overall['threshold']:.2f}**

These are aggregate results from the saved cross-generator evaluation, not a
guarantee for an individual upload. Performance varies by image source and
generator family.
"""


def build_app(
    predictor: PredictorProtocol,
    model_name: str = MODEL_SPECS["gated"].display_name,
    evaluation_path: Path | None = EVALUATION_PATH,
) -> gr.Blocks:
    examples_metadata = load_json(EXAMPLES_PATH)
    showcase_metadata = load_json(SHOWCASE_PATH)
    with gr.Blocks(
        title="AI Image Authenticity Detector",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
        css=CSS,
    ) as demo:
        gr.HTML(
            """
            <div class="hero">
              <h1>AI Image Authenticity Detector</h1>
              <p>See how one model examines spatial, frequency, and noise patterns.</p>
            </div>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Image to analyze",
                    type="pil",
                    image_mode="RGB",
                    sources=["upload", "clipboard"],
                    height=390,
                )
                with gr.Row():
                    analyze_button = gr.Button(
                        "Analyze Image", variant="primary", size="lg"
                    )
                    clear_button = gr.ClearButton(value="Clear", components=[input_image])

            with gr.Column(scale=1):
                result = gr.HTML(
                    '<div class="result-card"><h2>Ready to analyze</h2>'
                    "<p>Upload an image or select an example.</p></div>"
                )
                probabilities = gr.Label(
                    label="Model probabilities", num_top_classes=2
                )
                gr.Markdown(
                    f"Running **{model_name}** on **{predictor.device.upper()}**. "
                    "The model is loaded once at startup."
                )

        gr.Markdown("## What the model examines")
        with gr.Row():
            original_view = gr.HTML(
                '<div class="domain-card"><h3>Original image</h3><p>Analyze an image to populate this view.</p></div>'
            )
            frequency_view = gr.HTML(
                '<div class="domain-card"><h3>Frequency spectrum</h3><p>Waiting for an image.</p></div>'
            )
            noise_view = gr.HTML(
                '<div class="domain-card"><h3>Noise residuals</h3><p>Waiting for an image.</p></div>'
            )

        gr.Markdown("### Held-out examples")
        gr.Markdown("Run a fixed example directly; the known source label is shown on each button.")
        example_buttons = []
        with gr.Row():
            for item in examples_metadata:
                architecture = item["architecture"].replace("_", " ").title()
                known_label = item["known_label"]
                example_buttons.append(
                    (
                        gr.Button(f"{architecture} · known {known_label}"),
                        ROOT / item["file"],
                    )
                )

        with gr.Accordion(
            "Curated showcase — 10 real and 10 AI-generated images", open=False
        ):
            gr.Markdown(
                "These held-out images were visually reviewed and screened for "
                "correct predictions by both registered checkpoints. They are "
                "presentation examples, not an unbiased performance sample."
            )
            for known_label, heading in (
                ("real", "Known real"),
                ("AI-generated", "Known AI-generated"),
            ):
                gr.Markdown(f"#### {heading}")
                label_items = [
                    item
                    for item in showcase_metadata["items"]
                    if item["known_label"] == known_label
                ]
                for start in range(0, len(label_items), 5):
                    with gr.Row():
                        for item in label_items[start:start + 5]:
                            example_buttons.append(
                                (
                                    gr.Button(item["display_name"]),
                                    ROOT / item["file"],
                                )
                            )

        gr.HTML(
            """
            <div class="caveat">
              <strong>Research demo:</strong> This output is a model estimate,
              not proof that an image is authentic or AI-generated. Cropping,
              compression, editing, and unseen generators can change the result.
            </div>
            """
        )

        with gr.Accordion("How it works and benchmark details", open=False):
            gr.Markdown(
                """
The detector processes the same image through three branches: normal RGB
content, an FFT frequency spectrum, and SRM high-pass noise residuals. It
compares and fuses those features before producing two-class probabilities.
"""
            )
            gr.Markdown(benchmark_markdown(model_name, evaluation_path))

        outputs = [result, probabilities, original_view, frequency_view, noise_view]
        analyze_button.click(
            fn=lambda image: analyze_image(image, predictor),
            inputs=[input_image],
            outputs=outputs,
            api_name="analyze",
            concurrency_limit=1,
            show_progress="full",
        )
        for index, (button, example_path) in enumerate(example_buttons):
            button.click(
                fn=partial(analyze_example, example_path, predictor),
                inputs=None,
                outputs=outputs,
                api_name=False,
                concurrency_limit=1,
                show_progress="full",
            )
        clear_button.add(outputs)

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_SPECS),
        default="gated",
        help="Registered model to run (default: gated)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Custom checkpoint override; suppresses registered benchmark metadata",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = resolve_model_selection(args.model, args.checkpoint)
    print(f"Selected model: {selected.display_name}")
    print(f"Loading checkpoint: {selected.checkpoint}")
    predictor = DemoPredictor(selected.checkpoint, device=args.device)
    print(f"Model ready on {predictor.device.upper()} (threshold={predictor.threshold:.2f})")
    demo = build_app(
        predictor,
        model_name=selected.display_name,
        evaluation_path=selected.evaluation,
    )
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        inbrowser=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
