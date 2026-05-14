# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "google-genai",
#   "tripo3d",
#   "python-dotenv",
# ]
# ///
"""
Pipeline: topic text -> Nano Banana 2 concept image -> Tripo image-to-3D -> GLB.
Run: uv run generate.py "neuron"
Then: python3 -m http.server  (in the project dir) and open viewer.html.
"""

import argparse
import asyncio
import shutil
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from tripo3d import TripoClient, TaskStatus

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

CONCEPT_PROMPT = (
    "Educational scientific illustration of a {topic}, single isolated "
    "subject centered on a plain neutral background, clean 3D-friendly "
    "rendering, three-quarter view, soft even lighting, no text, no labels, "
    "no arrows, no annotations."
)


async def run(topic: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    concept_path = out_dir / "concept.png"

    print(f"[1/3] Generating concept image for '{topic}' via Nano Banana 2 …")
    gemini = genai.Client()
    response = gemini.models.generate_content(
        model="gemini-3.1-flash-image-preview",
        contents=CONCEPT_PROMPT.format(topic=topic),
    )

    image_part = next(
        (p for p in response.parts if p.inline_data is not None),
        None,
    )
    if image_part is None:
        raise SystemExit("Gemini returned no image. Response was text-only.")
    concept_path.write_bytes(image_part.inline_data.data)
    print(f"      -> {concept_path}")

    async with TripoClient() as tripo:
        balance = await tripo.get_balance()
        print(f"      Tripo balance: {balance.balance} credits "
              f"(frozen: {balance.frozen})")

        print("[2/3] Submitting image-to-3D task …")
        task_id = await tripo.image_to_model(image=str(concept_path))
        print(f"      task_id: {task_id}")

        task = await tripo.wait_for_task(task_id, verbose=True)
        if task.status != TaskStatus.SUCCESS:
            raise SystemExit(f"Tripo task ended in status {task.status}")

        print("[3/3] Downloading model files …")
        downloaded = await tripo.download_task_models(task, str(out_dir))
        for kind, path in downloaded.items():
            if path:
                print(f"      {kind}: {path}")

    glb_source = None
    for key in ("pbr_model", "model"):
        path = downloaded.get(key)
        if path and path.endswith(".glb"):
            glb_source = path
            break
    if glb_source is None:
        raise SystemExit("No GLB file in Tripo output.")

    stable = out_dir / "model.glb"
    if stable.resolve() != Path(glb_source).resolve():
        shutil.copy(glb_source, stable)
    print(f"\nDone. Stable viewer path: {stable}")
    print("Serve with `python3 -m http.server` and open viewer.html.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("topic", help='e.g. "neuron"')
    parser.add_argument("--out", type=Path, default=HERE / "output")
    args = parser.parse_args()
    asyncio.run(run(args.topic, args.out))


if __name__ == "__main__":
    main()
