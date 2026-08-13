# Modal model isolation

Every catalog model is mapped to its own `modal.Image` from the same immutable
contract used by local virtual environments and HPC containers.

```bash
pip install modal==1.1.4
modal setup
python modal/run.py --model lingshu_7b --json
```

Import `image_for("lingshu_7b")` from `modal.images` in a Modal function. Model
weights are never baked into the image: mount a private volume at `/models`, a
Hugging Face cache at `/cache/huggingface`, and outputs at `/outputs`. Secrets
such as `HF_TOKEN` must be attached at deployment time and must not appear in
the contract or image history.

The model-specific Python/CUDA/dependency pins live in
`src/medumm/environments/catalog/models.yaml`; generated Docker, Apptainer and
requirements files live in `environments/models/<model>/`.
