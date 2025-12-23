from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

from excel_output import save_to_excel
from system_extractor_core import ExtractorConfig, extract_system_pdf_with_config


def _default_system_pdfs(root: Path) -> List[Path]:
    pdfs: List[Path] = []
    for letter in ["A", "B", "C", "D", "E", "F"]:
        p = root / f"SYSTEM {letter}.pdf"
        if p.exists():
            pdfs.append(p)
    return pdfs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--out", type=str, default=str(Path(__file__).resolve().parent / "outputs" / "system_af.xlsx"))
    parser.add_argument("--pdf", action="append", default=None)
    parser.add_argument("--debug-equipment", type=str, default=None)
    parser.add_argument("--debug-page", type=int, default=None)
    parser.add_argument("--vision", action="store_true", help="Enable Vision API fallback for low-confidence cases")
    parser.add_argument("--vision-provider", choices=["OPENAI", "ANTHROPIC"], default=None)
    parser.add_argument("--vision-model", type=str, default=None)

    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    root = Path(args.root)

    if args.pdf:
        pdf_paths = [Path(p) for p in args.pdf]
    else:
        pdf_paths = _default_system_pdfs(root)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config = ExtractorConfig()

    if args.debug_equipment:
        config.debug_equipment = str(args.debug_equipment).strip()
    if args.debug_page:
        config.debug_page = int(args.debug_page)

    # Load from .env first (optional)
    try:
        from vision_clients import load_vision_config

        vcfg = load_vision_config()
        config.vision_enabled = vcfg.enabled
        config.vision_provider = vcfg.provider
        config.vision_min_confidence = vcfg.min_confidence
        if vcfg.provider == "OPENAI":
            config.vision_model = vcfg.openai_model
        else:
            config.vision_model = vcfg.anthropic_model
    except Exception:
        # fallback: if user set VISION_ENABLED without having deps
        config.vision_enabled = os.getenv("VISION_ENABLED", "0").strip() in {"1", "true", "TRUE", "yes", "YES"}

    # CLI overrides
    if args.vision:
        config.vision_enabled = True
    if args.vision_provider:
        config.vision_provider = args.vision_provider
    if args.vision_model:
        config.vision_model = args.vision_model

    if config.vision_enabled and not config.vision_model:
        if config.vision_provider.upper() == "OPENAI":
            config.vision_model = "gpt-4o-mini"
        else:
            config.vision_model = "claude-3-5-sonnet-20241022"

    all_data: Dict[str, List[dict]] = {}

    for pdf in pdf_paths:
        system_name = pdf.stem
        rows = extract_system_pdf_with_config(str(pdf), config)
        all_data[system_name] = rows

    save_to_excel(all_data, str(out_path))
    print(f"Saved: {out_path}")
    for k, v in all_data.items():
        print(f"{k}: {len(v)} rows")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
