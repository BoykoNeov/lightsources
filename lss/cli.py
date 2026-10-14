"""Command-line interface (design §13): ``lss refs | lampsheet | accuracy``."""

import argparse
import json
import sys
from pathlib import Path


def _cmd_refs(args) -> int:
    from lss.accuracy import references

    for ref in references.library().values():
        if args.family and ref.family != args.family:
            continue
        lo, hi = ref.valid_range
        print(f"{ref.id:<14} {ref.family:<13} {lo:g}-{hi:g} nm @ {ref.sampling_nm:g} nm  {ref.description}")
    return 0


def _cmd_lampsheet(args) -> int:
    from lss import lampsheet
    from lss.lamp import lamp as lamps

    lamp = lamps.load(args.lamp)
    levels = [args.level] if args.level is not None else sorted(lamp.models)
    sheets = [lampsheet.make(lamp, lv) for lv in levels]
    if args.json:
        print(json.dumps([s.as_dict() for s in sheets], indent=2))
    else:
        print("\n\n".join(lampsheet.format_text(s) for s in sheets))
    if args.spd_csv:
        level = levels[-1]
        lam, density = lampsheet.spd_rows(lamp, level)
        path = Path(args.spd_csv)
        with path.open("w", encoding="utf-8") as f:
            f.write(f"# {lamp.id} level {level}; mean spectral density per 1 nm bin\n")
            f.write("wavelength_nm,density\n")
            for w, d in zip(lam, density):
                f.write(f"{w:g},{d:.9g}\n")
        print(f"\nSPD written to {path.resolve()}", file=sys.stderr)
    return 0


def _cmd_accuracy(args) -> int:
    from lss.accuracy import report

    rows = report.run()
    print(report.format_text(rows))
    if args.update_golden:
        report.write_golden(rows)
        print(f"\ngolden file updated: {report.GOLDEN_PATH}")
        return 0
    failures = report.check_golden(rows)
    if failures:
        print("\nACCURACY REGRESSION:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nno regressions against the golden file")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="lss", description="Light source simulator")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("refs", help="list reference spectra")
    p.add_argument("--family", choices=["incandescent", "fluorescent", "hid", "led"])
    p.set_defaults(func=_cmd_refs)

    p = sub.add_parser("lampsheet", help="lamp sheet for a preset, a YAML file, or ref:<id>")
    p.add_argument("lamp")
    p.add_argument("--level", type=int, help="model level (default: all levels)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--spd-csv", metavar="PATH", help="also write the 1 nm SPD as CSV")
    p.set_defaults(func=_cmd_lampsheet)

    p = sub.add_parser("accuracy", help="score model levels against references")
    p.add_argument("--update-golden", action="store_true", help="accept current scores as the new baseline")
    p.set_defaults(func=_cmd_accuracy)

    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        # Windows consoles default to a legacy code page; never crash on a symbol.
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    try:
        return args.func(args)
    except KeyError as e:
        print(f"error: {e.args[0]}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
