"""Command-line interface (design §13): ``lss refs | lampsheet | accuracy | fit | chart | metamers``."""

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
        step = f"@ {ref.sampling_nm:g} nm" if ref.sampling_nm else f"at {ref.nodes.size} uneven nodes"
        print(f"{ref.id:<14} {ref.family:<13} {lo:g}-{hi:g} nm {step}  {ref.description}")
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


def _cmd_fit(args) -> int:
    from lss.accuracy import fit, references
    from lss.lamp import lamp as lamps

    loaded = [lamps.load(name) for name in args.lamp]
    ref = references.get(args.ref) if args.ref else None
    if ref is not None and len(loaded) > 1:
        print("error: --ref applies to a single lamp; a joint fit scores each lamp against its own", file=sys.stderr)
        return 2
    try:
        if args.hold_out:
            ids = [lamp.id for lamp in loaded]
            if args.hold_out not in ids:
                print(f"error: --hold-out {args.hold_out} is not one of the lamps given", file=sys.stderr)
                return 2
            joint, result = fit.hold_out([(lamp, args.level, None) for lamp in loaded], ids.index(args.hold_out),
                                         args.objective)
        elif len(loaded) > 1:
            result = fit.fit_joint([(lamp, args.level, None) for lamp in loaded], args.objective)
        else:
            result = fit.fit(loaded[0], args.level, ref, args.objective, args.param or None, shared=args.shared)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    elif args.hold_out:
        print(f"Out-of-sample check: shared shapes fitted without {args.hold_out}, "
              "then only its own mix fitted with them.\n")
        print(fit.format_text(result))
    elif len(loaded) > 1:
        print(fit.format_joint(result))
        print()
        print(fit.joint_yaml_snippet(result), end="")
    else:
        print(fit.format_text(result))
        print()
        print(fit.yaml_snippet(result), end="")
    return 0 if result.success else 1


def _cmd_chart(args) -> int:
    from lss.lamp import lamp as lamps
    from lss.studies import appearance, chart, samples

    lamp = lamps.load(args.lamp)
    level = args.level if args.level is not None else max(lamp.models)
    result = chart.make(appearance.lamp_light(lamp, level), samples.get(args.samples))
    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        print(chart.format_text(result))
    if args.svg:
        path = Path(args.svg)
        path.write_text(chart.svg(result), encoding="utf-8")
        print(f"\nswatches written to {path.resolve()}", file=sys.stderr)
    return 0


METAMER_LAMPS = ["ref:cie-a", "ref:cie-fl2", "ref:cie-fl11", "led_cie_b3", "ref:cie-hp1"]


def _cmd_metamers(args) -> int:
    from lss.lamp import lamp as lamps
    from lss.spectral.grids import analysis_grid
    from lss.studies import appearance, metamers

    lights = []
    for name in args.lamp or METAMER_LAMPS:
        lamp = lamps.load(name)
        lights.append(appearance.lamp_light(lamp, max(lamp.models)))
    try:
        pair, study = metamers.study(lights, args.base or metamers.DEFAULT_BASE)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(study.as_dict(), indent=2))
    else:
        print(metamers.format_text(study))
    if args.csv:
        path = Path(args.csv)
        with path.open("w", encoding="utf-8") as f:
            f.write(f"# metameric pair, CONSTRUCTED: '{pair.base}' and a partner matched under {pair.match}\n")
            f.write(f"# rule: {metamers.RULE}\n")
            f.write("wavelength_nm,first,second\n")
            for w, a, b in zip(analysis_grid().centres, pair.first, pair.second):
                f.write(f"{w:g},{a:.6f},{b:.6f}\n")
        print(f"\nreflectances written to {path.resolve()}", file=sys.stderr)
    return 0


def _cmd_leddrive(args) -> int:
    from lss.studies import led_drive

    if not led_drive.available():
        print(f"downloading the measured LED data ({led_drive.CITATION})\n"
              f"from {led_drive.URL}\ninto {led_drive.data_path()} (set LSS_CACHE to change); it stays out "
              "of the repository under its licence", file=sys.stderr)
    try:
        led_drive.fetch()
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except ImportError:
        print('error: reading the data needs openpyxl: pip install "lss[data]"', file=sys.stderr)
        return 2
    print(led_drive.format_text())
    return 0


def _cmd_streethps(args) -> int:
    from lss.accuracy import lica
    from lss.studies import street_hps

    if not lica.available():
        print(f"downloading LICA-UCM lamp spectra ({lica.CITATION})\ninto {lica.path('lps').parent} "
              "(set LSS_CACHE to change); they stay out of the repository at the site's request",
              file=sys.stderr)
    try:
        lica.fetch_all()
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(street_hps.format_text())
    return 0


def _cmd_streethpsfit(args) -> int:
    from lss.accuracy import lica
    from lss.studies import street_hps_fit

    try:
        lica.fetch_all()
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if not all(street_hps_fit.table_path(v).exists() for v in street_hps_fit.VARIANTS):
        print("building the six arc tables on first use (about half an hour on 16 cores; cached afterwards)", file=sys.stderr)
    print(street_hps_fit.format_text())
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

    p = sub.add_parser("fit", help="fit a model level's bounded parameters to a reference; with several "
                                   "lamps, fit the shared phosphor shapes to all of them at once")
    p.add_argument("lamp", nargs="+", help="one lamp, or several for a joint fit of their shared shapes")
    p.add_argument("--level", type=int, required=True, help="model level to fit")
    p.add_argument("--ref", help="reference id (default: the lamp's own; see `lss refs`)")
    p.add_argument("--objective", choices=["spectral", "colour", "uncertainty"], default="spectral",
                   help="spectral: the report's NRMS (default); colour: chromaticity du'v'; "
                        "uncertainty: each node weighted by the reference's stated uncertainty")
    p.add_argument("--param", action="append", metavar="NAME",
                   help="fit only this parameter (repeatable; default: every bounded one)")
    p.add_argument("--shared", action="store_true",
                   help="one lamp: also free the shared phosphor-library shapes (fits them to this lamp alone)")
    p.add_argument("--hold-out", metavar="LAMP",
                   help="out-of-sample check: fit the shared shapes to the other lamps, then score this one")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=_cmd_fit)

    p = sub.add_parser("chart", help="colour chart under a lamp: each sample's colour shift against "
                                     "TM-30's reference light at the same CCT")
    p.add_argument("lamp", help="preset, YAML file, or ref:<id>")
    p.add_argument("--level", type=int, help="model level (default: the highest)")
    p.add_argument("--samples", default="colorchecker", choices=["colorchecker", "cri", "ces"],
                   help="colorchecker: 24 ColorChecker patches (default); cri: CIE 13.3 TCS01-14; "
                        "ces: the 99 TM-30 samples")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--svg", metavar="PATH", help="also write swatches (lamp over reference) as SVG")
    p.set_defaults(func=_cmd_chart)

    p = sub.add_parser("metamers", help="a constructed pair of surfaces that match under D65: "
                                        "how far they split under each lamp")
    p.add_argument("lamp", nargs="*", help=f"lamps at their highest level (default: {' '.join(METAMER_LAMPS)})")
    p.add_argument("--base", help="ColorChecker patch the pair starts from (default: neutral 6.5)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--csv", metavar="PATH", help="also write the two reflectances (1 nm) as CSV")
    p.set_defaults(func=_cmd_metamers)

    p = sub.add_parser("leddrive", help="a measured warm-white LED at 150-500 mA (downloaded on first use): "
                                        "its colour shifts against LED Level 3's")
    p.set_defaults(func=_cmd_leddrive)

    p = sub.add_parser("streethps", help="nine measured HPS street lamps (downloaded on first use) "
                                         "against the rung-1 sodium arc, through the instrument's response")
    p.set_defaults(func=_cmd_streethps)
    p = sub.add_parser("streethpsfit", help="each HPS street lamp's arc temperature and sodium pressure, "
                                            "fitted to its blue peak and line powers; the red wing held out")
    p.set_defaults(func=_cmd_streethpsfit)

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
