#!/usr/bin/env python3
"""Generate an instrumented copy of GEOS-Chem TPCORE for harness tracing."""

from __future__ import annotations

import argparse
from pathlib import Path


REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (
        "  USE PRECISION_MOD    ! For GEOS-Chem Precision (fp)\n",
        "  USE PRECISION_MOD    ! For GEOS-Chem Precision (fp)\n"
        "  USE Tpcore_Trace_Mod, ONLY : Tpcore_Trace_Enabled, Tpcore_Trace_Setup, &\n"
        "       Tpcore_Trace_Field2d, Tpcore_Trace_Field3d\n",
    ),
    (
        "    ! Calculate surf. pressure at t+dt. (ccc, 11/20/08)\n"
        "    ps = ak(1)+sum(delp2,dim=3)\n",
        "    ! Calculate surf. pressure at t+dt. (ccc, 11/20/08)\n"
        "    ps = ak(1)+sum(delp2,dim=3)\n"
        "\n"
        "    IF ( Tpcore_Trace_Enabled() ) THEN\n"
        "       CALL Tpcore_Trace_Setup( delp1, delp2, cx(:,1:jm,:), cy(:,1:jm,:), wz, ps )\n"
        "    ENDIF\n",
    ),
    (
        "          q_ptr => NULL()\n"
        "\n"
        "      end do\n",
        "          IF ( Tpcore_Trace_Enabled() ) THEN\n"
        "             CALL Tpcore_Trace_Field2d( 'q_after_pole_average', iq, km - ik + 1, q_ptr(:,:,ik) )\n"
        "          ENDIF\n"
        "\n"
        "          q_ptr => NULL()\n"
        "\n"
        "      end do\n",
    ),
    (
        "          dq1(:,:,ik) = q_ptr(:,:,ik) * delp1(:,:,ik)\n",
        "          dq1(:,:,ik) = q_ptr(:,:,ik) * delp1(:,:,ik)\n"
        "          IF ( Tpcore_Trace_Enabled() ) THEN\n"
        "             CALL Tpcore_Trace_Field2d( 'dq_after_init_hpa', iq, km - ik + 1, dq1(:,:,ik) )\n"
        "          ENDIF\n",
    ),
    (
        "          q_ptr(:,:,ik) = q_ptr(:,:,ik) + ady + adx\n"
        "\n"
        "\n"
        "        ! ========\n",
        "          q_ptr(:,:,ik) = q_ptr(:,:,ik) + ady + adx\n"
        "          IF ( Tpcore_Trace_Enabled() ) THEN\n"
        "             CALL Tpcore_Trace_Field2d( 'q_after_cross_terms', iq, km - ik + 1, q_ptr(:,:,ik) )\n"
        "          ENDIF\n"
        "\n"
        "\n"
        "        ! ========\n",
    ),
    (
        "                 1,             jm,          IORD                           )\n",
        "                 1,             jm,          IORD                           )\n"
        "          IF ( Tpcore_Trace_Enabled() ) THEN\n"
        "             CALL Tpcore_Trace_Field2d( 'dq_after_xtp_hpa', iq, km - ik + 1, dq1(:,:,ik) )\n"
        "          ENDIF\n",
    ),
    (
        "                 1,    im,        1,             jm,            JORD        )\n",
        "                 1,    im,        1,             jm,            JORD        )\n"
        "          IF ( Tpcore_Trace_Enabled() ) THEN\n"
        "             CALL Tpcore_Trace_Field2d( 'dq_after_ytp_hpa', iq, km - ik + 1, dq1(:,:,ik) )\n"
        "          ENDIF\n",
    ),
    (
        "              km,   1,     im, 1,   jm,    1,            km                 )\n",
        "              km,   1,     im, 1,   jm,    1,            km                 )\n"
        "       IF ( Tpcore_Trace_Enabled() ) THEN\n"
        "          CALL Tpcore_Trace_Field3d( 'dq_after_fzppm_hpa', iq, dq1 )\n"
        "       ENDIF\n",
    ),
    (
        "               (dq1, j1p, j2p, 1, jm, 1, im, 1, jm, 1, im, 1, jm, 1, km     )\n"
        "       end if\n",
        "               (dq1, j1p, j2p, 1, jm, 1, im, 1, jm, 1, im, 1, jm, 1, km     )\n"
        "       end if\n"
        "\n"
        "       IF ( Tpcore_Trace_Enabled() ) THEN\n"
        "          CALL Tpcore_Trace_Field3d( 'dq_after_fill_hpa', iq, dq1 )\n"
        "       ENDIF\n",
    ),
    (
        "       WHERE ( q_ptr < 0.0_fp )\n"
        "          q_ptr = 1.0e-26_fp\n"
        "       ENDWHERE\n"
        "\n"
        "       q_ptr => NULL()\n",
        "       WHERE ( q_ptr < 0.0_fp )\n"
        "          q_ptr = 1.0e-26_fp\n"
        "       ENDWHERE\n"
        "\n"
        "       IF ( Tpcore_Trace_Enabled() ) THEN\n"
        "          CALL Tpcore_Trace_Field3d( 'tracer_conc_after', iq, State_Chm%Species(iq)%Conc )\n"
        "       ENDIF\n"
        "\n"
        "       q_ptr => NULL()\n",
    ),
)


def generate(source: Path, output: Path) -> None:
    text = source.read_text(encoding="utf-8")
    for old, new in REPLACEMENTS:
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"anchor matched {count} times, expected 1:\n{old}")
        text = text.replace(old, new)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generate(args.source, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
