#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GC_SOURCE = REPO_ROOT / "GCClassic/src/GEOS-Chem"
DEFAULT_COLUMNS = (
    (108, 59),  # 26N, 87.5E on the current 2x2.5 grid
    (121, 61),  # 30N, 120E
    (118, 63),  # 34N, 112.5E
    (120, 66),  # 40N, 117.5E
    (78, 40),  # 12S, 12.5E
    (130, 43),  # 6S, 142.5E
)


MAIN_INSERTIONS = (
    (
        "          ! Call the appropriate version of TPCORE\n"
        "          IF ( Input_Opt%LTRAN ) THEN\n",
        "          CALL WBT_Trace_Main_Loop( 'before_do_transport', State_Chm, State_Grid, State_Met )\n\n"
        "          ! Call the appropriate version of TPCORE\n"
        "          IF ( Input_Opt%LTRAN ) THEN\n",
    ),
    (
        "             IF ( VerboseAndRoot ) THEN\n"
        "                CALL Debug_Msg( '### MAIN: a DO_TRANSPORT' )\n"
        "             ENDIF\n",
        "             CALL WBT_Trace_Main_Loop( 'after_do_transport', State_Chm, State_Grid, State_Met )\n\n"
        "             IF ( VerboseAndRoot ) THEN\n"
        "                CALL Debug_Msg( '### MAIN: a DO_TRANSPORT' )\n"
        "             ENDIF\n",
    ),
    (
        "             IF ( VerboseAndRoot ) THEN\n"
        "                CALL Debug_Msg( '### MAIN: a SETUP_WETSCAV' )\n"
        "             ENDIF\n",
        "             CALL WBT_Trace_Main_Loop( 'after_setup_wetscav', State_Chm, State_Grid, State_Met )\n\n"
        "             IF ( VerboseAndRoot ) THEN\n"
        "                CALL Debug_Msg( '### MAIN: a SETUP_WETSCAV' )\n"
        "             ENDIF\n",
    ),
    (
        "          IF ( VerboseAndRoot ) THEN\n"
        "             CALL Debug_Msg( '### MAIN: a COMPUTE_PBL_HEIGHT' )\n"
        "          ENDIF\n",
        "          CALL WBT_Trace_Main_Loop( 'after_compute_pbl_height', State_Chm, State_Grid, State_Met )\n\n"
        "          IF ( VerboseAndRoot ) THEN\n"
        "             CALL Debug_Msg( '### MAIN: a COMPUTE_PBL_HEIGHT' )\n"
        "          ENDIF\n",
    ),
    (
        "          IF ( VerboseAndRoot ) THEN\n"
        "             CALL Debug_Msg( '### MAIN: a HEMCO PHASE 2' )\n"
        "          ENDIF\n",
        "          CALL WBT_Trace_Main_Loop( 'after_emissions_run_phase2', State_Chm, State_Grid, State_Met )\n\n"
        "          IF ( VerboseAndRoot ) THEN\n"
        "             CALL Debug_Msg( '### MAIN: a HEMCO PHASE 2' )\n"
        "          ENDIF\n",
    ),
    (
        "          IF ( VerboseAndRoot ) THEN\n"
        "             CALL Debug_Msg( '### MAIN: a Compute_Sflx_For_Vdiff' )\n"
        "          ENDIF\n",
        "          CALL WBT_Trace_Main_Loop( 'after_compute_sflx_for_vdiff', State_Chm, State_Grid, State_Met )\n\n"
        "          IF ( VerboseAndRoot ) THEN\n"
        "             CALL Debug_Msg( '### MAIN: a Compute_Sflx_For_Vdiff' )\n"
        "          ENDIF\n",
    ),
    (
        "          CALL Do_Mixing( Input_Opt,  State_Chm, State_Diag, &\n"
        "                          State_Grid, State_Met, RC )\n",
        "          CALL WBT_Trace_Main_Loop( 'before_do_mixing', State_Chm, State_Grid, State_Met )\n"
        "          CALL Do_Mixing( Input_Opt,  State_Chm, State_Diag, &\n"
        "                          State_Grid, State_Met, RC )\n",
    ),
    (
        "          IF ( VerboseAndRoot ) CALL Debug_Msg( '### MAIN: a TURBDAY:2' )\n",
        "          CALL WBT_Trace_Main_Loop( 'after_do_mixing', State_Chm, State_Grid, State_Met )\n\n"
        "          IF ( VerboseAndRoot ) CALL Debug_Msg( '### MAIN: a TURBDAY:2' )\n",
    ),
    (
        "             ! Call the appropriate convection routine\n"
        "             CALL Do_Convection( Input_Opt,  State_Chm, State_Diag, &\n",
        "             CALL WBT_Trace_Main_Loop( 'before_do_convection', State_Chm, State_Grid, State_Met )\n\n"
        "             ! Call the appropriate convection routine\n"
        "             CALL Do_Convection( Input_Opt,  State_Chm, State_Diag, &\n",
    ),
    (
        "             IF ( VerboseAndRoot ) THEN\n"
        "                CALL Debug_Msg( '### MAIN: a CONVECTION' )\n"
        "             ENDIF\n",
        "             CALL WBT_Trace_Main_Loop( 'after_do_convection', State_Chm, State_Grid, State_Met )\n\n"
        "             IF ( VerboseAndRoot ) THEN\n"
        "                CALL Debug_Msg( '### MAIN: a CONVECTION' )\n"
        "             ENDIF\n",
    ),
)


MIXING_INSERTIONS = (
    (
        "    USE Vdiff_Mod,      ONLY : Do_Vdiff\n",
        "    USE Vdiff_Mod,      ONLY : Do_Vdiff\n"
        "    USE Wombat_Main_Loop_Trace_Mod, ONLY : WBT_Trace_Main_Loop\n",
    ),
    (
        "    USE TIME_MOD,             ONLY : GET_TS_DYN, GET_TS_CONV, GET_TS_CHEM\n",
        "    USE TIME_MOD,             ONLY : GET_TS_DYN, GET_TS_CONV, GET_TS_CHEM\n"
        "    USE Wombat_Main_Loop_Trace_Mod, ONLY : WBT_Trace_Main_Loop, WBT_Trace_Do_Tend\n",
    ),
    (
        "    REAL(fp)                :: TS, TMP, FRQ, RKT, FRAC, FLUX, AREA_M2\n",
        "    REAL(fp)                :: TS, TMP, FRQ, RKT, FRAC, FLUX, AREA_M2\n"
        "    REAL(fp)                :: WBT_CONC_BEFORE\n",
    ),
    (
        "!$OMP PRIVATE( FRQ,      RKT,          FRAC,       FLUX,     Area_m2      ) &\n",
        "!$OMP PRIVATE( FRQ,      RKT,          FRAC,       FLUX,     Area_m2      ) &\n"
        "!$OMP PRIVATE( WBT_CONC_BEFORE                                           ) &\n",
    ),
    (
        "       CALL Do_Vdiff( Input_Opt,  State_Chm, State_Diag,                     &\n"
        "                      State_Grid, State_Met, RC                             )\n",
        "       CALL WBT_Trace_Main_Loop( 'before_do_vdiff', State_Chm, State_Grid, State_Met )\n"
        "       CALL Do_Vdiff( Input_Opt,  State_Chm, State_Diag,                     &\n"
        "                      State_Grid, State_Met, RC                             )\n"
        "       CALL WBT_Trace_Main_Loop( 'after_do_vdiff', State_Chm, State_Grid, State_Met )\n",
    ),
    (
        "    CALL DO_TEND( Input_Opt, State_Chm,  State_Diag, &\n"
        "                  State_Grid, State_Met, OnlyAbovePBL, RC )\n",
        "    CALL WBT_Trace_Main_Loop( 'before_do_tend', State_Chm, State_Grid, State_Met )\n"
        "    CALL DO_TEND( Input_Opt, State_Chm,  State_Diag, &\n"
        "                  State_Grid, State_Met, OnlyAbovePBL, RC )\n"
        "    CALL WBT_Trace_Main_Loop( 'after_do_tend', State_Chm, State_Grid, State_Met )\n",
    ),
    (
        "                ! Add emissions (if any)\n"
        "                ! Bug fix: allow negative fluxes. (ckeller, 4/12/17)\n"
        "                !IF ( FND .AND. (TMP > 0.0_fp) ) THEN\n"
        "                IF ( FND ) THEN\n",
        "                ! Add emissions (if any)\n"
        "                WBT_CONC_BEFORE = State_Chm%Species(N)%Conc(I,J,L)\n"
        "                FLUX = 0.0_fp\n"
        "                ! Bug fix: allow negative fluxes. (ckeller, 4/12/17)\n"
        "                !IF ( FND .AND. (TMP > 0.0_fp) ) THEN\n"
        "                IF ( FND ) THEN\n",
    ),
    (
        "                   ! Add to species array\n"
        "                   State_Chm%Species(N)%Conc(I,J,L) = &\n"
        "                         State_Chm%Species(N)%Conc(I,J,L) + FLUX\n"
        "                ENDIF\n",
        "                   ! Add to species array\n"
        "                   State_Chm%Species(N)%Conc(I,J,L) = &\n"
        "                         State_Chm%Species(N)%Conc(I,J,L) + FLUX\n"
        "                ENDIF\n"
        "                CALL WBT_Trace_Do_Tend( 'do_tend_emis', State_Chm, State_Grid, State_Met, &\n"
        "                                       NA, N, I, J, L, EmisSpec, FND, TMP, FLUX, &\n"
        "                                       WBT_CONC_BEFORE, State_Chm%Species(N)%Conc(I,J,L), &\n"
        "                                       TS, EMIS_TOP, PBL_TOP )\n",
    ),
)


TRANSPORT_INSERTIONS = (
    (
        "    USE TPCORE_FVDAS_MOD,   ONLY : TPCORE_FVDAS\n",
        "    USE TPCORE_FVDAS_MOD,   ONLY : TPCORE_FVDAS\n"
        "    USE Wombat_Main_Loop_Trace_Mod, ONLY : WBT_Trace_Tpcore_Input\n",
    ),
    (
        "    CALL DO_PJC_PFIX( State_Grid,  D_DYN, P_TP1, P_TP2, &\n"
        "                      State_Met%U, State_Met%V, XMASS, YMASS )\n",
        "    CALL DO_PJC_PFIX( State_Grid,  D_DYN, P_TP1, P_TP2, &\n"
        "                      State_Met%U, State_Met%V, XMASS, YMASS )\n"
        "    CALL WBT_Trace_Tpcore_Input( 'before_tpcore_fvdas', State_Chm, State_Grid, State_Met, &\n"
        "                                  P_TP1, P_TP2, XMASS, YMASS )\n",
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate GEOS-Chem main-loop trace instrumentation sources.")
    parser.add_argument("--gc-source", type=Path, default=DEFAULT_GC_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=Path("tools/gc_harness/build/main_loop_trace"))
    parser.add_argument(
        "--column-index",
        action="append",
        default=[],
        metavar="I,J",
        help="1-based GEOS-Chem column index to trace. May be repeated.",
    )
    parser.add_argument("--max-tracers", type=int, default=4)
    args = parser.parse_args()

    columns = tuple(_parse_column(value) for value in args.column_index) or DEFAULT_COLUMNS
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    main_source = args.gc_source / "Interfaces/GCClassic/main.F90"
    mixing_source = args.gc_source / "GeosCore/mixing_mod.F90"
    transport_source = args.gc_source / "GeosCore/transport_mod.F90"
    main_target = output_dir / "main.F90"
    mixing_target = output_dir / "mixing_mod.F90"
    transport_target = output_dir / "transport_mod.F90"
    module_target = output_dir / "wombat_main_loop_trace_mod.F90"

    main_text = _replace_once(
        main_source.read_text(encoding="utf-8"),
        "  USE VDIFF_MOD             ! For non-local PBL mixing (J. Lin)\n",
        "  USE VDIFF_MOD             ! For non-local PBL mixing (J. Lin)\n"
        "  USE Wombat_Main_Loop_Trace_Mod, ONLY : WBT_Trace_Main_Loop\n",
    )
    for needle, replacement in MAIN_INSERTIONS:
        main_text = _replace_once(main_text, needle, replacement)
    main_target.write_text(main_text, encoding="utf-8")

    mixing_text = mixing_source.read_text(encoding="utf-8")
    for needle, replacement in MIXING_INSERTIONS:
        mixing_text = _replace_once(mixing_text, needle, replacement)
    mixing_target.write_text(mixing_text, encoding="utf-8")
    transport_text = transport_source.read_text(encoding="utf-8")
    for needle, replacement in TRANSPORT_INSERTIONS:
        transport_text = _replace_once(transport_text, needle, replacement)
    transport_target.write_text(transport_text, encoding="utf-8")
    module_target.write_text(_trace_module(columns, args.max_tracers), encoding="utf-8")

    print(f"wrote_main_loop_trace_sources: {output_dir}")
    print(f"instrumented_main: {main_target}")
    print(f"instrumented_mixing: {mixing_target}")
    print(f"instrumented_transport: {transport_target}")
    print(f"trace_module: {module_target}")
    return 0


def _parse_column(value: str) -> tuple[int, int]:
    try:
        i, j = (int(item.strip()) for item in value.split(",", 1))
    except ValueError as exc:
        raise ValueError(f"--column-index must be I,J, got {value!r}") from exc
    if i <= 0 or j <= 0:
        raise ValueError("--column-index values are 1-based and must be positive")
    return i, j


def _replace_once(text: str, needle: str, replacement: str) -> str:
    if needle not in text:
        raise ValueError(f"could not find insertion point:\n{needle}")
    return text.replace(needle, replacement, 1)


def _trace_module(columns: tuple[tuple[int, int], ...], max_tracers: int) -> str:
    i_values = ", ".join(str(item[0]) for item in columns)
    j_values = ", ".join(str(item[1]) for item in columns)
    return f"""! Auto-generated by tools/gc_harness/generate_main_loop_trace_source.py
MODULE Wombat_Main_Loop_Trace_Mod
  USE Precision_Mod,  ONLY : fp
  USE State_Chm_Mod,  ONLY : ChmState
  USE State_Grid_Mod, ONLY : GrdState
  USE State_Met_Mod,  ONLY : MetState
  USE Tpcore_Trace_Mod, ONLY : Tpcore_Trace_Init, Tpcore_Trace_Write
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: WBT_Trace_Main_Loop, WBT_Trace_Do_Tend, WBT_Trace_Tpcore_Input

  INTEGER, PARAMETER :: N_TRACE_COLS = {len(columns)}
  INTEGER, PARAMETER :: MAX_TRACE_TRACERS = {max_tracers}
  INTEGER, PARAMETER :: TRACE_I(N_TRACE_COLS) = (/ {i_values} /)
  INTEGER, PARAMETER :: TRACE_J(N_TRACE_COLS) = (/ {j_values} /)
  REAL(fp), PARAMETER :: FILL = -9.87654321e36_fp
  LOGICAL, SAVE :: IsInitialized = .FALSE.
  INTEGER, SAVE :: TraceUnit = -1
  INTEGER, SAVE :: CallIndex = 0
  CHARACTER(LEN=512), SAVE :: SnapshotDir = ''
  CHARACTER(LEN=512), SAVE :: TpcoreTracePath = ''
  INTEGER, PARAMETER :: MAX_SNAPSHOT_STEPS = 128
  INTEGER, PARAMETER :: MAX_SNAPSHOT_LABELS = 64
  INTEGER, SAVE :: NSnapshotSteps = 0
  INTEGER, SAVE :: SnapshotSteps(MAX_SNAPSHOT_STEPS) = -1
  CHARACTER(LEN=64), SAVE :: SnapshotLabels(MAX_SNAPSHOT_LABELS) = ''
  INTEGER, SAVE :: SnapshotLabelCounts(MAX_SNAPSHOT_LABELS) = 0
  INTEGER, SAVE :: NSnapshotLabels = 0
  LOGICAL, SAVE :: TpcoreTraceInitialized = .FALSE.

CONTAINS

  SUBROUTINE WBT_Trace_Main_Loop( Label, State_Chm, State_Grid, State_Met )
    CHARACTER(LEN=*), INTENT(IN) :: Label
    TYPE(ChmState),   INTENT(IN) :: State_Chm
    TYPE(GrdState),   INTENT(IN) :: State_Grid
    TYPE(MetState),   INTENT(IN) :: State_Met
    INTEGER :: C, I, J, L, T, NTracer, SpeciesId, Occurrence
    CHARACTER(LEN=31) :: SpeciesName

    IF ( .NOT. IsInitialized ) CALL Init_Trace()
    CallIndex = CallIndex + 1
    Occurrence = Next_Label_Occurrence( Label )
    NTracer = MIN( State_Chm%nAdvect, MAX_TRACE_TRACERS )

    DO C = 1, N_TRACE_COLS
       I = TRACE_I(C)
       J = TRACE_J(C)
       IF ( I < 1 .OR. I > State_Grid%NX .OR. J < 1 .OR. J > State_Grid%NY ) CYCLE
       DO L = 1, State_Grid%NZ
          DO T = 1, NTracer
             SpeciesId = State_Chm%Map_Advect(T)
             SpeciesName = ''
             IF ( SpeciesId > 0 ) SpeciesName = State_Chm%SpcData(SpeciesId)%Info%Name
             WRITE( TraceUnit, '(I0,",",A,",",I0,",",I0,",",I0,",",I0,",",A,",",26(ES24.16,","),ES24.16)' ) &
                  CallIndex, TRIM(Label), I, J, L, T, TRIM(SpeciesName), &
                  Trace_Conc(State_Chm, SpeciesId, I, J, L), Trace_AD(State_Met, I, J, L), &
                  Trace_3D(State_Met%DELP_DRY, I, J, L), Trace_3D(State_Met%BXHEIGHT, I, J, L), &
                  Trace_2D(State_Met%PBL_TOP_L, I, J), Trace_2D(State_Met%PBL_TOP_m, I, J), &
                  Trace_SurfaceFlux(State_Chm, I, J, T), Trace_3D(State_Met%SPHU, I, J, L), &
                  Trace_3D(State_Met%T, I, J, L), Trace_3D(State_Met%CMFMC, I, J, L), &
                  Trace_3D(State_Met%DTRAIN, I, J, L), Trace_3D(State_Met%DQRCU, I, J, L), &
                  Trace_3D(State_Met%REEVAPCN, I, J, L), Trace_3D(State_Met%PFICU, I, J, L), &
                  Trace_3D(State_Met%U, I, J, L), Trace_3D(State_Met%V, I, J, L), &
                  Trace_3D(State_Met%PMID, I, J, L), Trace_3D(State_Met%PEDGE, I, J, L), &
                  Trace_3D(State_Met%PEDGE, I, J, L + 1), Trace_3D(State_Met%TV, I, J, L), &
                  Trace_2D(State_Met%HFLUX, I, J), Trace_2D(State_Met%EFLUX, I, J), &
                  Trace_2D(State_Met%USTAR, I, J), FILL, FILL, FILL, FILL
          ENDDO
       ENDDO
    ENDDO
    FLUSH( TraceUnit )
    CALL Maybe_Write_Snapshot( Label, Occurrence, State_Chm, State_Grid )
    IF ( TRIM(Label) == 'after_do_transport' ) CALL Tpcore_Trace_Write()
  END SUBROUTINE WBT_Trace_Main_Loop

  SUBROUTINE WBT_Trace_Do_Tend( Label, State_Chm, State_Grid, State_Met, AdvIndex, SpeciesId, I, J, L, &
                                EmisSpec, Found, HcoFlux, AppliedFlux, ConcBefore, ConcAfter, Ts, EmisTop, PblTop )
    CHARACTER(LEN=*), INTENT(IN) :: Label
    TYPE(ChmState),   INTENT(IN) :: State_Chm
    TYPE(GrdState),   INTENT(IN) :: State_Grid
    TYPE(MetState),   INTENT(IN) :: State_Met
    INTEGER,          INTENT(IN) :: AdvIndex, SpeciesId, I, J, L, EmisTop, PblTop
    LOGICAL,          INTENT(IN) :: EmisSpec, Found
    REAL(fp),         INTENT(IN) :: HcoFlux, AppliedFlux, ConcBefore, ConcAfter, Ts
    CHARACTER(LEN=31) :: SpeciesName

    IF ( .NOT. IsInitialized ) CALL Init_Trace()
    IF ( .NOT. Should_Trace( I, J, AdvIndex ) ) RETURN
    SpeciesName = ''
    IF ( SpeciesId > 0 ) SpeciesName = State_Chm%SpcData(SpeciesId)%Info%Name
!$OMP CRITICAL(WBT_TRACE_WRITE)
    CallIndex = CallIndex + 1
    WRITE( TraceUnit, '(I0,",",A,",",I0,",",I0,",",I0,",",I0,",",A,",",26(ES24.16,","),ES24.16)' ) &
         CallIndex, TRIM(Label), I, J, L, AdvIndex, TRIM(SpeciesName), &
         ConcAfter, Trace_AD(State_Met, I, J, L), Trace_3D(State_Met%DELP_DRY, I, J, L), &
         Trace_3D(State_Met%BXHEIGHT, I, J, L), REAL(EmisTop, fp), REAL(PblTop, fp), &
         Trace_SurfaceFlux(State_Chm, I, J, AdvIndex), Trace_3D(State_Met%SPHU, I, J, L), &
         Trace_3D(State_Met%T, I, J, L), Trace_3D(State_Met%CMFMC, I, J, L), &
         HcoFlux, AppliedFlux, ConcBefore, ConcAfter, &
         Trace_3D(State_Met%U, I, J, L), Trace_3D(State_Met%V, I, J, L), &
         Trace_3D(State_Met%PMID, I, J, L), Trace_3D(State_Met%PEDGE, I, J, L), &
         Trace_3D(State_Met%PEDGE, I, J, L + 1), Trace_3D(State_Met%TV, I, J, L), &
         Trace_2D(State_Met%HFLUX, I, J), Trace_2D(State_Met%EFLUX, I, J), &
         Trace_2D(State_Met%USTAR, I, J), FILL, FILL, FILL, FILL
    WRITE( TraceUnit, '(I0,",",A,",",I0,",",I0,",",I0,",",I0,",",A,",",26(ES24.16,","),ES24.16)' ) &
         CallIndex, 'do_tend_meta', I, J, L, AdvIndex, TRIM(SpeciesName), &
         MERGE( 1.0_fp, 0.0_fp, EmisSpec ), MERGE( 1.0_fp, 0.0_fp, Found ), Ts, REAL(EmisTop, fp), &
         REAL(PblTop, fp), ConcAfter - ConcBefore, HcoFlux, AppliedFlux, ConcBefore, ConcAfter, &
         Trace_AD(State_Met, I, J, L), Trace_3D(State_Met%DELP_DRY, I, J, L), Trace_3D(State_Met%BXHEIGHT, I, J, L), &
         Trace_SurfaceFlux(State_Chm, I, J, AdvIndex), Trace_3D(State_Met%U, I, J, L), &
         Trace_3D(State_Met%V, I, J, L), Trace_3D(State_Met%PMID, I, J, L), &
         Trace_3D(State_Met%PEDGE, I, J, L), Trace_3D(State_Met%PEDGE, I, J, L + 1), &
         Trace_3D(State_Met%TV, I, J, L), Trace_2D(State_Met%HFLUX, I, J), &
         Trace_2D(State_Met%EFLUX, I, J), Trace_2D(State_Met%USTAR, I, J), FILL, FILL, FILL, FILL
    FLUSH( TraceUnit )
!$OMP END CRITICAL(WBT_TRACE_WRITE)
  END SUBROUTINE WBT_Trace_Do_Tend

  SUBROUTINE WBT_Trace_Tpcore_Input( Label, State_Chm, State_Grid, State_Met, P_TP1, P_TP2, XMASS, YMASS )
    CHARACTER(LEN=*), INTENT(IN) :: Label
    TYPE(ChmState),   INTENT(IN) :: State_Chm
    TYPE(GrdState),   INTENT(IN) :: State_Grid
    TYPE(MetState),   INTENT(IN) :: State_Met
    REAL(fp),         INTENT(IN) :: P_TP1(:,:), P_TP2(:,:), XMASS(:,:,:), YMASS(:,:,:)
    INTEGER :: C, I, J, L, T, NTracer, SpeciesId, Occurrence
    CHARACTER(LEN=31) :: SpeciesName

    IF ( .NOT. IsInitialized ) CALL Init_Trace()
    CallIndex = CallIndex + 1
    Occurrence = Next_Label_Occurrence( Label )
    NTracer = MIN( State_Chm%nAdvect, MAX_TRACE_TRACERS )

    DO C = 1, N_TRACE_COLS
       I = TRACE_I(C)
       J = TRACE_J(C)
       IF ( I < 1 .OR. I > State_Grid%NX .OR. J < 1 .OR. J > State_Grid%NY ) CYCLE
       DO L = 1, State_Grid%NZ
          DO T = 1, NTracer
             SpeciesId = State_Chm%Map_Advect(T)
             SpeciesName = ''
             IF ( SpeciesId > 0 ) SpeciesName = State_Chm%SpcData(SpeciesId)%Info%Name
             WRITE( TraceUnit, '(I0,",",A,",",I0,",",I0,",",I0,",",I0,",",A,",",26(ES24.16,","),ES24.16)' ) &
                  CallIndex, TRIM(Label), I, J, L, T, TRIM(SpeciesName), &
                  Trace_Conc(State_Chm, SpeciesId, I, J, L), Trace_AD(State_Met, I, J, L), &
                  Trace_3D(State_Met%DELP_DRY, I, J, L), Trace_3D(State_Met%BXHEIGHT, I, J, L), &
                  Trace_2D(State_Met%PBL_TOP_L, I, J), Trace_2D(State_Met%PBL_TOP_m, I, J), &
                  Trace_SurfaceFlux(State_Chm, I, J, T), Trace_3D(State_Met%SPHU, I, J, L), &
                  Trace_3D(State_Met%T, I, J, L), Trace_3D(State_Met%CMFMC, I, J, L), &
                  Trace_3D(State_Met%DTRAIN, I, J, L), Trace_3D(State_Met%DQRCU, I, J, L), &
                  Trace_3D(State_Met%REEVAPCN, I, J, L), Trace_3D(State_Met%PFICU, I, J, L), &
                  Trace_3D(State_Met%U, I, J, L), Trace_3D(State_Met%V, I, J, L), &
                  Trace_3D(State_Met%PMID, I, J, L), Trace_3D(State_Met%PEDGE, I, J, L), &
                  Trace_3D(State_Met%PEDGE, I, J, L + 1), Trace_3D(State_Met%TV, I, J, L), &
                  Trace_2D(State_Met%HFLUX, I, J), Trace_2D(State_Met%EFLUX, I, J), &
                  Trace_2D(State_Met%USTAR, I, J), P_TP1(I,J), P_TP2(I,J), XMASS(I,J,L), YMASS(I,J,L)
          ENDDO
       ENDDO
    ENDDO
    FLUSH( TraceUnit )
    CALL Maybe_Write_Snapshot( Label, Occurrence, State_Chm, State_Grid )
    CALL Maybe_Write_Tpcore_Input_Snapshot( Label, Occurrence, P_TP1, P_TP2, XMASS, YMASS )
    CALL Maybe_Init_Tpcore_Trace( State_Chm, State_Grid, NTracer )
  END SUBROUTINE WBT_Trace_Tpcore_Input

  SUBROUTINE Init_Trace()
    CHARACTER(LEN=512) :: Path
    CHARACTER(LEN=1024) :: Steps
    INTEGER :: Status
    CALL GET_ENVIRONMENT_VARIABLE( 'WOMBAT_GC_TRACE_CSV', Path, STATUS=Status )
    IF ( Status /= 0 .OR. LEN_TRIM(Path) == 0 ) Path = 'wombat_gc_main_loop_trace.csv'
    OPEN( NEWUNIT=TraceUnit, FILE=TRIM(Path), STATUS='REPLACE', ACTION='WRITE' )
    WRITE( TraceUnit, '(A)' ) &
         'call_index,boundary,i,j,l,tracer,tracer_name,tracer_conc,ad_kg,delp_dry_hpa,bxheight_m' // &
         ',pbl_top_l,pbl_top_m,surface_flux_kg_m2_s,sphu_g_kg,temperature_k,cmfmc_kg_m2_s' // &
         ',dtrain_kg_m2_s,dqrcu_kg_kg_s,reevapcn_kg_kg_s,pficu_kg_m2_s' // &
         ',u_m_s,v_m_s,pmid_hpa,pedge_lower_hpa,pedge_upper_hpa,tv_k,hflux_w_m2,eflux_w_m2,ustar_m_s' // &
         ',p_tp1_hpa,p_tp2_hpa,xmass_hpa,ymass_hpa'
    CALL GET_ENVIRONMENT_VARIABLE( 'WOMBAT_GC_SNAPSHOT_DIR', SnapshotDir, STATUS=Status )
    IF ( Status /= 0 ) SnapshotDir = ''
    CALL GET_ENVIRONMENT_VARIABLE( 'WOMBAT_GC_TPCORE_TRACE_NC', TpcoreTracePath, STATUS=Status )
    IF ( Status /= 0 ) TpcoreTracePath = ''
    CALL GET_ENVIRONMENT_VARIABLE( 'WOMBAT_GC_SNAPSHOT_STEPS', Steps, STATUS=Status )
    IF ( Status == 0 .AND. LEN_TRIM(Steps) > 0 ) CALL Parse_Snapshot_Steps( Steps )
    IsInitialized = .TRUE.
  END SUBROUTINE Init_Trace

  SUBROUTINE Parse_Snapshot_Steps( Text )
    CHARACTER(LEN=*), INTENT(IN) :: Text
    INTEGER :: Start, Comma, Value, Status
    CHARACTER(LEN=32) :: Token

    NSnapshotSteps = 0
    Start = 1
    DO WHILE ( Start <= LEN_TRIM(Text) .AND. NSnapshotSteps < MAX_SNAPSHOT_STEPS )
       Comma = INDEX( Text(Start:), ',' )
       IF ( Comma == 0 ) THEN
          Token = ADJUSTL( Text(Start:) )
          Start = LEN_TRIM(Text) + 1
       ELSE
          Token = ADJUSTL( Text(Start:Start+Comma-2) )
          Start = Start + Comma
       ENDIF
       IF ( LEN_TRIM(Token) == 0 ) CYCLE
       READ( Token, *, IOSTAT=Status ) Value
       IF ( Status == 0 .AND. Value >= 0 ) THEN
          NSnapshotSteps = NSnapshotSteps + 1
          SnapshotSteps(NSnapshotSteps) = Value
       ENDIF
    ENDDO
  END SUBROUTINE Parse_Snapshot_Steps

  INTEGER FUNCTION Next_Label_Occurrence( Label )
    CHARACTER(LEN=*), INTENT(IN) :: Label
    INTEGER :: N
    DO N = 1, NSnapshotLabels
       IF ( TRIM(SnapshotLabels(N)) == TRIM(Label) ) THEN
          Next_Label_Occurrence = SnapshotLabelCounts(N)
          SnapshotLabelCounts(N) = SnapshotLabelCounts(N) + 1
          RETURN
       ENDIF
    ENDDO
    IF ( NSnapshotLabels < MAX_SNAPSHOT_LABELS ) THEN
       NSnapshotLabels = NSnapshotLabels + 1
       SnapshotLabels(NSnapshotLabels) = Label
       SnapshotLabelCounts(NSnapshotLabels) = 1
    ENDIF
    Next_Label_Occurrence = 0
  END FUNCTION Next_Label_Occurrence

  LOGICAL FUNCTION Should_Write_Snapshot( Occurrence )
    INTEGER, INTENT(IN) :: Occurrence
    INTEGER :: N
    Should_Write_Snapshot = .FALSE.
    IF ( LEN_TRIM(SnapshotDir) == 0 .OR. NSnapshotSteps == 0 ) RETURN
    DO N = 1, NSnapshotSteps
       IF ( SnapshotSteps(N) == Occurrence ) THEN
          Should_Write_Snapshot = .TRUE.
          RETURN
       ENDIF
    ENDDO
  END FUNCTION Should_Write_Snapshot

  SUBROUTINE Maybe_Write_Snapshot( Label, Occurrence, State_Chm, State_Grid )
    CHARACTER(LEN=*), INTENT(IN) :: Label
    INTEGER,          INTENT(IN) :: Occurrence
    TYPE(ChmState),   INTENT(IN) :: State_Chm
    TYPE(GrdState),   INTENT(IN) :: State_Grid
    INTEGER :: I, J, L, T, Unit, NTracer, SpeciesId
    CHARACTER(LEN=1024) :: Path
    REAL(fp), ALLOCATABLE :: Snapshot(:,:,:,:)

    IF ( .NOT. Should_Write_Snapshot( Occurrence ) ) RETURN
    NTracer = MIN( State_Chm%nAdvect, MAX_TRACE_TRACERS )
    IF ( NTracer <= 0 ) RETURN
    ALLOCATE( Snapshot(State_Grid%NX, State_Grid%NY, State_Grid%NZ, NTracer) )
    Snapshot = FILL
    DO T = 1, NTracer
       SpeciesId = State_Chm%Map_Advect(T)
       IF ( SpeciesId <= 0 ) CYCLE
       IF ( .NOT. ASSOCIATED( State_Chm%Species(SpeciesId)%Conc ) ) CYCLE
       DO L = 1, State_Grid%NZ
          DO J = 1, State_Grid%NY
             DO I = 1, State_Grid%NX
                Snapshot(I,J,L,T) = State_Chm%Species(SpeciesId)%Conc(I,J,L)
             ENDDO
          ENDDO
       ENDDO
    ENDDO

    WRITE( Path, '(A,"/",A,"_",I6.6,".bin")' ) TRIM(SnapshotDir), TRIM(Label), Occurrence
    OPEN( NEWUNIT=Unit, FILE=TRIM(Path), STATUS='REPLACE', ACCESS='STREAM', FORM='UNFORMATTED' )
    WRITE( Unit ) State_Grid%NX, State_Grid%NY, State_Grid%NZ, NTracer
    WRITE( Unit ) Snapshot
    CLOSE( Unit )
    DEALLOCATE( Snapshot )
  END SUBROUTINE Maybe_Write_Snapshot

  SUBROUTINE Maybe_Write_Tpcore_Input_Snapshot( Label, Occurrence, P_TP1, P_TP2, XMASS, YMASS )
    CHARACTER(LEN=*), INTENT(IN) :: Label
    INTEGER,          INTENT(IN) :: Occurrence
    REAL(fp),         INTENT(IN) :: P_TP1(:,:), P_TP2(:,:), XMASS(:,:,:), YMASS(:,:,:)
    INTEGER :: Unit
    CHARACTER(LEN=1024) :: Path

    IF ( .NOT. Should_Write_Snapshot( Occurrence ) ) RETURN
    WRITE( Path, '(A,"/",A,"_inputs_",I6.6,".bin")' ) TRIM(SnapshotDir), TRIM(Label), Occurrence
    OPEN( NEWUNIT=Unit, FILE=TRIM(Path), STATUS='REPLACE', ACCESS='STREAM', FORM='UNFORMATTED' )
    WRITE( Unit ) SIZE(P_TP1, 1), SIZE(P_TP1, 2), SIZE(XMASS, 3)
    WRITE( Unit ) P_TP1
    WRITE( Unit ) P_TP2
    WRITE( Unit ) XMASS
    WRITE( Unit ) YMASS
    CLOSE( Unit )
  END SUBROUTINE Maybe_Write_Tpcore_Input_Snapshot

  SUBROUTINE Maybe_Init_Tpcore_Trace( State_Chm, State_Grid, NTracer )
    TYPE(ChmState), INTENT(IN) :: State_Chm
    TYPE(GrdState), INTENT(IN) :: State_Grid
    INTEGER,        INTENT(IN) :: NTracer
    IF ( TpcoreTraceInitialized ) RETURN
    TpcoreTraceInitialized = .TRUE.
    IF ( LEN_TRIM(TpcoreTracePath) == 0 .OR. NTracer <= 0 ) RETURN
    CALL Tpcore_Trace_Init( TRIM(TpcoreTracePath), State_Grid%NX, State_Grid%NY, State_Grid%NZ, NTracer, 600.0_fp )
  END SUBROUTINE Maybe_Init_Tpcore_Trace

  LOGICAL FUNCTION Should_Trace( I, J, T )
    INTEGER, INTENT(IN) :: I, J, T
    INTEGER :: C
    Should_Trace = .FALSE.
    IF ( T < 1 .OR. T > MAX_TRACE_TRACERS ) RETURN
    DO C = 1, N_TRACE_COLS
       IF ( I == TRACE_I(C) .AND. J == TRACE_J(C) ) THEN
          Should_Trace = .TRUE.
          RETURN
       ENDIF
    ENDDO
  END FUNCTION Should_Trace

  REAL(fp) FUNCTION Trace_Conc( State_Chm, SpeciesId, I, J, L )
    TYPE(ChmState), INTENT(IN) :: State_Chm
    INTEGER, INTENT(IN) :: SpeciesId, I, J, L
    Trace_Conc = FILL
    IF ( SpeciesId > 0 ) THEN
       IF ( ASSOCIATED( State_Chm%Species(SpeciesId)%Conc ) ) Trace_Conc = State_Chm%Species(SpeciesId)%Conc(I,J,L)
    ENDIF
  END FUNCTION Trace_Conc

  REAL(fp) FUNCTION Trace_AD( State_Met, I, J, L )
    TYPE(MetState), INTENT(IN) :: State_Met
    INTEGER, INTENT(IN) :: I, J, L
    Trace_AD = Trace_3D( State_Met%AD, I, J, L )
  END FUNCTION Trace_AD

  REAL(fp) FUNCTION Trace_SurfaceFlux( State_Chm, I, J, T )
    TYPE(ChmState), INTENT(IN) :: State_Chm
    INTEGER, INTENT(IN) :: I, J, T
    Trace_SurfaceFlux = FILL
    IF ( ASSOCIATED( State_Chm%SurfaceFlux ) ) Trace_SurfaceFlux = State_Chm%SurfaceFlux(I,J,T)
  END FUNCTION Trace_SurfaceFlux

  REAL(fp) FUNCTION Trace_2D( Field, I, J )
    REAL(fp), POINTER, INTENT(IN) :: Field(:,:)
    INTEGER, INTENT(IN) :: I, J
    Trace_2D = FILL
    IF ( ASSOCIATED( Field ) ) Trace_2D = Field(I,J)
  END FUNCTION Trace_2D

  REAL(fp) FUNCTION Trace_3D( Field, I, J, L )
    REAL(fp), POINTER, INTENT(IN) :: Field(:,:,:)
    INTEGER, INTENT(IN) :: I, J, L
    Trace_3D = FILL
    IF ( ASSOCIATED( Field ) ) Trace_3D = Field(I,J,L)
  END FUNCTION Trace_3D

END MODULE Wombat_Main_Loop_Trace_Mod
"""


if __name__ == "__main__":
    raise SystemExit(main())
