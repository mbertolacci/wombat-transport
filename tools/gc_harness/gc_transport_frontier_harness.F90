! Persistent GEOS-Chem transport-chain worker for the frontier coordinator.
!
! The three input files are the existing base_initial_transport_chain_v3
! handoff inputs.  Meteorology and state are loaded once, while the first
! tracer template is replicated to the requested runtime tracer count.  The
! line protocol deliberately stays small:
!
!   READY THREADS
!   RUN TARGET_MONOTONIC_NS
!   DONE STARTED_NS COMPLETED_NS CHECKSUM
!   STOP 0

program gc_transport_frontier_harness
  use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, c_long
  use, intrinsic :: iso_fortran_env, only: input_unit, output_unit
  use netcdf
  use omp_lib, only: omp_get_max_threads
  use precision_mod, only: fp
  use input_opt_mod, only: OptInput
  use species_mod, only: Species
  use state_grid_mod, only: GrdState
  use state_chm_mod, only: ChmState
  use state_diag_mod, only: DgnState
  use state_met_mod, only: MetState
  use pressure_mod, only: INIT_PRESSURE, GET_AP, GET_BP
  use pjc_pfix_mod, only: DO_PJC_PFIX
  use tpcore_fvdas_mod, only: INIT_TPCORE, TPCORE_FVDAS
  use vdiff_mod, only: Init_Vdiff, Max_PblHt_For_Vdiff, VDIFFDR
  use convection_mod, only: DO_CLOUD_CONVECTION
  use time_mod, only: SET_TIMESTEPS
  use PhysConstants, only: Re
  implicit none

  type, bind(C) :: c_timespec
     integer(c_long) :: tv_sec
     integer(c_long) :: tv_nsec
  end type c_timespec

  interface
     function c_clock_gettime(clock_id, value) bind(C, name="clock_gettime") result(status)
       import :: c_int, c_timespec
       integer(c_int), value :: clock_id
       type(c_timespec), intent(out) :: value
       integer(c_int) :: status
     end function c_clock_gettime

     function c_usleep(useconds) bind(C, name="usleep") result(status)
       import :: c_int
       integer(c_int), value :: useconds
       integer(c_int) :: status
     end function c_usleep
  end interface

  integer(c_int), parameter :: CLOCK_MONOTONIC = 1_c_int
  type(OptInput) :: Input_Opt
  type(GrdState) :: State_Grid
  type(ChmState) :: State_Chm
  type(DgnState) :: State_Diag
  type(MetState) :: State_Met
  character(len=1024) :: tpcore_path, vdiff_path, convection_path
  character(len=1024) :: argument
  character(len=64) :: command
  character(len=2048) :: line
  integer :: nx, ny, nz, ntracer, warmup, source_tracers
  integer :: ncid, rc, iteration, ios
  integer(c_int64_t) :: target_ns, started_ns, completed_ns
  real(fp) :: dt_s, checksum
  real(fp), allocatable :: lon(:), lat(:), area(:,:)
  real(fp), allocatable :: tracer_template(:,:,:,:), surface_flux_template(:,:,:)
  real(fp), allocatable, target :: p1(:,:), p2(:,:), p1_base(:,:), p2_base(:,:)
  real(fp), allocatable, target :: tpcore_u(:,:,:), tpcore_v(:,:,:)
  real(fp), allocatable, target :: xmass(:,:,:), ymass(:,:,:)
  real(fp), allocatable :: sphu_base(:,:,:), vdiff_bxheight(:,:,:), convection_bxheight(:,:,:)
  real(fp), allocatable :: p_temp(:,:)
  real(fp), allocatable :: ak(:), bk(:), area_y(:), ymid_r(:)
  real(fp), pointer :: p_uwnd(:,:,:), p_vwnd(:,:,:), p_xmass(:,:,:), p_ymass(:,:,:)
  integer :: jfirst, jlast, ng, mg, n_adj

  if (command_argument_count() /= 5) then
     write(*,*) 'usage: gc_transport_frontier_harness TPCORE_IN.nc VDIFF.nc CONVECTION.nc NTRACER WARMUP'
     stop 2
  endif
  call get_command_argument(1, tpcore_path)
  call get_command_argument(2, vdiff_path)
  call get_command_argument(3, convection_path)
  call get_command_argument(4, argument)
  read(argument, *, iostat=ios) ntracer
  if (ios /= 0 .or. ntracer < 1) stop 'NTRACER must be positive'
  call get_command_argument(5, argument)
  read(argument, *, iostat=ios) warmup
  if (ios /= 0 .or. warmup < 0) stop 'WARMUP must be nonnegative'

  call check(nf90_open(trim(tpcore_path), nf90_nowrite, ncid), 'open TPCORE input')
  nx = dim_len(ncid, 'lon')
  ny = dim_len(ncid, 'lat')
  nz = dim_len(ncid, 'lev')
  source_tracers = dim_len(ncid, 'tracer')
  allocate(lon(nx), lat(ny), area(nx,ny))
  allocate(tracer_template(nx,ny,nz,source_tracers))
  allocate(p1(nx,ny), p2(nx,ny), p1_base(nx,ny), p2_base(nx,ny))
  allocate(tpcore_u(nx,ny,nz), tpcore_v(nx,ny,nz))
  allocate(xmass(nx,ny,nz), ymass(nx,ny,nz), p_temp(nx,ny))
  call read_tpcore_fixture(ncid, lon, lat, area, tracer_template, p1_base, p2_base, &
                           tpcore_u, tpcore_v, dt_s)
  call check(nf90_close(ncid), 'close TPCORE input')
  call allocate_grid_storage(State_Grid, nx, ny)
  call init_grid(State_Grid, nx, ny, nz, lon, lat, area)
  call init_options(Input_Opt, int(dt_s))
  call INIT_PRESSURE(minimal_input_options(), State_Grid, rc)
  if (rc /= 0) stop 'INIT_PRESSURE failed'
  p1 = p1_base
  p2 = p2_base
  call DO_PJC_PFIX(State_Grid, dt_s, p1, p2, tpcore_u, tpcore_v, xmass, ymass)
  call allocate_state(State_Chm, State_Met, nx, ny, nz, ntracer)
  call init_chem(State_Chm, tracer_template, ntracer)

  call check(nf90_open(trim(vdiff_path), nf90_nowrite, ncid), 'open VDIFF input')
  call require_dimensions(ncid, nx, ny, nz)
  call read_vdiff_fixture(ncid, State_Chm, State_Met, surface_flux_template, ntracer)
  call check(nf90_close(ncid), 'close VDIFF input')
  allocate(sphu_base(nx,ny,nz), vdiff_bxheight(nx,ny,nz), convection_bxheight(nx,ny,nz))
  sphu_base = State_Met%SPHU
  vdiff_bxheight = State_Met%BXHEIGHT

  call check(nf90_open(trim(convection_path), nf90_nowrite, ncid), 'open convection input')
  call require_dimensions(ncid, nx, ny, nz)
  call read_convection_fixture(ncid, State_Met, convection_bxheight)
  call check(nf90_close(ncid), 'close convection input')

  call SET_TIMESTEPS(Input_Opt, int(dt_s), int(dt_s), int(dt_s), int(dt_s), &
                     int(dt_s), int(dt_s), int(dt_s))
  call Init_Vdiff(Input_Opt, State_Chm, State_Grid, rc)
  if (rc /= 0) stop 'Init_Vdiff failed'
  call Max_PblHt_For_Vdiff(Input_Opt, State_Grid, State_Met, rc)
  if (rc /= 0) stop 'Max_PblHt_For_Vdiff failed'

  allocate(ak(nz+1), bk(nz+1), area_y(ny), ymid_r(ny))
  call init_tpcore_arrays(State_Grid, area, dt_s, ak, bk, area_y, ymid_r, &
                          jfirst, jlast, ng, mg)
  n_adj = 0

  do iteration = 1, warmup
     call prepare_step()
     call run_transport_chain(started_ns, completed_ns)
  enddo

  write(output_unit, '(A,1X,I0)') 'READY', omp_get_max_threads()
  flush(output_unit)
  do
     read(input_unit, '(A)', iostat=ios) line
     if (ios /= 0) exit
     read(line, *, iostat=ios) command, target_ns
     if (ios /= 0) stop 'invalid worker command'
     if (trim(command) == 'STOP') exit
     if (trim(command) /= 'RUN') stop 'unknown worker command'
     call prepare_step()
     call wait_until(target_ns)
     call run_transport_chain(started_ns, completed_ns)
     checksum = tracer_checksum(State_Chm)
     write(output_unit, '(A,1X,I0,1X,I0,1X,ES24.16E3)') &
          'DONE', started_ns, completed_ns, checksum
     flush(output_unit)
  enddo

contains

  function minimal_input_options()
    type(OptInput) :: minimal_input_options
  end function minimal_input_options

  subroutine prepare_step()
    p1 = p1_base
    p2 = p2_base
    State_Met%SPHU = sphu_base
    State_Met%BXHEIGHT = vdiff_bxheight
  end subroutine prepare_step

  subroutine run_transport_chain(start_ns, stop_ns)
    integer(c_int64_t), intent(out) :: start_ns, stop_ns
    integer :: local_rc
    start_ns = monotonic_ns()
    call DO_PJC_PFIX(State_Grid, dt_s, p1, p2, tpcore_u, tpcore_v, xmass, ymass)
    p_uwnd => tpcore_u(:,:,nz:1:-1)
    p_vwnd => tpcore_v(:,:,nz:1:-1)
    p_xmass => xmass(:,:,nz:1:-1)
    p_ymass => ymass(:,:,nz:1:-1)
    call TPCORE_FVDAS(dt_s, Re, nx, ny, nz, jfirst, jlast, ng, mg, ntracer, &
         ak, bk, p_uwnd, p_vwnd, p1, p2, p_temp, 3, 3, 7, n_adj, &
         p_xmass, p_ymass, .true., area_y, State_Chm, State_Diag)
    p_uwnd => null()
    p_vwnd => null()
    p_xmass => null()
    p_ymass => null()
    call VDIFFDR(Input_Opt, State_Chm, State_Diag, State_Grid, State_Met, local_rc)
    if (local_rc /= 0) stop 'VDIFFDR failed'
    State_Met%BXHEIGHT = convection_bxheight
    call run_convection(local_rc)
    if (local_rc /= 0) stop 'DO_CLOUD_CONVECTION failed'
    stop_ns = monotonic_ns()
  end subroutine run_transport_chain

  subroutine run_convection(error_code)
    integer, intent(out) :: error_code
    integer :: i, j, local_rc
    real(fp), allocatable :: fsol(:,:), diag14(:,:), diag38(:,:)
    error_code = 0
    !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(I,J,LOCAL_RC,FSOL,DIAG14,DIAG38)
    allocate(fsol(nz,ntracer), diag14(nz,ntracer), diag38(nz,max(1,ntracer)))
    fsol = 0.0_fp
    diag14 = 0.0_fp
    diag38 = 0.0_fp
    !$OMP DO SCHEDULE(DYNAMIC,8) COLLAPSE(2)
    do j = 1, ny
    do i = 1, nx
       local_rc = 0
       call DO_CLOUD_CONVECTION(Input_Opt, State_Chm, State_Diag, State_Grid, State_Met, &
            i, j, State_Grid%Area_M2(i,j), fsol, dt_s, .false., diag14, .false., diag38, local_rc)
       if (local_rc /= 0) then
          !$OMP CRITICAL(gc_frontier_error)
          error_code = local_rc
          !$OMP END CRITICAL(gc_frontier_error)
       endif
    enddo
    enddo
    !$OMP END DO
    deallocate(fsol, diag14, diag38)
    !$OMP END PARALLEL
  end subroutine run_convection

  subroutine allocate_grid_storage(grid, nx_value, ny_value)
    type(GrdState), intent(inout) :: grid
    integer, intent(in) :: nx_value, ny_value
    allocate(grid%Area_M2(nx_value,ny_value), grid%XMid(nx_value,ny_value))
    allocate(grid%YMid(nx_value,ny_value), grid%YMid_R(nx_value,ny_value))
    allocate(grid%XEdge(nx_value+1,ny_value), grid%YEdge(nx_value,ny_value+1))
    allocate(grid%YEdge_R(nx_value,ny_value+1), grid%YSIN(nx_value,ny_value+1))
  end subroutine allocate_grid_storage

  subroutine allocate_state(chm, met, nx_value, ny_value, nz_value, tracer_count)
    type(ChmState), intent(inout) :: chm
    type(MetState), intent(inout) :: met
    integer, intent(in) :: nx_value, ny_value, nz_value, tracer_count
    integer :: n
    allocate(met%U(nx_value,ny_value,nz_value), met%V(nx_value,ny_value,nz_value))
    allocate(met%T(nx_value,ny_value,nz_value), met%SPHU(nx_value,ny_value,nz_value))
    allocate(met%PMID(nx_value,ny_value,nz_value), met%PEDGE(nx_value,ny_value,nz_value+1))
    allocate(met%TV(nx_value,ny_value,nz_value), met%BXHEIGHT(nx_value,ny_value,nz_value))
    allocate(met%AD(nx_value,ny_value,nz_value))
    allocate(met%PBLH(nx_value,ny_value), met%PBL_TOP_m(nx_value,ny_value))
    allocate(met%HFLUX(nx_value,ny_value), met%EFLUX(nx_value,ny_value), met%USTAR(nx_value,ny_value))
    allocate(met%CMFMC(nx_value,ny_value,nz_value+1), met%DQRCU(nx_value,ny_value,nz_value))
    allocate(met%DTRAIN(nx_value,ny_value,nz_value), met%PFICU(nx_value,ny_value,nz_value+1))
    allocate(met%PFLCU(nx_value,ny_value,nz_value+1), met%REEVAPCN(nx_value,ny_value,nz_value))
    allocate(met%DELP_DRY(nx_value,ny_value,nz_value), met%DELP(nx_value,ny_value,nz_value))
    allocate(met%PRECCON(nx_value,ny_value))
    allocate(chm%SurfaceFlux(nx_value,ny_value,tracer_count))
    allocate(chm%Map_Advect(tracer_count), chm%Map_Tracer(tracer_count))
    allocate(chm%Species(tracer_count), chm%SpcData(tracer_count))
    allocate(chm%H2O2AfterChem(nx_value,ny_value,nz_value), chm%SO2AfterChem(nx_value,ny_value,nz_value))
    met%CMFMC = 0.0_fp
    met%PFICU = 0.0_fp
    met%PFLCU = 0.0_fp
    chm%H2O2AfterChem = 0.0_fp
    chm%SO2AfterChem = 0.0_fp
    do n = 1, tracer_count
       allocate(chm%Species(n)%Conc(nx_value,ny_value,nz_value))
       allocate(chm%SpcData(n)%Info)
       chm%Species(n)%Units = 0
       chm%Species(n)%Previous_Units = 0
    enddo
  end subroutine allocate_state

  subroutine init_grid(grid, nx_value, ny_value, nz_value, longitude, latitude, cell_area)
    type(GrdState), intent(inout) :: grid
    integer, intent(in) :: nx_value, ny_value, nz_value
    real(fp), intent(in) :: longitude(nx_value), latitude(ny_value), cell_area(nx_value,ny_value)
    integer :: i, j
    grid%GridRes = '2.0x2.5'
    grid%DX = longitude(2) - longitude(1)
    grid%DY = latitude(2) - latitude(1)
    grid%NX = nx_value
    grid%NY = ny_value
    grid%NZ = nz_value
    grid%GlobalNX = nx_value
    grid%GlobalNY = ny_value
    grid%NativeNZ = nz_value
    grid%XMin = minval(longitude)
    grid%XMax = maxval(longitude)
    grid%YMin = minval(latitude)
    grid%YMax = maxval(latitude)
    grid%HalfPolar = .true.
    grid%Center180 = .false.
    grid%NestedGrid = .false.
    grid%NorthBuffer = 0
    grid%SouthBuffer = 0
    grid%EastBuffer = 0
    grid%WestBuffer = 0
    grid%Area_M2 = cell_area
    grid%XEdge = 0.0_fp
    grid%YEdge = 0.0_fp
    grid%YEdge_R = 0.0_fp
    grid%YSIN = 0.0_fp
    do j = 1, ny_value
    do i = 1, nx_value
       grid%XMid(i,j) = longitude(i)
       grid%YMid(i,j) = latitude(j)
       grid%YMid_R(i,j) = latitude(j) * acos(-1.0_fp) / 180.0_fp
    enddo
    enddo
  end subroutine init_grid

  subroutine init_options(options, timestep)
    type(OptInput), intent(inout) :: options
    integer, intent(in) :: timestep
    options%DryRun = .false.
    options%Verbose = .false.
    options%amIRoot = .false.
    options%LTURB = .true.
    options%LNLPBL = .true.
    options%PBL_DRYDEP = .false.
    options%LCONV = .true.
    options%ITS_A_MERCURY_SIM = .false.
    options%Reconstruct_Conv_Precip_Flux = .false.
    options%MetField = 'MERRA2'
    options%TS_CONV = timestep
    options%TS_DYN = timestep
  end subroutine init_options

  subroutine init_chem(chm, template, tracer_count)
    type(ChmState), intent(inout) :: chm
    real(fp), intent(in) :: template(:,:,:,:)
    integer, intent(in) :: tracer_count
    integer :: n, source
    chm%nSpecies = tracer_count
    chm%nAdvect = tracer_count
    chm%nTracer = tracer_count
    chm%nWetDep = 0
    do n = 1, tracer_count
       source = mod(n - 1, size(template,4)) + 1
       chm%Species(n)%Conc = template(:,:,:,source) + real(n - 1, fp) * 1.0e-7_fp
       chm%Map_Advect(n) = n
       chm%Map_Tracer(n) = n
       chm%SpcData(n)%Info%ModelId = n
       chm%SpcData(n)%Info%AdvectId = n
       chm%SpcData(n)%Info%TracerId = n
       chm%SpcData(n)%Info%WetDepId = 0
       write(chm%SpcData(n)%Info%Name, '(A,I5.5)') 'frontier_', n
       chm%SpcData(n)%Info%Is_Advected = .true.
       chm%SpcData(n)%Info%Is_Tracer = .true.
       chm%SpcData(n)%Info%Is_WetDep = .false.
       chm%SpcData(n)%Info%Is_Hg2 = .false.
       chm%SpcData(n)%Info%Is_HgP = .false.
    enddo
    State_Diag%Archive_AdvFluxZonal = .false.
    State_Diag%Archive_AdvFluxMerid = .false.
    State_Diag%Archive_AdvFluxVert = .false.
    State_Diag%Archive_CloudConvFlux = .false.
    State_Diag%Archive_WetLossConv = .false.
    State_Diag%Archive_SatDiagnWetLossConv = .false.
  end subroutine init_chem

  subroutine init_tpcore_arrays(grid, cell_area, timestep, ak_value, bk_value, area_value, ymid_value, &
                                first_j, last_j, ghost_x, ghost_y)
    type(GrdState), intent(in) :: grid
    real(fp), intent(in) :: cell_area(:,:), timestep
    real(fp), intent(out) :: ak_value(:), bk_value(:), area_value(:), ymid_value(:)
    integer, intent(out) :: first_j, last_j, ghost_x, ghost_y
    integer :: level, source_level, status
    do level = 1, grid%NZ + 1
       source_level = grid%NZ + 2 - level
       ak_value(level) = GET_AP(source_level)
       bk_value(level) = GET_BP(source_level)
    enddo
    area_value = cell_area(1,:)
    ymid_value = grid%YMid_R(1,:)
    ghost_x = 0
    ghost_y = 0
    call INIT_TPCORE(grid%NX, grid%NY, grid%NZ, first_j, last_j, ghost_x, ghost_y, &
                     timestep, Re, ymid_value, .false., status)
    if (status /= 0) stop 'INIT_TPCORE failed'
  end subroutine init_tpcore_arrays

  subroutine read_tpcore_fixture(file_id, longitude, latitude, cell_area, template, &
                                 pressure1, pressure2, wind_u, wind_v, timestep)
    integer, intent(in) :: file_id
    real(fp), intent(out) :: longitude(:), latitude(:), cell_area(:,:), template(:,:,:,:)
    real(fp), intent(out) :: pressure1(:,:), pressure2(:,:), wind_u(:,:,:), wind_v(:,:,:), timestep
    call get_var_1d(file_id, 'lon', longitude)
    call get_var_1d(file_id, 'lat', latitude)
    call get_var_2d(file_id, 'area_m2', cell_area)
    call get_var_2d(file_id, 'p1_hpa', pressure1)
    call get_var_2d(file_id, 'p2_hpa', pressure2)
    call get_var_3d(file_id, 'u_m_s', wind_u)
    call get_var_3d(file_id, 'v_m_s', wind_v)
    call get_var_4d(file_id, 'tracer_conc', template)
    call check(nf90_get_att(file_id, nf90_global, 'dt_s', timestep), 'read TPCORE dt_s')
  end subroutine read_tpcore_fixture

  subroutine read_vdiff_fixture(file_id, chm, met, flux_template, tracer_count)
    integer, intent(in) :: file_id, tracer_count
    type(ChmState), intent(inout) :: chm
    type(MetState), intent(inout) :: met
    real(fp), allocatable, intent(out) :: flux_template(:,:,:)
    integer :: source_count, n, source
    source_count = dim_len(file_id, 'tracer')
    allocate(flux_template(nx,ny,source_count))
    call get_var_3d_tracer(file_id, 'surface_flux_kg_m2_s', flux_template)
    do n = 1, tracer_count
       source = mod(n - 1, source_count) + 1
       chm%SurfaceFlux(:,:,n) = flux_template(:,:,source)
    enddo
    call get_var_3d(file_id, 'u_m_s', met%U)
    call get_var_3d(file_id, 'v_m_s', met%V)
    call get_var_3d(file_id, 'temperature_k', met%T)
    call get_var_3d(file_id, 'specific_humidity_kg_kg', met%SPHU)
    met%SPHU = met%SPHU * 1.0e3_fp
    call get_var_3d(file_id, 'pmid_hpa', met%PMID)
    call get_var_3d(file_id, 'pedge_hpa', met%PEDGE)
    call get_var_3d(file_id, 'virtual_temperature_k', met%TV)
    call get_var_3d(file_id, 'bxheight_m', met%BXHEIGHT)
    call get_var_3d(file_id, 'dry_air_mass_kg', met%AD)
    call get_var_2d(file_id, 'pbl_top_m', met%PBL_TOP_m)
    met%PBLH = met%PBL_TOP_m
    call get_var_2d(file_id, 'hflux_w_m2', met%HFLUX)
    call get_var_2d(file_id, 'eflux_w_m2', met%EFLUX)
    call get_var_2d(file_id, 'ustar_m_s', met%USTAR)
  end subroutine read_vdiff_fixture

  subroutine read_convection_fixture(file_id, met, conv_bxheight)
    integer, intent(in) :: file_id
    type(MetState), intent(inout) :: met
    real(fp), intent(out) :: conv_bxheight(:,:,:)
    real(fp), allocatable :: upper(:,:,:)
    allocate(upper(nx,ny,nz))
    call get_var_3d(file_id, 'cmfmc_kg_m2_s', upper)
    met%CMFMC(:,:,2:nz+1) = upper
    call get_var_3d(file_id, 'dtrain_kg_m2_s', met%DTRAIN)
    call get_var_3d(file_id, 'dqrcu_kg_kg_s', met%DQRCU)
    call get_var_3d(file_id, 'reevapcn_kg_kg_s', met%REEVAPCN)
    call get_var_3d(file_id, 'delp_dry_hpa', met%DELP_DRY)
    call get_var_3d(file_id, 'delp_hpa', met%DELP)
    call get_var_3d(file_id, 'bxheight_m', conv_bxheight)
    call get_var_3d(file_id, 'temperature_k', met%T)
    call get_var_3d(file_id, 'pficu_kg_m2_s', upper)
    met%PFICU(:,:,2:nz+1) = upper
    call get_var_3d(file_id, 'pflcu_kg_m2_s', upper)
    met%PFLCU(:,:,2:nz+1) = upper
    call get_var_2d(file_id, 'precccon_mm_day', met%PRECCON)
  end subroutine read_convection_fixture

  subroutine require_dimensions(file_id, expected_x, expected_y, expected_z)
    integer, intent(in) :: file_id, expected_x, expected_y, expected_z
    if (dim_len(file_id, 'lon') /= expected_x) stop 'fixture longitude mismatch'
    if (dim_len(file_id, 'lat') /= expected_y) stop 'fixture latitude mismatch'
    if (dim_len(file_id, 'lev') /= expected_z) stop 'fixture level mismatch'
  end subroutine require_dimensions

  real(fp) function tracer_checksum(chm)
    type(ChmState), intent(in) :: chm
    integer :: n
    tracer_checksum = 0.0_fp
    do n = 1, chm%nTracer
       tracer_checksum = tracer_checksum + sum(chm%Species(n)%Conc)
    enddo
    tracer_checksum = tracer_checksum / real(nx * ny * nz * chm%nTracer, fp)
  end function tracer_checksum

  integer(c_int64_t) function monotonic_ns()
    type(c_timespec) :: value
    integer(c_int) :: status
    status = c_clock_gettime(CLOCK_MONOTONIC, value)
    if (status /= 0) stop 'clock_gettime failed'
    monotonic_ns = int(value%tv_sec, c_int64_t) * 1000000000_c_int64_t &
                 + int(value%tv_nsec, c_int64_t)
  end function monotonic_ns

  subroutine wait_until(deadline_ns)
    integer(c_int64_t), intent(in) :: deadline_ns
    integer(c_int64_t) :: remaining
    integer(c_int) :: status, sleep_us
    do
       remaining = deadline_ns - monotonic_ns()
       if (remaining <= 0_c_int64_t) exit
       if (remaining > 2000000_c_int64_t) then
          sleep_us = int((remaining - 1000000_c_int64_t) / 1000_c_int64_t, c_int)
          status = c_usleep(sleep_us)
       endif
    enddo
  end subroutine wait_until

  integer function dim_len(file_id, name)
    integer, intent(in) :: file_id
    character(len=*), intent(in) :: name
    integer :: dimension_id
    call check(nf90_inq_dimid(file_id, name, dimension_id), 'inq dim '//trim(name))
    call check(nf90_inquire_dimension(file_id, dimension_id, len=dim_len), 'inq dim len '//trim(name))
  end function dim_len

  subroutine get_var_1d(file_id, name, values)
    integer, intent(in) :: file_id
    character(len=*), intent(in) :: name
    real(fp), intent(out) :: values(:)
    integer :: variable_id
    call check(nf90_inq_varid(file_id, name, variable_id), 'inq var '//trim(name))
    call check(nf90_get_var(file_id, variable_id, values), 'get var '//trim(name))
  end subroutine get_var_1d

  subroutine get_var_2d(file_id, name, values)
    integer, intent(in) :: file_id
    character(len=*), intent(in) :: name
    real(fp), intent(out) :: values(:,:)
    integer :: variable_id
    call check(nf90_inq_varid(file_id, name, variable_id), 'inq var '//trim(name))
    call check(nf90_get_var(file_id, variable_id, values), 'get var '//trim(name))
  end subroutine get_var_2d

  subroutine get_var_3d(file_id, name, values)
    integer, intent(in) :: file_id
    character(len=*), intent(in) :: name
    real(fp), intent(out) :: values(:,:,:)
    integer :: variable_id
    call check(nf90_inq_varid(file_id, name, variable_id), 'inq var '//trim(name))
    call check(nf90_get_var(file_id, variable_id, values), 'get var '//trim(name))
    values = values(:,:,size(values,3):1:-1)
  end subroutine get_var_3d

  subroutine get_var_3d_tracer(file_id, name, values)
    integer, intent(in) :: file_id
    character(len=*), intent(in) :: name
    real(fp), intent(out) :: values(:,:,:)
    integer :: variable_id, n
    real(fp), allocatable :: canonical(:,:,:)
    allocate(canonical(size(values,3),size(values,1),size(values,2)))
    call check(nf90_inq_varid(file_id, name, variable_id), 'inq var '//trim(name))
    call check(nf90_get_var(file_id, variable_id, canonical), 'get var '//trim(name))
    do n = 1, size(values,3)
       values(:,:,n) = canonical(n,:,:)
    enddo
  end subroutine get_var_3d_tracer

  subroutine get_var_4d(file_id, name, values)
    integer, intent(in) :: file_id
    character(len=*), intent(in) :: name
    real(fp), intent(out) :: values(:,:,:,:)
    integer :: variable_id, n
    real(fp), allocatable :: canonical(:,:,:,:)
    allocate(canonical(size(values,4),size(values,1),size(values,2),size(values,3)))
    call check(nf90_inq_varid(file_id, name, variable_id), 'inq var '//trim(name))
    call check(nf90_get_var(file_id, variable_id, canonical), 'get var '//trim(name))
    do n = 1, size(values,4)
       values(:,:,:,n) = canonical(n,:,:,size(values,3):1:-1)
    enddo
  end subroutine get_var_4d

  subroutine check(status, context)
    integer, intent(in) :: status
    character(len=*), intent(in) :: context
    if (status /= nf90_noerr) then
       write(*,*) trim(context), ': ', trim(nf90_strerror(status))
       stop 10
    endif
  end subroutine check

end program gc_transport_frontier_harness
