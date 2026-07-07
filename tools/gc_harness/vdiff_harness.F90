program vdiff_harness
  use netcdf
  use precision_mod, only: fp
  use input_opt_mod, only: OptInput
  use state_grid_mod, only: GrdState
  use state_chm_mod, only: ChmState
  use state_diag_mod, only: DgnState
  use state_met_mod, only: MetState
  use time_mod, only: SET_TIMESTEPS
  use vdiff_mod, only: Init_Vdiff, Max_PblHt_For_Vdiff, VDIFFDR
  use vdiff_trace_mod, only: Vdiff_Trace_Get
  implicit none

  type(OptInput) :: Input_Opt
  type(GrdState) :: State_Grid
  type(ChmState) :: State_Chm
  type(DgnState) :: State_Diag
  type(MetState) :: State_Met
  character(len=1024) :: input_path, output_path
  integer :: ncid, nx, ny, nz, nilev, ntracer, rc
  real(fp) :: dt_s
  real(fp), allocatable :: lon(:), lat(:), area(:,:)
  real(fp), allocatable :: tracer(:,:,:,:), tracer0(:,:,:,:), sflux(:,:,:)
  real(fp), allocatable :: kvh(:,:,:), kvm(:,:,:), tpert(:,:), qpert(:,:)

  if (command_argument_count() /= 2) then
     write(*,*) 'usage: vdiff_harness INPUT.nc OUTPUT.nc'
     stop 2
  endif
  call get_command_argument(1, input_path)
  call get_command_argument(2, output_path)

  call check(nf90_open(trim(input_path), nf90_nowrite, ncid), 'open input')
  nx = dim_len(ncid, 'lon')
  ny = dim_len(ncid, 'lat')
  nz = dim_len(ncid, 'lev')
  nilev = dim_len(ncid, 'ilev')
  ntracer = dim_len(ncid, 'tracer')
  if (nilev /= nz + 1) stop 'ilev must equal lev + 1'

  allocate(lon(nx), lat(ny), area(nx,ny))
  allocate(tracer(nx,ny,nz,ntracer), tracer0(nx,ny,nz,ntracer), sflux(nx,ny,ntracer))
  allocate(kvh(nx,ny,nz+1), kvm(nx,ny,nz+1), tpert(nx,ny), qpert(nx,ny))
  call allocate_state(State_Grid, State_Chm, State_Met, nx, ny, nz, ntracer)
  call read_fixture(ncid, lon, lat, area, tracer, sflux, State_Met, dt_s)
  call check(nf90_close(ncid), 'close input')
  tracer0 = tracer

  call init_grid(State_Grid, nx, ny, nz, lon, lat, area)
  call init_options(Input_Opt, int(dt_s))
  call init_chem(State_Chm, tracer, sflux, ntracer)
  call SET_TIMESTEPS(Input_Opt, int(dt_s), int(dt_s), int(dt_s), int(dt_s), int(dt_s), int(dt_s), int(dt_s))
  call Init_Vdiff(Input_Opt, State_Chm, State_Grid, rc)
  if (rc /= 0) stop 'Init_Vdiff failed'
  call Max_PblHt_For_Vdiff(Input_Opt, State_Grid, State_Met, rc)
  if (rc /= 0) stop 'Max_PblHt_For_Vdiff failed'
  call VDIFFDR(Input_Opt, State_Chm, State_Diag, State_Grid, State_Met, rc)
  if (rc /= 0) stop 'VDIFFDR failed'
  call Vdiff_Trace_Get(kvh, kvm, tpert, qpert)
  call copy_tracers_from_state(State_Chm, tracer)
  call write_output(trim(output_path), tracer0, tracer, State_Met, kvh, kvm, tpert, qpert)

contains

  subroutine allocate_state(grid, chm, met, nx, ny, nz, ntracer)
    type(GrdState), intent(inout) :: grid
    type(ChmState), intent(inout) :: chm
    type(MetState), intent(inout) :: met
    integer, intent(in) :: nx, ny, nz, ntracer
    integer :: n
    allocate(met%U(nx,ny,nz), met%V(nx,ny,nz), met%T(nx,ny,nz), met%SPHU(nx,ny,nz))
    allocate(met%PMID(nx,ny,nz), met%PEDGE(nx,ny,nz+1), met%TV(nx,ny,nz))
    allocate(met%BXHEIGHT(nx,ny,nz), met%AD(nx,ny,nz))
    allocate(met%PBLH(nx,ny), met%PBL_TOP_m(nx,ny), met%HFLUX(nx,ny), met%EFLUX(nx,ny), met%USTAR(nx,ny))
    allocate(grid%Area_M2(nx,ny), grid%XMid(nx,ny), grid%YMid(nx,ny), grid%YMid_R(nx,ny))
    allocate(chm%SurfaceFlux(nx,ny,ntracer), chm%Species(ntracer))
    do n = 1, ntracer
       allocate(chm%Species(n)%Conc(nx,ny,nz))
       chm%Species(n)%Units = 0
       chm%Species(n)%Previous_Units = 0
    enddo
  end subroutine allocate_state

  subroutine init_grid(grid, nx, ny, nz, lon, lat, area)
    type(GrdState), intent(inout) :: grid
    integer, intent(in) :: nx, ny, nz
    real(fp), intent(in) :: lon(nx), lat(ny), area(nx,ny)
    integer :: i, j
    grid%GridRes = '2.0x2.5'
    grid%NX = nx
    grid%NY = ny
    grid%NZ = nz
    grid%GlobalNX = nx
    grid%GlobalNY = ny
    grid%NativeNZ = nz
    grid%HalfPolar = .true.
    grid%NestedGrid = .false.
    grid%Area_M2 = area
    do j = 1, ny
    do i = 1, nx
       grid%XMid(i,j) = lon(i)
       grid%YMid(i,j) = lat(j)
       grid%YMid_R(i,j) = lat(j) * acos(-1.0_fp) / 180.0_fp
    enddo
    enddo
  end subroutine init_grid

  subroutine init_options(opt, dt)
    type(OptInput), intent(inout) :: opt
    integer, intent(in) :: dt
    opt%DryRun = .false.
    opt%Verbose = .false.
    opt%amIRoot = .false.
    opt%LTURB = .true.
    opt%LNLPBL = .true.
    opt%PBL_DRYDEP = .false.
    opt%TS_CONV = dt
    opt%TS_DYN = dt
  end subroutine init_options

  subroutine init_chem(chm, tracer, sflux, ntracer)
    type(ChmState), intent(inout) :: chm
    real(fp), intent(in) :: tracer(:,:,:,:), sflux(:,:,:)
    integer, intent(in) :: ntracer
    integer :: n
    chm%nSpecies = ntracer
    chm%nAdvect = ntracer
    chm%nTracer = ntracer
    chm%SurfaceFlux = sflux
    do n = 1, ntracer
       chm%Species(n)%Conc = tracer(:,:,:,n)
    enddo
  end subroutine init_chem

  subroutine copy_tracers_from_state(chm, tracer)
    type(ChmState), intent(in) :: chm
    real(fp), intent(out) :: tracer(:,:,:,:)
    integer :: n
    do n = 1, size(tracer,4)
       tracer(:,:,:,n) = chm%Species(n)%Conc
    enddo
  end subroutine copy_tracers_from_state

  subroutine read_fixture(ncid, lon, lat, area, tracer, sflux, met, dt_s)
    integer, intent(in) :: ncid
    real(fp), intent(out) :: lon(:), lat(:), area(:,:), tracer(:,:,:,:), sflux(:,:,:), dt_s
    type(MetState), intent(inout) :: met
    call get_var_1d(ncid, 'lon', lon)
    call get_var_1d(ncid, 'lat', lat)
    call get_var_2d(ncid, 'area_m2', area)
    call get_var_4d(ncid, 'tracer_conc', tracer)
    call get_var_3d(ncid, 'surface_flux_kg_m2_s', sflux)
    call get_var_3d(ncid, 'u_m_s', met%U)
    call get_var_3d(ncid, 'v_m_s', met%V)
    call get_var_3d(ncid, 'temperature_k', met%T)
    call get_var_3d(ncid, 'specific_humidity_kg_kg', met%SPHU)
    met%SPHU = met%SPHU * 1.0e3_fp
    call get_var_3d(ncid, 'pmid_hpa', met%PMID)
    call get_var_3d(ncid, 'pedge_hpa', met%PEDGE)
    call get_var_3d(ncid, 'virtual_temperature_k', met%TV)
    call get_var_3d(ncid, 'bxheight_m', met%BXHEIGHT)
    call get_var_3d(ncid, 'dry_air_mass_kg', met%AD)
    call get_var_2d(ncid, 'pbl_top_m', met%PBL_TOP_m)
    met%PBLH = met%PBL_TOP_m
    call get_var_2d(ncid, 'hflux_w_m2', met%HFLUX)
    call get_var_2d(ncid, 'eflux_w_m2', met%EFLUX)
    call get_var_2d(ncid, 'ustar_m_s', met%USTAR)
    call check(nf90_get_att(ncid, nf90_global, 'dt_s', dt_s), 'read dt_s')
  end subroutine read_fixture

  subroutine write_output(path, tracer0, tracer, met, kvh, kvm, tpert, qpert)
    character(len=*), intent(in) :: path
    real(fp), intent(in) :: tracer0(:,:,:,:), tracer(:,:,:,:), kvh(:,:,:), kvm(:,:,:), tpert(:,:), qpert(:,:)
    type(MetState), intent(in) :: met
    integer :: ncid, lon_dim, lat_dim, lev_dim, ilev_dim, tracer_dim
    integer :: id
    real(fp) :: mass0(size(tracer,4)), mass1(size(tracer,4))
    integer :: n
    integer :: tracer_id, sphu_id, kvh_id, kvm_id, pbl_id, tpert_id, qpert_id, mass0_id, mass1_id
    do n = 1, size(tracer,4)
       mass0(n) = sum(tracer0(:,:,:,n) * met%AD)
       mass1(n) = sum(tracer(:,:,:,n) * met%AD)
    enddo
    call check(nf90_create(path, nf90_clobber, ncid), 'create output')
    call check(nf90_def_dim(ncid, 'lon', size(tracer,1), lon_dim), 'def lon')
    call check(nf90_def_dim(ncid, 'lat', size(tracer,2), lat_dim), 'def lat')
    call check(nf90_def_dim(ncid, 'lev', size(tracer,3), lev_dim), 'def lev')
    call check(nf90_def_dim(ncid, 'ilev', size(kvh,3), ilev_dim), 'def ilev')
    call check(nf90_def_dim(ncid, 'tracer', size(tracer,4), tracer_dim), 'def tracer')
    call check(nf90_put_att(ncid, nf90_global, 'harness', 'vdiffdr-output-v1'), 'put harness')
    call check(nf90_put_att(ncid, nf90_global, 'negative_count_before_clip', 0), 'put neg before')
    call check(nf90_put_att(ncid, nf90_global, 'negative_count_after_clip', count(tracer < 0.0_fp)), 'put neg after')
    call check(nf90_def_var(ncid, 'tracer_conc_after', nf90_double, &
         (/ lon_dim, lat_dim, lev_dim, tracer_dim /), tracer_id), 'def tracer')
    call check(nf90_def_var(ncid, 'specific_humidity_after', nf90_double, &
         (/ lon_dim, lat_dim, lev_dim /), sphu_id), 'def sphu')
    call check(nf90_def_var(ncid, 'kvh_m2_s', nf90_double, (/ lon_dim, lat_dim, ilev_dim /), kvh_id), 'def kvh')
    call check(nf90_def_var(ncid, 'kvm_m2_s', nf90_double, (/ lon_dim, lat_dim, ilev_dim /), kvm_id), 'def kvm')
    call check(nf90_def_var(ncid, 'pbl_top_m', nf90_double, (/ lon_dim, lat_dim /), pbl_id), 'def pbl')
    call check(nf90_def_var(ncid, 'tpert_k', nf90_double, (/ lon_dim, lat_dim /), tpert_id), 'def tpert')
    call check(nf90_def_var(ncid, 'qpert_kg_kg', nf90_double, (/ lon_dim, lat_dim /), qpert_id), 'def qpert')
    call check(nf90_def_var(ncid, 'initial_tracer_mass', nf90_double, (/ tracer_dim /), mass0_id), 'def mass0')
    call check(nf90_def_var(ncid, 'final_tracer_mass', nf90_double, (/ tracer_dim /), mass1_id), 'def mass1')
    call check(nf90_enddef(ncid), 'enddef output')
    call check(nf90_put_var(ncid, tracer_id, tracer), 'put tracer')
    call check(nf90_put_var(ncid, sphu_id, met%SPHU * 1.0e-3_fp), 'put sphu')
    call check(nf90_put_var(ncid, kvh_id, kvh), 'put kvh')
    call check(nf90_put_var(ncid, kvm_id, kvm), 'put kvm')
    call check(nf90_put_var(ncid, pbl_id, met%PBL_TOP_m), 'put pbl')
    call check(nf90_put_var(ncid, tpert_id, tpert), 'put tpert')
    call check(nf90_put_var(ncid, qpert_id, qpert), 'put qpert')
    call check(nf90_put_var(ncid, mass0_id, mass0), 'put mass0')
    call check(nf90_put_var(ncid, mass1_id, mass1), 'put mass1')
    call check(nf90_close(ncid), 'close output')
  end subroutine write_output

  integer function dim_len(ncid, name)
    integer, intent(in) :: ncid
    character(len=*), intent(in) :: name
    integer :: dimid
    call check(nf90_inq_dimid(ncid, name, dimid), 'inq dim '//trim(name))
    call check(nf90_inquire_dimension(ncid, dimid, len=dim_len), 'inq dim len '//trim(name))
  end function dim_len

  subroutine get_var_1d(ncid, name, values)
    integer, intent(in) :: ncid
    character(len=*), intent(in) :: name
    real(fp), intent(out) :: values(:)
    integer :: varid
    call check(nf90_inq_varid(ncid, name, varid), 'inq var '//trim(name))
    call check(nf90_get_var(ncid, varid, values), 'get var '//trim(name))
  end subroutine get_var_1d

  subroutine get_var_2d(ncid, name, values)
    integer, intent(in) :: ncid
    character(len=*), intent(in) :: name
    real(fp), intent(out) :: values(:,:)
    integer :: varid
    call check(nf90_inq_varid(ncid, name, varid), 'inq var '//trim(name))
    call check(nf90_get_var(ncid, varid, values), 'get var '//trim(name))
  end subroutine get_var_2d

  subroutine get_var_3d(ncid, name, values)
    integer, intent(in) :: ncid
    character(len=*), intent(in) :: name
    real(fp), intent(out) :: values(:,:,:)
    integer :: varid
    call check(nf90_inq_varid(ncid, name, varid), 'inq var '//trim(name))
    call check(nf90_get_var(ncid, varid, values), 'get var '//trim(name))
  end subroutine get_var_3d

  subroutine get_var_4d(ncid, name, values)
    integer, intent(in) :: ncid
    character(len=*), intent(in) :: name
    real(fp), intent(out) :: values(:,:,:,:)
    integer :: varid
    call check(nf90_inq_varid(ncid, name, varid), 'inq var '//trim(name))
    call check(nf90_get_var(ncid, varid, values), 'get var '//trim(name))
  end subroutine get_var_4d

  subroutine check(status, context)
    integer, intent(in) :: status
    character(len=*), intent(in) :: context
    if (status /= nf90_noerr) then
       write(*,*) trim(context), ': ', trim(nf90_strerror(status))
       stop 10
    endif
  end subroutine check

end program vdiff_harness
