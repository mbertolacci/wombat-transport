! Minimal GEOS-Chem transport operator harness.
!
! This executable is intentionally full-state-shaped but narrow: it creates the
! grid state needed by DO_PJC_PFIX/TPCORE_FVDAS and reads operator arrays from
! a NetCDF fixture written by wombat_transport.gc_harness. If the input has no
! tracer dimension it runs PJC only; otherwise it also runs one TPCORE step.

program pjc_pfix_harness
  use netcdf
  use precision_mod, only: fp
  use state_grid_mod, only: GrdState
  use state_chm_mod, only: ChmState
  use state_diag_mod, only: DgnState
  use pressure_mod, only: INIT_PRESSURE, GET_AP, GET_BP
  use pjc_pfix_mod, only: DO_PJC_PFIX
  use tpcore_fvdas_mod, only: INIT_TPCORE, TPCORE_FVDAS
  use PhysConstants, only: Re
  implicit none

  type(GrdState) :: State_Grid
  type(ChmState) :: State_Chm
  type(DgnState) :: State_Diag
  character(len=1024) :: input_path
  character(len=1024) :: output_path
  integer :: argc
  integer :: ncid
  integer :: nx, ny, nz, nilev, ntracer
  integer :: rc
  real(fp) :: dt_s
  logical :: has_tracers
  real(fp), allocatable :: lon(:), lat(:), hyai(:), hybi(:)
  real(fp), allocatable :: area(:,:), p1(:,:), p2(:,:)
  real(fp), allocatable :: u(:,:,:), v(:,:,:), xmass(:,:,:), ymass(:,:,:)
  real(fp), allocatable :: tracer(:,:,:,:), ps(:,:)

  argc = command_argument_count()
  if (argc /= 2) then
     write(*,*) 'usage: pjc_pfix_harness INPUT.nc OUTPUT.nc'
     stop 2
  endif
  call get_command_argument(1, input_path)
  call get_command_argument(2, output_path)

  call check(nf90_open(trim(input_path), nf90_nowrite, ncid), 'open input')
  call read_dimensions(ncid, nx, ny, nz, nilev, ntracer)
  has_tracers = ntracer > 0
  allocate(lon(nx), lat(ny), hyai(nilev), hybi(nilev))
  allocate(area(nx,ny), p1(nx,ny), p2(nx,ny))
  allocate(u(nx,ny,nz), v(nx,ny,nz), xmass(nx,ny,nz), ymass(nx,ny,nz))
  allocate(ps(nx,ny), tracer(nx,ny,nz,max(1,ntracer)))
  call read_fixture(ncid, lon, lat, hyai, hybi, area, p1, p2, u, v, dt_s)
  if (has_tracers) then
     call get_var_4d(ncid, 'tracer_conc', tracer)
  endif
  call check(nf90_close(ncid), 'close input')

  call init_grid_state(State_Grid, nx, ny, nz, lon, lat, area)

  ! DO_PJC_PFIX obtains Ap/Bp through pressure_mod. INIT_PRESSURE uses the
  ! standard GEOS-Chem vertical coordinate for this configured 47-level grid.
  ! If this ever diverges from the fixture hyai/hybi, add an explicit
  ! Accept_External_ApBp path in a build that exposes it.
  call INIT_PRESSURE(minimal_input_options(), State_Grid, rc)
  if (rc /= 0) then
     write(*,*) 'INIT_PRESSURE failed: ', rc
     stop 3
  endif

  call DO_PJC_PFIX( &
       State_Grid, dt_s, p1, p2, u, v, xmass, ymass )

  if (has_tracers) then
     call run_tpcore_step(State_Grid, State_Chm, State_Diag, dt_s, p1, p2, u, v, xmass, ymass, area, tracer, ps)
  endif

  call write_output(trim(output_path), xmass, ymass, has_tracers, tracer, ps)

contains

  ! INIT_PRESSURE requires OptInput in the interface, but the 47-level branch
  ! used by this harness only needs State_Grid%NZ to choose Ap/Bp constants.
  ! Later harness modes should replace this with populated options when the
  ! called operator actually reads them.
  function minimal_input_options()
    use input_opt_mod, only: OptInput
    type(OptInput) :: minimal_input_options
  end function minimal_input_options

  subroutine init_grid_state(grid, nx, ny, nz, lon, lat, area)
    type(GrdState), intent(inout) :: grid
    integer, intent(in) :: nx, ny, nz
    real(fp), intent(in) :: lon(nx), lat(ny), area(nx,ny)
    integer :: i, j

    grid%GridRes = '2.0x2.5'
    grid%DX = lon(2) - lon(1)
    grid%DY = lat(2) - lat(1)
    grid%NX = nx
    grid%NY = ny
    grid%NZ = nz
    grid%GlobalNX = nx
    grid%GlobalNY = ny
    grid%NativeNZ = nz
    grid%XMin = minval(lon)
    grid%XMax = maxval(lon)
    grid%YMin = minval(lat)
    grid%YMax = maxval(lat)
    grid%HalfPolar = .true.
    grid%Center180 = .false.
    grid%NestedGrid = .false.
    grid%NorthBuffer = 0
    grid%SouthBuffer = 0
    grid%EastBuffer = 0
    grid%WestBuffer = 0

    allocate(grid%XMid(nx,ny), grid%YMid(nx,ny), grid%Area_M2(nx,ny))
    allocate(grid%XEdge(nx+1,ny), grid%YEdge(nx,ny+1))
    allocate(grid%YMid_R(nx,ny), grid%YEdge_R(nx,ny+1), grid%YSIN(nx,ny+1))
    do j = 1, ny
       do i = 1, nx
          grid%XMid(i,j) = lon(i)
          grid%YMid(i,j) = lat(j)
          grid%YMid_R(i,j) = lat(j) * acos(-1.0_fp) / 180.0_fp
          grid%Area_M2(i,j) = area(i,j)
       enddo
    enddo
    grid%XEdge = 0.0_fp
    grid%YEdge = 0.0_fp
    grid%YEdge_R = 0.0_fp
    grid%YSIN = 0.0_fp
  end subroutine init_grid_state

  subroutine run_tpcore_step(grid, chm, diag, dt_s, p1, p2, u, v, xmass, ymass, area, tracer, ps)
    type(GrdState), intent(in) :: grid
    type(ChmState), intent(inout) :: chm
    type(DgnState), intent(inout) :: diag
    real(fp), intent(in) :: dt_s
    real(fp), intent(inout), target :: p1(:,:), p2(:,:), v(:,:,:)
    real(fp), intent(in), target :: u(:,:,:), xmass(:,:,:), ymass(:,:,:)
    real(fp), intent(in) :: area(:,:)
    real(fp), intent(inout) :: tracer(:,:,:,:)
    real(fp), intent(out) :: ps(:,:)
    integer :: ntracer, iq, l, k
    integer :: jfirst, jlast, ng, mg, n_adj, rc
    integer, parameter :: iord = 3, jord = 3, kord = 7
    logical, parameter :: lfill = .true.
    real(fp), allocatable :: ak(:), bk(:), area_y(:), ymid_r(:), p_temp(:,:)
    real(fp), pointer :: p_uwnd(:,:,:), p_vwnd(:,:,:), p_xmass(:,:,:), p_ymass(:,:,:)

    ntracer = size(tracer, 4)
    allocate(chm%Species(ntracer))
    do iq = 1, ntracer
       allocate(chm%Species(iq)%Conc(grid%NX, grid%NY, grid%NZ))
       chm%Species(iq)%Conc = tracer(:,:,:,iq)
       chm%Species(iq)%Units = 0
       chm%Species(iq)%Previous_Units = 0
    enddo
    chm%nSpecies = ntracer
    chm%nAdvect = ntracer
    chm%nTracer = ntracer

    diag%Archive_AdvFluxZonal = .false.
    diag%Archive_AdvFluxMerid = .false.
    diag%Archive_AdvFluxVert = .false.

    allocate(ak(grid%NZ+1), bk(grid%NZ+1), area_y(grid%NY), ymid_r(grid%NY), p_temp(grid%NX,grid%NY))
    do l = 1, grid%NZ+1
       k = (grid%NZ + 1) - l + 1
       ak(l) = GET_AP(k)
       bk(l) = GET_BP(k)
    enddo
    do k = 1, grid%NY
       area_y(k) = area(1,k)
       ymid_r(k) = grid%YMid_R(1,k)
    enddo

    ng = 0
    mg = 0
    n_adj = 0
    call INIT_TPCORE(grid%NX, grid%NY, grid%NZ, jfirst, jlast, ng, mg, dt_s, Re, ymid_r, .false., rc)
    if (rc /= 0) then
       write(*,*) 'INIT_TPCORE failed: ', rc
       stop 5
    endif

    p_uwnd  => u(:,:,grid%NZ:1:-1)
    p_vwnd  => v(:,:,grid%NZ:1:-1)
    p_xmass => xmass(:,:,grid%NZ:1:-1)
    p_ymass => ymass(:,:,grid%NZ:1:-1)

    call TPCORE_FVDAS( dt_s, Re, grid%NX, grid%NY, grid%NZ, jfirst, jlast, ng, mg, ntracer, &
                       ak, bk, p_uwnd, p_vwnd, p1, p2, p_temp, iord, jord, kord, n_adj,    &
                       p_xmass, p_ymass, lfill, area_y, chm, diag )
    ps = p_temp

    do iq = 1, ntracer
       tracer(:,:,:,iq) = chm%Species(iq)%Conc
    enddo

    p_uwnd => NULL()
    p_vwnd => NULL()
    p_xmass => NULL()
    p_ymass => NULL()
  end subroutine run_tpcore_step

  subroutine read_dimensions(ncid, nx, ny, nz, nilev, ntracer)
    integer, intent(in) :: ncid
    integer, intent(out) :: nx, ny, nz, nilev, ntracer
    nx = dim_len(ncid, 'lon')
    ny = dim_len(ncid, 'lat')
    nz = dim_len(ncid, 'lev')
    nilev = dim_len(ncid, 'ilev')
    ntracer = optional_dim_len(ncid, 'tracer')
  end subroutine read_dimensions

  subroutine read_fixture(ncid, lon, lat, hyai, hybi, area, p1, p2, u, v, dt_s)
    integer, intent(in) :: ncid
    real(fp), intent(out) :: lon(:), lat(:), hyai(:), hybi(:)
    real(fp), intent(out) :: area(:,:), p1(:,:), p2(:,:), u(:,:,:), v(:,:,:)
    real(fp), intent(out) :: dt_s
    call get_var_1d(ncid, 'lon', lon)
    call get_var_1d(ncid, 'lat', lat)
    call get_var_1d(ncid, 'hyai', hyai)
    call get_var_1d(ncid, 'hybi', hybi)
    call get_var_2d(ncid, 'area_m2', area)
    call get_var_2d(ncid, 'p1_hpa', p1)
    call get_var_2d(ncid, 'p2_hpa', p2)
    call get_var_3d(ncid, 'u_m_s', u)
    call get_var_3d(ncid, 'v_m_s', v)
    call check(nf90_get_att(ncid, nf90_global, 'dt_s', dt_s), 'read dt_s')
  end subroutine read_fixture

  subroutine write_output(path, xmass, ymass, has_tracers, tracer, ps)
    character(len=*), intent(in) :: path
    real(fp), intent(in) :: xmass(:,:,:), ymass(:,:,:)
    logical, intent(in) :: has_tracers
    real(fp), intent(in) :: tracer(:,:,:,:), ps(:,:)
    integer :: ncid, lon_dim, lat_dim, lev_dim, tracer_dim
    integer :: x_id, y_id, tracer_id, ps_id
    call check(nf90_create(path, nf90_clobber, ncid), 'create output')
    call check(nf90_def_dim(ncid, 'lon', size(xmass,1), lon_dim), 'def lon')
    call check(nf90_def_dim(ncid, 'lat', size(xmass,2), lat_dim), 'def lat')
    call check(nf90_def_dim(ncid, 'lev', size(xmass,3), lev_dim), 'def lev')
    if (has_tracers) then
       call check(nf90_def_dim(ncid, 'tracer', size(tracer,4), tracer_dim), 'def tracer')
       call check(nf90_put_att(ncid, nf90_global, 'harness', 'transport-step-output-v1'), 'put harness')
    else
       call check(nf90_put_att(ncid, nf90_global, 'harness', 'pjc-pfix-output-v1'), 'put harness')
    endif
    call check(nf90_def_var(ncid, 'xmass_hpa', nf90_double, (/ lon_dim, lat_dim, lev_dim /), x_id), 'def xmass')
    call check(nf90_def_var(ncid, 'ymass_hpa', nf90_double, (/ lon_dim, lat_dim, lev_dim /), y_id), 'def ymass')
    if (has_tracers) then
       call check(nf90_def_var(ncid, 'tracer_conc_after', nf90_double, (/ lon_dim, lat_dim, lev_dim, tracer_dim /), &
                               tracer_id), 'def tracer_conc_after')
       call check(nf90_def_var(ncid, 'surface_pressure_hpa', nf90_double, (/ lon_dim, lat_dim /), ps_id), 'def ps')
    endif
    call check(nf90_enddef(ncid), 'enddef')
    call check(nf90_put_var(ncid, x_id, xmass), 'write xmass')
    call check(nf90_put_var(ncid, y_id, ymass), 'write ymass')
    if (has_tracers) then
       call check(nf90_put_var(ncid, tracer_id, tracer), 'write tracer_conc_after')
       call check(nf90_put_var(ncid, ps_id, ps), 'write ps')
    endif
    call check(nf90_close(ncid), 'close output')
  end subroutine write_output

  function dim_len(ncid, name) result(length)
    integer, intent(in) :: ncid
    character(len=*), intent(in) :: name
    integer :: dimid, length
    call check(nf90_inq_dimid(ncid, name, dimid), 'inq dim '//trim(name))
    call check(nf90_inquire_dimension(ncid, dimid, len=length), 'inq len '//trim(name))
  end function dim_len

  function optional_dim_len(ncid, name) result(length)
    integer, intent(in) :: ncid
    character(len=*), intent(in) :: name
    integer :: dimid, length, status
    status = nf90_inq_dimid(ncid, name, dimid)
    if (status == nf90_noerr) then
       call check(nf90_inquire_dimension(ncid, dimid, len=length), 'inq len '//trim(name))
    else
       length = 0
    endif
  end function optional_dim_len

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
       stop 4
    endif
  end subroutine check

end program pjc_pfix_harness
