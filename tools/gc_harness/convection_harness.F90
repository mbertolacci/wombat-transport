program convection_harness
  use netcdf
  use precision_mod, only: fp
  use input_opt_mod, only: OptInput
  use species_mod, only: Species
  use state_grid_mod, only: GrdState
  use state_chm_mod, only: ChmState
  use state_diag_mod, only: DgnState
  use state_met_mod, only: MetState
  use convection_mod, only: DO_CLOUD_CONVECTION
  implicit none

  real(fp), parameter :: G0_100 = 100.0_fp / 9.80665_fp
  type(OptInput) :: Input_Opt
  type(GrdState) :: State_Grid
  type(ChmState) :: State_Chm
  type(DgnState) :: State_Diag
  type(MetState) :: State_Met
  character(len=1024) :: input_path, output_path
  integer :: ncid, nx, ny, nz, ntracer, rc, i, j
  real(fp) :: dt_s
  real(fp), allocatable :: lon(:), lat(:), area(:,:)
  real(fp), allocatable :: tracer0(:,:,:,:), tracer(:,:,:,:), cmfmc_upper(:,:,:)
  real(fp), allocatable :: fsol(:,:), diag14_col(:,:), diag38(:,:), diag14_out(:,:,:,:)

  if (command_argument_count() /= 2) then
     write(*,*) 'usage: convection_harness INPUT.nc OUTPUT.nc'
     stop 2
  endif
  call get_command_argument(1, input_path)
  call get_command_argument(2, output_path)

  call check(nf90_open(trim(input_path), nf90_nowrite, ncid), 'open input')
  nx = dim_len(ncid, 'lon')
  ny = dim_len(ncid, 'lat')
  nz = dim_len(ncid, 'lev')
  ntracer = dim_len(ncid, 'tracer')

  allocate(lon(nx), lat(ny), area(nx,ny))
  allocate(tracer0(nx,ny,nz,ntracer), tracer(nx,ny,nz,ntracer), cmfmc_upper(nx,ny,nz))
  allocate(fsol(nz,ntracer), diag14_col(nz,ntracer), diag38(nz,max(1,ntracer)))
  allocate(diag14_out(nx,ny,nz,ntracer))
  call allocate_state(State_Grid, State_Chm, State_Met, nx, ny, nz, ntracer)
  call read_fixture(ncid, lon, lat, area, tracer, cmfmc_upper, State_Met, dt_s)
  call check(nf90_close(ncid), 'close input')
  tracer0 = tracer

  call init_grid(State_Grid, nx, ny, nz, lon, lat, area)
  call init_options(Input_Opt, int(dt_s))
  call init_chem(State_Chm, tracer, ntracer)
  State_Met%CMFMC(:,:,2:nz+1) = cmfmc_upper
  State_Diag%Archive_CloudConvFlux = .false.
  State_Diag%Archive_WetLossConv = .false.
  State_Diag%Archive_SatDiagnWetLossConv = .false.
  fsol = 0.0_fp
  diag14_out = 0.0_fp

  do j = 1, ny
  do i = 1, nx
     call DO_CLOUD_CONVECTION( Input_Opt, State_Chm, State_Diag, State_Grid, State_Met, &
          i, j, State_Grid%Area_M2(i,j), fsol, dt_s, .true., diag14_col, .false., diag38, rc )
     if (rc /= 0) stop 'DO_CLOUD_CONVECTION failed'
     diag14_out(i,j,:,:) = diag14_col
  enddo
  enddo

  call copy_tracers_from_state(State_Chm, tracer)
  call write_output(trim(output_path), tracer0, tracer, State_Met, area, diag14_out, dt_s)

contains

  subroutine allocate_state(grid, chm, met, nx, ny, nz, ntracer)
    type(GrdState), intent(inout) :: grid
    type(ChmState), intent(inout) :: chm
    type(MetState), intent(inout) :: met
    integer, intent(in) :: nx, ny, nz, ntracer
    integer :: n
    allocate(met%BXHEIGHT(nx,ny,nz), met%CMFMC(nx,ny,nz+1), met%DQRCU(nx,ny,nz))
    allocate(met%DTRAIN(nx,ny,nz), met%PFICU(nx,ny,nz+1), met%PFLCU(nx,ny,nz+1))
    allocate(met%REEVAPCN(nx,ny,nz), met%DELP_DRY(nx,ny,nz), met%DELP(nx,ny,nz))
    allocate(met%T(nx,ny,nz), met%PRECCON(nx,ny))
    allocate(grid%Area_M2(nx,ny), grid%XMid(nx,ny), grid%YMid(nx,ny), grid%YMid_R(nx,ny))
    allocate(chm%Map_Advect(ntracer), chm%Map_Tracer(ntracer), chm%Species(ntracer), chm%SpcData(ntracer))
    allocate(chm%H2O2AfterChem(nx,ny,nz), chm%SO2AfterChem(nx,ny,nz))
    met%CMFMC = 0.0_fp
    met%PFICU = 0.0_fp
    met%PFLCU = 0.0_fp
    chm%H2O2AfterChem = 0.0_fp
    chm%SO2AfterChem = 0.0_fp
    do n = 1, ntracer
       allocate(chm%Species(n)%Conc(nx,ny,nz))
       allocate(chm%SpcData(n)%Info)
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
    opt%LCONV = .true.
    opt%ITS_A_MERCURY_SIM = .false.
    opt%Reconstruct_Conv_Precip_Flux = .false.
    opt%MetField = 'MERRA2'
    opt%TS_CONV = dt
    opt%TS_DYN = dt
  end subroutine init_options

  subroutine init_chem(chm, tracer, ntracer)
    type(ChmState), intent(inout) :: chm
    real(fp), intent(in) :: tracer(:,:,:,:)
    integer, intent(in) :: ntracer
    integer :: n
    chm%nSpecies = ntracer
    chm%nAdvect = ntracer
    chm%nTracer = ntracer
    chm%nWetDep = 0
    do n = 1, ntracer
       chm%Map_Advect(n) = n
       chm%Map_Tracer(n) = n
       chm%Species(n)%Conc = tracer(:,:,:,n)
       chm%SpcData(n)%Info%ModelId = n
       chm%SpcData(n)%Info%AdvectId = n
       chm%SpcData(n)%Info%TracerId = n
       chm%SpcData(n)%Info%WetDepId = 0
       write(chm%SpcData(n)%Info%Name, '(A,I3.3)') 'conv_', n
       chm%SpcData(n)%Info%Is_Advected = .true.
       chm%SpcData(n)%Info%Is_Tracer = .true.
       chm%SpcData(n)%Info%Is_WetDep = .false.
       chm%SpcData(n)%Info%Is_Hg2 = .false.
       chm%SpcData(n)%Info%Is_HgP = .false.
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

  subroutine read_fixture(ncid, lon, lat, area, tracer, cmfmc_upper, met, dt_s)
    integer, intent(in) :: ncid
    real(fp), intent(out) :: lon(:), lat(:), area(:,:), tracer(:,:,:,:), cmfmc_upper(:,:,:), dt_s
    type(MetState), intent(inout) :: met
    real(fp), allocatable :: edge_tmp(:,:,:)
    call get_var_1d(ncid, 'lon', lon)
    call get_var_1d(ncid, 'lat', lat)
    call get_var_2d(ncid, 'area_m2', area)
    call get_var_4d(ncid, 'tracer_conc', tracer)
    call get_var_3d(ncid, 'cmfmc_kg_m2_s', cmfmc_upper)
    call get_var_3d(ncid, 'dtrain_kg_m2_s', met%DTRAIN)
    call get_var_3d(ncid, 'dqrcu_kg_kg_s', met%DQRCU)
    call get_var_3d(ncid, 'reevapcn_kg_kg_s', met%REEVAPCN)
    call get_var_3d(ncid, 'delp_dry_hpa', met%DELP_DRY)
    call get_var_3d(ncid, 'delp_hpa', met%DELP)
    call get_var_3d(ncid, 'bxheight_m', met%BXHEIGHT)
    call get_var_3d(ncid, 'temperature_k', met%T)
    allocate(edge_tmp(size(cmfmc_upper,1),size(cmfmc_upper,2),size(cmfmc_upper,3)))
    call get_var_3d(ncid, 'pficu_kg_m2_s', edge_tmp)
    met%PFICU(:,:,2:size(edge_tmp,3)+1) = edge_tmp
    call get_var_3d(ncid, 'pflcu_kg_m2_s', edge_tmp)
    met%PFLCU(:,:,2:size(edge_tmp,3)+1) = edge_tmp
    deallocate(edge_tmp)
    call get_var_2d(ncid, 'precccon_mm_day', met%PRECCON)
    call check(nf90_get_att(ncid, nf90_global, 'dt_s', dt_s), 'read dt_s')
  end subroutine read_fixture

  subroutine write_output(path, tracer0, tracer, met, area, diag14, dt_s)
    character(len=*), intent(in) :: path
    real(fp), intent(in) :: tracer0(:,:,:,:), tracer(:,:,:,:), area(:,:), diag14(:,:,:,:), dt_s
    type(MetState), intent(in) :: met
    integer :: ncid, lon_dim, lat_dim, lev_dim, tracer_dim
    integer :: tracer_id, diag_id, mass0_id, mass1_id
    real(fp) :: mass0(size(tracer,4)), mass1(size(tracer,4))
    real(fp) :: bmass(size(tracer,1),size(tracer,2),size(tracer,3))
    integer :: n
    bmass = met%DELP_DRY * G0_100
    do n = 1, size(tracer,4)
       mass0(n) = sum(tracer0(:,:,:,n) * bmass * spread(area, 3, size(tracer,3)))
       mass1(n) = sum(tracer(:,:,:,n) * bmass * spread(area, 3, size(tracer,3)))
    enddo
    call check(nf90_create(path, nf90_clobber, ncid), 'create output')
    call check(nf90_def_dim(ncid, 'lon', size(tracer,1), lon_dim), 'def lon')
    call check(nf90_def_dim(ncid, 'lat', size(tracer,2), lat_dim), 'def lat')
    call check(nf90_def_dim(ncid, 'lev', size(tracer,3), lev_dim), 'def lev')
    call check(nf90_def_dim(ncid, 'tracer', size(tracer,4), tracer_dim), 'def tracer')
    call check(nf90_put_att(ncid, nf90_global, 'harness', 'convection-output-v1'), 'put harness')
    call check(nf90_put_att(ncid, nf90_global, 'negative_count_before', count(tracer0 < 0.0_fp)), 'put neg before')
    call check(nf90_put_att(ncid, nf90_global, 'negative_count_after', count(tracer < 0.0_fp)), 'put neg after')
    call check(nf90_put_att(ncid, nf90_global, 'internal_steps', max(int(dt_s) / 300, 1)), 'put steps')
    call check(nf90_put_att(ncid, nf90_global, 'internal_dt_s', dt_s / real(max(int(dt_s) / 300, 1), fp)), 'put idt')
    call check(nf90_def_var(ncid, 'tracer_conc_after', nf90_double, &
         (/ lon_dim, lat_dim, lev_dim, tracer_dim /), tracer_id), 'def tracer')
    call check(nf90_def_var(ncid, 'diag14_mass_flux', nf90_double, &
         (/ lon_dim, lat_dim, lev_dim, tracer_dim /), diag_id), 'def diag')
    call check(nf90_def_var(ncid, 'initial_tracer_mass', nf90_double, (/ tracer_dim /), mass0_id), 'def mass0')
    call check(nf90_def_var(ncid, 'final_tracer_mass', nf90_double, (/ tracer_dim /), mass1_id), 'def mass1')
    call check(nf90_enddef(ncid), 'enddef output')
    call check(nf90_put_var(ncid, tracer_id, tracer), 'put tracer')
    call check(nf90_put_var(ncid, diag_id, diag14), 'put diag')
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

  subroutine check(status, label)
    integer, intent(in) :: status
    character(len=*), intent(in) :: label
    if (status /= nf90_noerr) then
       write(*,*) trim(label)//': '//trim(nf90_strerror(status))
       stop 1
    endif
  end subroutine check

end program convection_harness
