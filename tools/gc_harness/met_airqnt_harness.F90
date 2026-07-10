program met_airqnt_harness
  use netcdf
  use precision_mod, only: fp
  use input_opt_mod, only: OptInput
  use state_chm_mod, only: ChmState
  use state_grid_mod, only: GrdState
  use state_met_mod, only: MetState
  use pressure_mod, only: INIT_PRESSURE, SET_FLOATING_PRESSURES
  use calc_met_mod, only: AIRQNT, AVGPOLE, INTERP, SET_DRY_SURFACE_PRESSURE
  implicit none

  type(OptInput) :: Input_Opt
  type(ChmState) :: State_Chm
  type(GrdState) :: State_Grid
  type(MetState) :: State_Met
  character(len=1024) :: input_path, output_path
  integer :: ncid, nx, ny, nz, nilev, ntime0, ntime1, ntdt, rc
  real(fp), allocatable :: lon(:), lat(:), area(:,:)

  if (command_argument_count() /= 2) then
     write(*,*) 'usage: met_airqnt_harness INPUT.nc OUTPUT.nc'
     stop 2
  endif
  call get_command_argument(1, input_path)
  call get_command_argument(2, output_path)

  call check(nf90_open(trim(input_path), nf90_nowrite, ncid), 'open input')
  nx = dim_len(ncid, 'lon')
  ny = dim_len(ncid, 'lat')
  nz = dim_len(ncid, 'lev')
  nilev = dim_len(ncid, 'ilev')
  if (nilev /= nz + 1) stop 'ilev must equal lev + 1'

  allocate(lon(nx), lat(ny), area(nx,ny))
  call allocate_state(State_Grid, State_Met, nx, ny, nz)
  call read_fixture(ncid, lon, lat, area, State_Met, ntime0, ntime1, ntdt)
  call check(nf90_close(ncid), 'close input')

  call init_grid(State_Grid, nx, ny, nz, lon, lat, area)
  call INIT_PRESSURE(Input_Opt, State_Grid, rc)
  if (rc /= 0) stop 'INIT_PRESSURE failed'

  call SET_DRY_SURFACE_PRESSURE(State_Grid, State_Met, 1)
  call SET_DRY_SURFACE_PRESSURE(State_Grid, State_Met, 2)
  call AVGPOLE(State_Grid, State_Met%PS1_DRY)
  call AVGPOLE(State_Grid, State_Met%PS2_DRY)
  call AVGPOLE(State_Grid, State_Met%PS1_WET)
  call AVGPOLE(State_Grid, State_Met%PS2_WET)
  call INTERP(ntime0, ntime1, ntdt, Input_Opt, State_Grid, State_Met)
  call SET_FLOATING_PRESSURES(State_Grid, State_Met, rc)
  if (rc /= 0) stop 'SET_FLOATING_PRESSURES failed'
  call AIRQNT(Input_Opt, State_Chm, State_Grid, State_Met, rc, Update_Mixing_Ratio=.false.)
  if (rc /= 0) stop 'AIRQNT failed'

  call write_output(trim(output_path), State_Met)

contains

  subroutine allocate_state(grid, met, nx, ny, nz)
    type(GrdState), intent(inout) :: grid
    type(MetState), intent(inout) :: met
    integer, intent(in) :: nx, ny, nz
    allocate(grid%Area_M2(nx,ny), grid%XMid(nx,ny), grid%YMid(nx,ny), grid%YMid_R(nx,ny))
    allocate(grid%XEdge(nx+1,ny), grid%YEdge(nx,ny+1), grid%YEdge_R(nx,ny+1), grid%YSIN(nx,ny+1))
    allocate(met%PS1_WET(nx,ny), met%PS2_WET(nx,ny), met%PSC2_WET(nx,ny))
    allocate(met%PS1_DRY(nx,ny), met%PS2_DRY(nx,ny), met%PSC2_DRY(nx,ny))
    allocate(met%SPHU1(nx,ny,nz), met%SPHU2(nx,ny,nz), met%SPHU(nx,ny,nz), met%SPHU_PREV(nx,ny,nz))
    allocate(met%TMPU1(nx,ny,nz), met%TMPU2(nx,ny,nz), met%T(nx,ny,nz))
    allocate(met%PEDGE(nx,ny,nz+1), met%PEDGE_DRY(nx,ny,nz+1))
    allocate(met%PMID(nx,ny,nz), met%PMID_DRY(nx,ny,nz), met%DELP(nx,ny,nz), met%DELP_DRY(nx,ny,nz))
    allocate(met%THETA(nx,ny,nz), met%TV(nx,ny,nz), met%AVGW(nx,ny,nz))
    allocate(met%BXHEIGHT(nx,ny,nz), met%AIRVOL(nx,ny,nz), met%AD(nx,ny,nz), met%DP_DRY_PREV(nx,ny,nz))
    allocate(met%AIRDEN(nx,ny,nz), met%AIRNUMDEN(nx,ny,nz), met%MAIRDEN(nx,ny,nz))
    allocate(met%TROPP(nx,ny), met%TropLev(nx,ny), met%TropHt(nx,ny), met%ChemGridLev(nx,ny))
    allocate(met%FRLAND(nx,ny), met%FRSNOW(nx,ny), met%FRLAKE(nx,ny))
    allocate(met%FROCEAN(nx,ny), met%FRSEAICE(nx,ny), met%FRLANDICE(nx,ny))
    allocate(met%IsLand(nx,ny), met%IsWater(nx,ny), met%IsIce(nx,ny), met%IsSnow(nx,ny))
    allocate(met%LocalSolarTime(nx,ny), met%IsLocalNoon(nx,ny))
    allocate(met%InChemGrid(nx,ny,nz), met%InStratMeso(nx,ny,nz))
    allocate(met%InStratosphere(nx,ny,nz), met%InTroposphere(nx,ny,nz))
    met%PS1_DRY = 0.0_fp
    met%PS2_DRY = 0.0_fp
    met%PSC2_DRY = 0.0_fp
    met%PSC2_WET = 0.0_fp
    met%SPHU = 0.0_fp
    met%SPHU_PREV = 0.0_fp
    met%T = 0.0_fp
    met%PEDGE = 0.0_fp
    met%PEDGE_DRY = 0.0_fp
    met%PMID = 0.0_fp
    met%PMID_DRY = 0.0_fp
    met%DELP = 0.0_fp
    met%DELP_DRY = 0.0_fp
    met%DP_DRY_PREV = 0.0_fp
    met%THETA = 0.0_fp
    met%TV = 0.0_fp
    met%AVGW = 0.0_fp
    met%BXHEIGHT = 0.0_fp
    met%AIRVOL = 0.0_fp
    met%AD = 0.0_fp
    met%AIRDEN = 0.0_fp
    met%AIRNUMDEN = 0.0_fp
    met%MAIRDEN = 0.0_fp
    met%TROPP = 200.0_fp
    met%TropLev = 0
    met%TropHt = 0.0_fp
    met%ChemGridLev = 0
    met%FRLAND = 1.0_fp
    met%FRSNOW = 0.0_fp
    met%FRLAKE = 0.0_fp
    met%FROCEAN = 0.0_fp
    met%FRSEAICE = 0.0_fp
    met%FRLANDICE = 0.0_fp
    met%IsLand = .false.
    met%IsWater = .false.
    met%IsIce = .false.
    met%IsSnow = .false.
    met%LocalSolarTime = 0.0_fp
    met%IsLocalNoon = .false.
    met%InChemGrid = .false.
    met%InStratMeso = .false.
    met%InStratosphere = .false.
    met%InTroposphere = .false.
  end subroutine allocate_state

  subroutine init_grid(grid, nx, ny, nz, lon, lat, area)
    type(GrdState), intent(inout) :: grid
    integer, intent(in) :: nx, ny, nz
    real(fp), intent(in) :: lon(nx), lat(ny), area(nx,ny)
    integer :: i, j
    real(fp) :: dy

    grid%GridRes = '2.0x2.5'
    grid%NX = nx
    grid%NY = ny
    grid%NZ = nz
    grid%GlobalNX = nx
    grid%GlobalNY = ny
    grid%NativeNZ = nz
    grid%MaxChemLev = nz
    grid%MaxStratLev = nz
    grid%MaxTropLev = nz
    grid%HalfPolar = .true.
    grid%NestedGrid = .false.
    grid%Area_M2 = area
    if (ny > 1) then
       dy = lat(2) - lat(1)
    else
       dy = 180.0_fp
    endif
    grid%DX = 360.0_fp / real(nx, fp)
    grid%DY = dy
    do j = 1, ny
    do i = 1, nx
       grid%XMid(i,j) = lon(i)
       grid%YMid(i,j) = lat(j)
       grid%YMid_R(i,j) = lat(j) * acos(-1.0_fp) / 180.0_fp
    enddo
    enddo
    grid%XEdge = 0.0_fp
    grid%YEdge = 0.0_fp
    do j = 1, ny + 1
    do i = 1, nx
       if (j == 1) then
          grid%YEdge(i,j) = -90.0_fp
       else if (j == ny + 1) then
          grid%YEdge(i,j) = 90.0_fp
       else
          grid%YEdge(i,j) = 0.5_fp * (lat(j-1) + lat(j))
       endif
       grid%YEdge_R(i,j) = grid%YEdge(i,j) * acos(-1.0_fp) / 180.0_fp
       grid%YSIN(i,j) = sin(grid%YEdge_R(i,j))
    enddo
    enddo
  end subroutine init_grid

  subroutine read_fixture(ncid, lon, lat, area, met, ntime0, ntime1, ntdt)
    integer, intent(in) :: ncid
    real(fp), intent(out) :: lon(:), lat(:), area(:,:)
    type(MetState), intent(inout) :: met
    integer, intent(out) :: ntime0, ntime1, ntdt
    call get_var_1d(ncid, 'lon', lon)
    call get_var_1d(ncid, 'lat', lat)
    call get_var_2d(ncid, 'area_m2', area)
    call get_var_2d(ncid, 'ps1_wet_hpa', met%PS1_WET)
    call get_var_2d(ncid, 'ps2_wet_hpa', met%PS2_WET)
    call get_var_3d(ncid, 'sphu1_kg_kg', met%SPHU1)
    call get_var_3d(ncid, 'sphu2_kg_kg', met%SPHU2)
    met%SPHU1 = met%SPHU1 * 1.0e3_fp
    met%SPHU2 = met%SPHU2 * 1.0e3_fp
    call get_var_3d(ncid, 'tmpu1_k', met%TMPU1)
    call get_var_3d(ncid, 'tmpu2_k', met%TMPU2)
    call check(nf90_get_att(ncid, nf90_global, 'ntime0_s', ntime0), 'read ntime0_s')
    call check(nf90_get_att(ncid, nf90_global, 'ntime1_s', ntime1), 'read ntime1_s')
    call check(nf90_get_att(ncid, nf90_global, 'ntdt_s', ntdt), 'read ntdt_s')
  end subroutine read_fixture

  subroutine write_output(path, met)
    character(len=*), intent(in) :: path
    type(MetState), intent(in) :: met
    integer :: ncid, lon_dim, lat_dim, lev_dim, ilev_dim
    integer :: ps1w_id, ps2w_id, ps1d_id, ps2d_id, pscw_id, pscd_id
    integer :: pedge_id, pmid_id, delp_id, pedged_id, pmidd_id, delpd_id
    integer :: sphu_id, temp_id, avgw_id, tv_id, bxh_id, ad_id, airvol_id
    call check(nf90_create(path, nf90_clobber, ncid), 'create output')
    call check(nf90_def_dim(ncid, 'lon', size(met%PS1_WET,1), lon_dim), 'def lon')
    call check(nf90_def_dim(ncid, 'lat', size(met%PS1_WET,2), lat_dim), 'def lat')
    call check(nf90_def_dim(ncid, 'lev', size(met%DELP,3), lev_dim), 'def lev')
    call check(nf90_def_dim(ncid, 'ilev', size(met%PEDGE,3), ilev_dim), 'def ilev')
    call check(nf90_put_att(ncid, nf90_global, 'harness', 'met-airqnt-output-v1'), 'put harness')
    call check(nf90_def_var(ncid, 'ps1_wet_hpa', nf90_double, (/ lon_dim, lat_dim /), ps1w_id), 'def ps1 wet')
    call check(nf90_def_var(ncid, 'ps2_wet_hpa', nf90_double, (/ lon_dim, lat_dim /), ps2w_id), 'def ps2 wet')
    call check(nf90_def_var(ncid, 'ps1_dry_hpa', nf90_double, (/ lon_dim, lat_dim /), ps1d_id), 'def ps1 dry')
    call check(nf90_def_var(ncid, 'ps2_dry_hpa', nf90_double, (/ lon_dim, lat_dim /), ps2d_id), 'def ps2 dry')
    call check(nf90_def_var(ncid, 'psc2_wet_hpa', nf90_double, (/ lon_dim, lat_dim /), pscw_id), 'def psc wet')
    call check(nf90_def_var(ncid, 'psc2_dry_hpa', nf90_double, (/ lon_dim, lat_dim /), pscd_id), 'def psc dry')
    call check(nf90_def_var(ncid, 'wet_pressure_edges_hpa', nf90_double, (/ lon_dim, lat_dim, ilev_dim /), pedge_id), 'def pedge')
    call check(nf90_def_var(ncid, 'wet_pressure_mid_hpa', nf90_double, (/ lon_dim, lat_dim, lev_dim /), pmid_id), 'def pmid')
    call check(nf90_def_var(ncid, 'wet_pressure_thickness_hpa', nf90_double, (/ lon_dim, lat_dim, lev_dim /), delp_id), 'def delp')
    call check(nf90_def_var(ncid, 'dry_partial_pressure_edges_hpa', nf90_double, &
         (/ lon_dim, lat_dim, ilev_dim /), pedged_id), 'def pedge dry')
    call check(nf90_def_var(ncid, 'dry_partial_pressure_mid_hpa', nf90_double, &
         (/ lon_dim, lat_dim, lev_dim /), pmidd_id), 'def pmid dry')
    call check(nf90_def_var(ncid, 'delp_dry_hpa', nf90_double, (/ lon_dim, lat_dim, lev_dim /), delpd_id), 'def delp dry')
    call check(nf90_def_var(ncid, 'specific_humidity_kg_kg', nf90_double, (/ lon_dim, lat_dim, lev_dim /), sphu_id), 'def sphu')
    call check(nf90_def_var(ncid, 'temperature_k', nf90_double, (/ lon_dim, lat_dim, lev_dim /), temp_id), 'def temp')
    call check(nf90_def_var(ncid, 'water_vapor_vv_dry', nf90_double, (/ lon_dim, lat_dim, lev_dim /), avgw_id), 'def avgw')
    call check(nf90_def_var(ncid, 'virtual_temperature_k', nf90_double, (/ lon_dim, lat_dim, lev_dim /), tv_id), 'def tv')
    call check(nf90_def_var(ncid, 'bxheight_m', nf90_double, (/ lon_dim, lat_dim, lev_dim /), bxh_id), 'def bxheight')
    call check(nf90_def_var(ncid, 'dry_air_mass_kg', nf90_double, (/ lon_dim, lat_dim, lev_dim /), ad_id), 'def ad')
    call check(nf90_def_var(ncid, 'air_volume_m3', nf90_double, (/ lon_dim, lat_dim, lev_dim /), airvol_id), 'def airvol')
    call check(nf90_enddef(ncid), 'enddef output')
    call check(nf90_put_var(ncid, ps1w_id, met%PS1_WET), 'put ps1 wet')
    call check(nf90_put_var(ncid, ps2w_id, met%PS2_WET), 'put ps2 wet')
    call check(nf90_put_var(ncid, ps1d_id, met%PS1_DRY), 'put ps1 dry')
    call check(nf90_put_var(ncid, ps2d_id, met%PS2_DRY), 'put ps2 dry')
    call check(nf90_put_var(ncid, pscw_id, met%PSC2_WET), 'put psc wet')
    call check(nf90_put_var(ncid, pscd_id, met%PSC2_DRY), 'put psc dry')
    call check(nf90_put_var(ncid, pedge_id, met%PEDGE), 'put pedge')
    call check(nf90_put_var(ncid, pmid_id, met%PMID), 'put pmid')
    call check(nf90_put_var(ncid, delp_id, met%DELP), 'put delp')
    call check(nf90_put_var(ncid, pedged_id, met%PEDGE_DRY), 'put pedge dry')
    call check(nf90_put_var(ncid, pmidd_id, met%PMID_DRY), 'put pmid dry')
    call check(nf90_put_var(ncid, delpd_id, met%DELP_DRY), 'put delp dry')
    call check(nf90_put_var(ncid, sphu_id, met%SPHU * 1.0e-3_fp), 'put sphu')
    call check(nf90_put_var(ncid, temp_id, met%T), 'put temp')
    call check(nf90_put_var(ncid, avgw_id, met%AVGW), 'put avgw')
    call check(nf90_put_var(ncid, tv_id, met%TV), 'put tv')
    call check(nf90_put_var(ncid, bxh_id, met%BXHEIGHT), 'put bxheight')
    call check(nf90_put_var(ncid, ad_id, met%AD), 'put ad')
    call check(nf90_put_var(ncid, airvol_id, met%AIRVOL), 'put airvol')
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

  subroutine check(status, context)
    integer, intent(in) :: status
    character(len=*), intent(in) :: context
    if (status /= nf90_noerr) then
       write(*,*) trim(context), ': ', trim(nf90_strerror(status))
       stop 10
    endif
  end subroutine check

end program met_airqnt_harness
