module Tpcore_Trace_Mod
  use netcdf
  use precision_mod, only: fp
  implicit none
  private

  public :: Tpcore_Trace_Init
  public :: Tpcore_Trace_Enabled
  public :: Tpcore_Trace_Setup
  public :: Tpcore_Trace_Field2d
  public :: Tpcore_Trace_Field3d
  public :: Tpcore_Trace_Write

  logical, save :: enabled = .false.
  character(len=1024), save :: trace_path = ''
  real(fp), save :: trace_dt_s = 0.0_fp
  real(fp), allocatable, save :: delp1_hpa(:,:,:), delp2_hpa(:,:,:)
  real(fp), allocatable, save :: cx(:,:,:), cy(:,:,:), vertical_mass_flux_hpa(:,:,:)
  real(fp), allocatable, save :: surface_pressure_hpa(:,:)
  real(fp), allocatable, save :: q_after_pole_average(:,:,:,:)
  real(fp), allocatable, save :: dq_after_init_hpa(:,:,:,:)
  real(fp), allocatable, save :: q_after_cross_terms(:,:,:,:)
  real(fp), allocatable, save :: dq_after_xtp_hpa(:,:,:,:)
  real(fp), allocatable, save :: dq_after_ytp_hpa(:,:,:,:)
  real(fp), allocatable, save :: dq_after_fzppm_hpa(:,:,:,:)
  real(fp), allocatable, save :: dq_after_fill_hpa(:,:,:,:)
  real(fp), allocatable, save :: tracer_conc_after(:,:,:,:)

contains

  subroutine Tpcore_Trace_Init(path, im, jm, km, nq, dt_s)
    character(len=*), intent(in) :: path
    integer, intent(in) :: im, jm, km, nq
    real(fp), intent(in) :: dt_s

    trace_path = path
    trace_dt_s = dt_s
    enabled = len_trim(path) > 0 .and. nq > 0
    if (.not. enabled) return

    allocate(delp1_hpa(im,jm,km), delp2_hpa(im,jm,km))
    allocate(cx(im,jm,km), cy(im,jm,km), vertical_mass_flux_hpa(im,jm,km))
    allocate(surface_pressure_hpa(im,jm))
    allocate(q_after_pole_average(im,jm,km,nq))
    allocate(dq_after_init_hpa(im,jm,km,nq))
    allocate(q_after_cross_terms(im,jm,km,nq))
    allocate(dq_after_xtp_hpa(im,jm,km,nq))
    allocate(dq_after_ytp_hpa(im,jm,km,nq))
    allocate(dq_after_fzppm_hpa(im,jm,km,nq))
    allocate(dq_after_fill_hpa(im,jm,km,nq))
    allocate(tracer_conc_after(im,jm,km,nq))
    delp1_hpa = 0.0_fp
    delp2_hpa = 0.0_fp
    cx = 0.0_fp
    cy = 0.0_fp
    vertical_mass_flux_hpa = 0.0_fp
    surface_pressure_hpa = 0.0_fp
    q_after_pole_average = 0.0_fp
    dq_after_init_hpa = 0.0_fp
    q_after_cross_terms = 0.0_fp
    dq_after_xtp_hpa = 0.0_fp
    dq_after_ytp_hpa = 0.0_fp
    dq_after_fzppm_hpa = 0.0_fp
    dq_after_fill_hpa = 0.0_fp
    tracer_conc_after = 0.0_fp
  end subroutine Tpcore_Trace_Init

  logical function Tpcore_Trace_Enabled()
    Tpcore_Trace_Enabled = enabled
  end function Tpcore_Trace_Enabled

  subroutine Tpcore_Trace_Setup(delp1, delp2, courant_x, courant_y, wz, ps)
    real(fp), intent(in) :: delp1(:,:,:), delp2(:,:,:), courant_x(:,:,:), courant_y(:,:,:), wz(:,:,:), ps(:,:)
    integer :: km, k
    if (.not. enabled) return
    km = size(delp1, 3)
    do k = 1, km
       delp1_hpa(:,:,km-k+1) = delp1(:,:,k)
       delp2_hpa(:,:,km-k+1) = delp2(:,:,k)
       cx(:,:,km-k+1) = courant_x(:,:,k)
       cy(:,:,km-k+1) = courant_y(:,:,k)
       vertical_mass_flux_hpa(:,:,km-k+1) = wz(:,:,k)
    enddo
    surface_pressure_hpa = ps
  end subroutine Tpcore_Trace_Setup

  subroutine Tpcore_Trace_Field2d(stage, iq, k_project, values)
    character(len=*), intent(in) :: stage
    integer, intent(in) :: iq, k_project
    real(fp), intent(in) :: values(:,:)
    if (.not. enabled) return
    select case (trim(stage))
    case ('q_after_pole_average')
       q_after_pole_average(:,:,k_project,iq) = values
    case ('dq_after_init_hpa')
       dq_after_init_hpa(:,:,k_project,iq) = values
    case ('q_after_cross_terms')
       q_after_cross_terms(:,:,k_project,iq) = values
    case ('dq_after_xtp_hpa')
       dq_after_xtp_hpa(:,:,k_project,iq) = values
    case ('dq_after_ytp_hpa')
       dq_after_ytp_hpa(:,:,k_project,iq) = values
    case default
       write(*,*) 'unknown TPCORE trace 2-D stage: ', trim(stage)
       stop 11
    end select
  end subroutine Tpcore_Trace_Field2d

  subroutine Tpcore_Trace_Field3d(stage, iq, values)
    character(len=*), intent(in) :: stage
    integer, intent(in) :: iq
    real(fp), intent(in) :: values(:,:,:)
    integer :: km, k
    if (.not. enabled) return
    km = size(values, 3)
    select case (trim(stage))
    case ('dq_after_fzppm_hpa')
       do k = 1, km
          dq_after_fzppm_hpa(:,:,km-k+1,iq) = values(:,:,k)
       enddo
    case ('dq_after_fill_hpa')
       do k = 1, km
          dq_after_fill_hpa(:,:,km-k+1,iq) = values(:,:,k)
       enddo
    case ('tracer_conc_after')
       tracer_conc_after(:,:,:,iq) = values
    case default
       write(*,*) 'unknown TPCORE trace 3-D stage: ', trim(stage)
       stop 12
    end select
  end subroutine Tpcore_Trace_Field3d

  subroutine Tpcore_Trace_Write()
    integer :: ncid, lon_dim, lat_dim, lev_dim, tracer_dim
    integer :: id
    if (.not. enabled) return

    call check(nf90_create(trim(trace_path), nf90_clobber, ncid), 'create TPCORE trace')
    call check(nf90_def_dim(ncid, 'lon', size(delp1_hpa,1), lon_dim), 'def trace lon')
    call check(nf90_def_dim(ncid, 'lat', size(delp1_hpa,2), lat_dim), 'def trace lat')
    call check(nf90_def_dim(ncid, 'lev', size(delp1_hpa,3), lev_dim), 'def trace lev')
    call check(nf90_def_dim(ncid, 'tracer', size(q_after_pole_average,4), tracer_dim), 'def trace tracer')
    call check(nf90_put_att(ncid, nf90_global, 'harness', 'tpcore-trace-v1'), 'put trace harness')
    call check(nf90_put_att(ncid, nf90_global, 'dt_s', trace_dt_s), 'put trace dt_s')
    call def_var_3d(ncid, 'delp1_hpa', lon_dim, lat_dim, lev_dim, id)
    call def_var_3d(ncid, 'delp2_hpa', lon_dim, lat_dim, lev_dim, id)
    call def_var_3d(ncid, 'cx', lon_dim, lat_dim, lev_dim, id)
    call def_var_3d(ncid, 'cy', lon_dim, lat_dim, lev_dim, id)
    call def_var_3d(ncid, 'vertical_mass_flux_hpa', lon_dim, lat_dim, lev_dim, id)
    call def_var_2d(ncid, 'surface_pressure_hpa', lon_dim, lat_dim, id)
    call def_var_4d(ncid, 'q_after_pole_average', lon_dim, lat_dim, lev_dim, tracer_dim, id)
    call def_var_4d(ncid, 'dq_after_init_hpa', lon_dim, lat_dim, lev_dim, tracer_dim, id)
    call def_var_4d(ncid, 'q_after_cross_terms', lon_dim, lat_dim, lev_dim, tracer_dim, id)
    call def_var_4d(ncid, 'dq_after_xtp_hpa', lon_dim, lat_dim, lev_dim, tracer_dim, id)
    call def_var_4d(ncid, 'dq_after_ytp_hpa', lon_dim, lat_dim, lev_dim, tracer_dim, id)
    call def_var_4d(ncid, 'dq_after_fzppm_hpa', lon_dim, lat_dim, lev_dim, tracer_dim, id)
    call def_var_4d(ncid, 'dq_after_fill_hpa', lon_dim, lat_dim, lev_dim, tracer_dim, id)
    call def_var_4d(ncid, 'tracer_conc_after', lon_dim, lat_dim, lev_dim, tracer_dim, id)
    call check(nf90_enddef(ncid), 'enddef trace')

    call put_var_3d(ncid, 'delp1_hpa', delp1_hpa)
    call put_var_3d(ncid, 'delp2_hpa', delp2_hpa)
    call put_var_3d(ncid, 'cx', cx)
    call put_var_3d(ncid, 'cy', cy)
    call put_var_3d(ncid, 'vertical_mass_flux_hpa', vertical_mass_flux_hpa)
    call put_var_2d(ncid, 'surface_pressure_hpa', surface_pressure_hpa)
    call put_var_4d(ncid, 'q_after_pole_average', q_after_pole_average)
    call put_var_4d(ncid, 'dq_after_init_hpa', dq_after_init_hpa)
    call put_var_4d(ncid, 'q_after_cross_terms', q_after_cross_terms)
    call put_var_4d(ncid, 'dq_after_xtp_hpa', dq_after_xtp_hpa)
    call put_var_4d(ncid, 'dq_after_ytp_hpa', dq_after_ytp_hpa)
    call put_var_4d(ncid, 'dq_after_fzppm_hpa', dq_after_fzppm_hpa)
    call put_var_4d(ncid, 'dq_after_fill_hpa', dq_after_fill_hpa)
    call put_var_4d(ncid, 'tracer_conc_after', tracer_conc_after)
    call check(nf90_close(ncid), 'close trace')
  end subroutine Tpcore_Trace_Write

  subroutine def_var_2d(ncid, name, lon_dim, lat_dim, varid)
    integer, intent(in) :: ncid, lon_dim, lat_dim
    character(len=*), intent(in) :: name
    integer, intent(out) :: varid
    call check(nf90_def_var(ncid, name, nf90_double, (/ lon_dim, lat_dim /), varid), 'def '//trim(name))
  end subroutine def_var_2d

  subroutine def_var_3d(ncid, name, lon_dim, lat_dim, lev_dim, varid)
    integer, intent(in) :: ncid, lon_dim, lat_dim, lev_dim
    character(len=*), intent(in) :: name
    integer, intent(out) :: varid
    call check(nf90_def_var(ncid, name, nf90_double, (/ lon_dim, lat_dim, lev_dim /), varid), 'def '//trim(name))
  end subroutine def_var_3d

  subroutine def_var_4d(ncid, name, lon_dim, lat_dim, lev_dim, tracer_dim, varid)
    integer, intent(in) :: ncid, lon_dim, lat_dim, lev_dim, tracer_dim
    character(len=*), intent(in) :: name
    integer, intent(out) :: varid
    call check(nf90_def_var(ncid, name, nf90_double, (/ lon_dim, lat_dim, lev_dim, tracer_dim /), varid), &
               'def '//trim(name))
  end subroutine def_var_4d

  subroutine put_var_2d(ncid, name, values)
    integer, intent(in) :: ncid
    character(len=*), intent(in) :: name
    real(fp), intent(in) :: values(:,:)
    integer :: varid
    call check(nf90_inq_varid(ncid, name, varid), 'inq '//trim(name))
    call check(nf90_put_var(ncid, varid, values), 'put '//trim(name))
  end subroutine put_var_2d

  subroutine put_var_3d(ncid, name, values)
    integer, intent(in) :: ncid
    character(len=*), intent(in) :: name
    real(fp), intent(in) :: values(:,:,:)
    integer :: varid
    call check(nf90_inq_varid(ncid, name, varid), 'inq '//trim(name))
    call check(nf90_put_var(ncid, varid, values), 'put '//trim(name))
  end subroutine put_var_3d

  subroutine put_var_4d(ncid, name, values)
    integer, intent(in) :: ncid
    character(len=*), intent(in) :: name
    real(fp), intent(in) :: values(:,:,:,:)
    integer :: varid
    call check(nf90_inq_varid(ncid, name, varid), 'inq '//trim(name))
    call check(nf90_put_var(ncid, varid, values), 'put '//trim(name))
  end subroutine put_var_4d

  subroutine check(status, context)
    integer, intent(in) :: status
    character(len=*), intent(in) :: context
    if (status /= nf90_noerr) then
       write(*,*) trim(context), ': ', trim(nf90_strerror(status))
       stop 13
    endif
  end subroutine check

end module Tpcore_Trace_Mod
