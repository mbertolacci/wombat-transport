program history_harness
  use precision_mod, only: fp
  use physconstants, only: airmw
  use input_opt_mod, only: OptInput, Set_Input_Opt
  use state_grid_mod, only: GrdState, Init_State_Grid
  use state_chm_mod, only: ChmState, Init_State_Chm
  use state_diag_mod, only: DgnState, Init_State_Diag
  use state_met_mod, only: MetState
  use diaglist_mod, only: DgnList, Init_DiagList, Cleanup_DiagList
  use taggeddiaglist_mod, only: TaggedDgnList, Init_TaggedDiagList, Cleanup_TaggedDiagList
  use pressure_mod, only: INIT_PRESSURE
  use grid_registry_mod, only: Init_Grid_Registry, Cleanup_Grid_Registry
  use diagnostics_mod, only: Set_Diagnostics_EndofTimestep
  use history_mod, only: History_Init, History_SetTime, History_Update, History_Write, History_Cleanup
  use unitconv_mod, only: KG_SPECIES_PER_KG_DRY_AIR
  implicit none

  type(OptInput) :: Input_Opt
  type(GrdState) :: State_Grid
  type(ChmState) :: State_Chm
  type(DgnState) :: State_Diag
  type(MetState) :: State_Met
  type(DgnList) :: Diag_List
  type(TaggedDgnList) :: TaggedDiag_List
  integer :: rc, nsteps, ntracer, dt_s, step
  character(len=1024) :: history_path, species_path

  if (command_argument_count() /= 4) then
     write(*,*) 'usage: history_harness HISTORY.rc species_database.yml NSTEPS DT_S'
     stop 2
  endif
  call get_command_argument(1, history_path)
  call get_command_argument(2, species_path)
  call read_int_arg(3, nsteps)
  call read_int_arg(4, dt_s)
  ntracer = count_species(trim(species_path))

  call Set_Input_Opt(.true., Input_Opt, rc)
  call assert_success(rc, 'Set_Input_Opt')
  call init_options(Input_Opt, trim(history_path), trim(species_path), nsteps, dt_s, ntracer)

  call Init_State_Grid(Input_Opt, State_Grid, rc)
  call assert_success(rc, 'Init_State_Grid')
  call init_grid(State_Grid)
  call INIT_PRESSURE(Input_Opt, State_Grid, rc)
  call assert_success(rc, 'INIT_PRESSURE')
  call Init_Grid_Registry(Input_Opt, State_Grid, rc)
  call assert_success(rc, 'Init_Grid_Registry')

  call Init_DiagList(Input_Opt%amIRoot, trim(history_path), Diag_List, rc)
  call assert_success(rc, 'Init_DiagList')
  call Init_TaggedDiagList(Input_Opt%amIRoot, Diag_List, TaggedDiag_List, rc)
  call assert_success(rc, 'Init_TaggedDiagList')

  call Init_State_Chm(Input_Opt, State_Chm, State_Grid, rc)
  call assert_success(rc, 'Init_State_Chm')
  call Init_State_Diag(Input_Opt, State_Chm, State_Grid, Diag_List, TaggedDiag_List, State_Diag, rc)
  call assert_success(rc, 'Init_State_Diag')

  call History_Init(Input_Opt, State_Met, State_Chm, State_Diag, State_Grid, rc)
  call assert_success(rc, 'History_Init')

  do step = 1, nsteps
     call fill_synthetic_species(State_Chm, step)
     call Set_Diagnostics_EndofTimestep(Input_Opt, State_Chm, State_Diag, State_Grid, State_Met, rc)
     call assert_success(rc, 'Set_Diagnostics_EndofTimestep')
     call History_SetTime(Input_Opt, rc)
     call assert_success(rc, 'History_SetTime')
     call History_Update(Input_Opt, State_Diag, rc)
     call assert_success(rc, 'History_Update')
     call History_Write(Input_Opt, State_Chm, State_Diag, rc)
     call assert_success(rc, 'History_Write')
  enddo

  call History_Cleanup(rc)
  call assert_success(rc, 'History_Cleanup')
  call Cleanup_Grid_Registry(rc)
  call assert_success(rc, 'Cleanup_Grid_Registry')
  call Cleanup_TaggedDiagList(TaggedDiag_List, rc)
  call assert_success(rc, 'Cleanup_TaggedDiagList')
  call Cleanup_DiagList(Diag_List, rc)
  call assert_success(rc, 'Cleanup_DiagList')

contains

  subroutine read_int_arg(index, value)
    integer, intent(in) :: index
    integer, intent(out) :: value
    character(len=64) :: raw
    integer :: ios
    call get_command_argument(index, raw)
    read(raw, *, iostat=ios) value
    if (ios /= 0) then
       write(*,*) 'invalid integer argument: ', trim(raw)
       stop 2
    endif
  end subroutine read_int_arg

  integer function count_species(path)
    character(len=*), intent(in) :: path
    character(len=512) :: line
    integer :: unit, ios
    count_species = 0
    open(newunit=unit, file=path, status='old', action='read', iostat=ios)
    if (ios /= 0) then
       write(*,*) 'could not open species database: ', trim(path)
       stop 2
    endif
    do
       read(unit, '(A)', iostat=ios) line
       if (ios /= 0) exit
       if (len_trim(line) == 0) cycle
       if (line(1:1) /= ' ' .and. index(line, ':') > 0) count_species = count_species + 1
    enddo
    close(unit)
  end function count_species

  subroutine init_options(opt, history_path, species_path, nsteps, dt_s, ntracer)
    type(OptInput), intent(inout) :: opt
    character(len=*), intent(in) :: history_path, species_path
    integer, intent(in) :: nsteps, dt_s, ntracer
    integer :: n
    opt%DryRun = .false.
    opt%Verbose = .false.
    opt%amIRoot = .true.
    opt%useTimers = .false.
    opt%HistoryInputFile = history_path
    opt%SpcDatabaseFile = species_path
    opt%SimulationName = 'history_harness'
    opt%MetField = 'MERRA2'
    opt%NymdB = 20140901
    opt%NhmsB = 0
    opt%NymdE = 20140902
    opt%NhmsE = 0
    opt%SimLengthSec = nsteps * dt_s
    opt%TS_DYN = dt_s
    opt%TS_CHEM = dt_s
    opt%ITS_A_TRACER_SIM = .true.
    opt%ITS_A_CO2_SIM = .false.
    opt%ITS_A_FULLCHEM_SIM = .false.
    opt%ITS_A_MERCURY_SIM = .false.
    opt%N_ADVECT = ntracer
    if (associated(opt%AdvectSpc_Name)) deallocate(opt%AdvectSpc_Name)
    allocate(opt%AdvectSpc_Name(ntracer))
    do n = 1, ntracer
       write(opt%AdvectSpc_Name(n), '(A,I3.3)') 'hist_', n
    enddo
  end subroutine init_options

  subroutine init_grid(grid)
    type(GrdState), intent(inout) :: grid
    integer :: i, j
    real(fp), parameter :: lon_values(4) = (/ -180.0_fp, -90.0_fp, 0.0_fp, 90.0_fp /)
    real(fp), parameter :: lat_values(3) = (/ -60.0_fp, 0.0_fp, 60.0_fp /)
    grid%GridRes = 'history-harness'
    grid%DX = 90.0_fp
    grid%DY = 60.0_fp
    grid%NX = 4
    grid%NY = 3
    grid%NZ = 47
    grid%GlobalNX = grid%NX
    grid%GlobalNY = grid%NY
    grid%NativeNZ = grid%NZ
    grid%XMin = -225.0_fp
    grid%XMax = 135.0_fp
    grid%YMin = -90.0_fp
    grid%YMax = 90.0_fp
    grid%HalfPolar = .false.
    grid%Center180 = .false.
    grid%NestedGrid = .false.
    grid%NorthBuffer = 0
    grid%SouthBuffer = 0
    grid%EastBuffer = 0
    grid%WestBuffer = 0
    allocate(grid%XMid(grid%NX,grid%NY), grid%YMid(grid%NX,grid%NY), grid%Area_M2(grid%NX,grid%NY))
    allocate(grid%XEdge(grid%NX+1,grid%NY), grid%YEdge(grid%NX,grid%NY+1))
    allocate(grid%YMid_R(grid%NX,grid%NY), grid%YEdge_R(grid%NX,grid%NY+1), grid%YSIN(grid%NX,grid%NY+1))
    do j = 1, grid%NY
       do i = 1, grid%NX
          grid%XMid(i,j) = lon_values(i)
          grid%YMid(i,j) = lat_values(j)
          grid%YMid_R(i,j) = lat_values(j) * acos(-1.0_fp) / 180.0_fp
          grid%Area_M2(i,j) = 1.0e10_fp
       enddo
    enddo
    grid%XEdge = 0.0_fp
    grid%YEdge = 0.0_fp
    grid%YEdge_R = 0.0_fp
    grid%YSIN = 0.0_fp
  end subroutine init_grid

  subroutine fill_synthetic_species(chm, step)
    type(ChmState), intent(inout) :: chm
    integer, intent(in) :: step
    integer :: i, j, l, n
    real(fp) :: vv
    do n = 1, chm%nSpecies
       do l = 1, size(chm%Species(n)%Conc, 3)
          do j = 1, size(chm%Species(n)%Conc, 2)
             do i = 1, size(chm%Species(n)%Conc, 1)
                vv = real(n * 1000 + step, fp) &
                   + real(l, fp) * 0.125_fp &
                   + real(j, fp) * 0.01_fp &
                   + real(i, fp) * 0.001_fp
                chm%Species(n)%Conc(i,j,l) = vv * chm%SpcData(n)%Info%MW_g / AIRMW
             enddo
          enddo
       enddo
       chm%Species(n)%Units = KG_SPECIES_PER_KG_DRY_AIR
       chm%Species(n)%Previous_Units = KG_SPECIES_PER_KG_DRY_AIR
    enddo
  end subroutine fill_synthetic_species

  subroutine assert_success(rc, label)
    integer, intent(in) :: rc
    character(len=*), intent(in) :: label
    if (rc /= 0) then
       write(*,*) trim(label), ' failed with rc=', rc
       stop 1
    endif
  end subroutine assert_success

end program history_harness
